package main

// turn_ca_file: an explicit CA bundle for `turns:` (TLS-over-TCP) TURN servers.
//
// WHY (measured 2026-10-08): pion/ice v4.4.4 dials a `turns:` server with
//   tls.Client(tcpConn, &tls.Config{ServerName: url.Host, InsecureSkipVerify: a.insecureSkipVerify})
// (gather.go:1186) -- no RootCAs, so the trust store is always the PLATFORM one. On Windows Go ignores
// SSL_CERT_FILE unless GODEBUG=x509sslcertoverrideplatform=1 (go.mod says `go 1.24.0`), so a relay
// with a private CA failed `x509: certificate signed by unknown authority` x4 with SSL_CERT_FILE alone
// and passed only with the extra GODEBUG -- a setting no operator should have to know about.
//
// pion exposes no TLS-config hook for TURN, but it does expose a proxy dialer (SettingEngine
// SetICEProxyDialer): for a TCP-transport TURN URL it then takes the stream from the dialer as-is and
// does NOT wrap it in TLS itself (gather.go:1090-1108, relayProtocol "tls" for a turns: URL). So the
// sidecar installs a dialer -- only when turn_ca_file is set -- that does the TLS handshake for the
// configured `turns:` addresses with the operator's CA added to the system pool, and a plain TCP dial for
// `turn:...?transport=tcp` addresses. Certificate verification is never skipped.
//
// Not covered: `turns:...?transport=udp` (DTLS), which pion dials itself (gather.go:1144) with no
// root-CA option; validate() refuses that combination instead of silently ignoring the CA file.

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"net"
	"os"
	"strconv"
	"time"

	"github.com/pion/stun/v4"
)

const turnDialTimeout = 10 * time.Second

// turnTLSDialer implements golang.org/x/net/proxy.Dialer.
type turnTLSDialer struct {
	pool    *x509.CertPool
	tlsHost map[string]string // "host:port" (pion's turnServerAddr) -> TLS ServerName, for turns: URLs
	tcp     map[string]bool   // "host:port" of turn:...?transport=tcp URLs (plain TCP)
	local   net.IP            // non-nil = bind_iface/bind_ip pin: every TURN TCP/TLS dial leaves from this address
}

func (d *turnTLSDialer) Dial(network, addr string) (net.Conn, error) {
	nd := &net.Dialer{Timeout: turnDialTimeout}
	if d.local != nil {
		nd.LocalAddr = &net.TCPAddr{IP: d.local}
	}
	if sn, ok := d.tlsHost[addr]; ok {
		return tls.DialWithDialer(nd, network, addr, &tls.Config{ServerName: sn, RootCAs: d.pool,
			MinVersion: tls.VersionTLS12})
	}
	if d.tcp[addr] {
		return nd.Dial(network, addr)
	}
	return nil, fmt.Errorf("turn dialer: %s is not a configured TURN TCP/TLS address", addr)
}

// turnAddrs splits the configured TURN URLs into TLS (turns: over TCP) and plain-TCP server addresses,
// keyed exactly as pion/ice builds turnServerAddr (net.JoinHostPort(url.Host, port), gather.go:1060).
func turnAddrs(urls []string) (tlsHost map[string]string, tcp map[string]bool, dtls bool, err error) {
	tlsHost, tcp = map[string]string{}, map[string]bool{}
	for _, raw := range urls {
		u, perr := stun.ParseURI(raw)
		if perr != nil {
			return nil, nil, false, fmt.Errorf("turn url %q: %v", raw, perr)
		}
		addr := net.JoinHostPort(u.Host, strconv.Itoa(u.Port))
		switch {
		case u.Scheme == stun.SchemeTypeTURNS && u.Proto == stun.ProtoTypeTCP:
			tlsHost[addr] = u.Host
		case u.Scheme == stun.SchemeTypeTURNS:
			dtls = true
		case u.Scheme == stun.SchemeTypeTURN && u.Proto == stun.ProtoTypeTCP:
			tcp[addr] = true
		}
	}
	for a := range tlsHost {
		if tcp[a] {
			return nil, nil, false, fmt.Errorf("turn address %s is configured as both turns: and turn:?transport=tcp; "+
				"one dialer cannot tell them apart", a)
		}
	}
	return tlsHost, tcp, dtls, nil
}

// newTURNTLSDialer loads the CA file (PEM) and returns the dialer, or an error that names the problem.
func newTURNTLSDialer(caFile string, urls []string) (*turnTLSDialer, error) {
	tlsHost, tcp, dtls, err := turnAddrs(urls)
	if err != nil {
		return nil, err
	}
	if dtls {
		return nil, errors.New("turn_ca_file does not apply to turns:...?transport=udp (DTLS): pion dials it " +
			"with the platform trust store only; use turns: over TCP")
	}
	if len(tlsHost) == 0 {
		return nil, errors.New("turn_ca_file is set but no turns: (TLS over TCP) URL is configured")
	}
	pem, err := os.ReadFile(caFile)
	if err != nil {
		return nil, fmt.Errorf("turn_ca_file: %v", err)
	}
	pool, err := x509.SystemCertPool()
	if err != nil || pool == nil {
		pool = x509.NewCertPool()
	}
	if !pool.AppendCertsFromPEM(pem) {
		return nil, fmt.Errorf("turn_ca_file %s holds no PEM certificate", caFile)
	}
	return &turnTLSDialer{pool: pool, tlsHost: tlsHost, tcp: tcp}, nil
}

// newTURNPinnedDialer: for a PINNED sidecar without turn_ca_file. pion would dial TCP/TLS TURN with a nil local
// address (gather.go:1110, 1177), i.e. via the default route, so the pinned sidecar dials them itself from the
// pinned address. TLS verification is the platform store (pool nil), exactly what pion does for turns:.
// Returns (nil, nil) when no TCP/TLS TURN URL is configured (nothing to dial; UDP TURN binds via the IP filter).
func newTURNPinnedDialer(urls []string, local net.IP) (*turnTLSDialer, error) {
	tlsHost, tcp, dtls, err := turnAddrs(urls)
	if err != nil {
		return nil, err
	}
	if dtls {
		return nil, errors.New("a pinned sidecar cannot dial turns:...?transport=udp (DTLS) from the pinned address")
	}
	if len(tlsHost) == 0 && len(tcp) == 0 {
		return nil, nil
	}
	return &turnTLSDialer{pool: nil, tlsHost: tlsHost, tcp: tcp, local: local}, nil
}
