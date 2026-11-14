package main

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/pion/ice/v4"
	"github.com/pion/logging"
	"github.com/pion/transport/v5"
	"github.com/pion/transport/v5/stdnet"
	"github.com/pion/webrtc/v4"
)

// Sidecar holds one node's state: certificate (its identity for the run), pins, peers, parts, fetches.
type Sidecar struct {
	cfg   *Config
	em    *Emitter
	relay *Relay
	seal  *sealer     // nil = plaintext records (loopback code test without a key)
	sign  *signer     // nil = unsigned records (no per-miner keys configured; sign.go)
	api   *webrtc.API // offerer side (and probes): plain sockets
	// apiAnswer: the ANSWERER's PeerConnections. Same settings, but its ICE sockets are TTL-primed (prime.go) so its
	// early checks open its own NAT without poisoning the offerer's. == api when priming is off.
	apiAnswer *webrtc.API
	netBase   transport.Net // test hook: the network ICE uses (nil = the OS stack)
	lowSend   lowSender     // test hook: how a primed socket sends a NAT-local packet (nil = IP TTL)
	cert      webrtc.Certificate
	fp        string // sha-256 fingerprint of cert, "AB:CD:..."

	rejMu    sync.Mutex
	rejected map[string]bool // sha of relay records that failed authentication (reported once each)

	ctx    context.Context
	cancel context.CancelFunc

	mu         sync.Mutex
	pins       map[string]string // peer node -> pinned fingerprint (first sight; never replaced)
	peers      map[string]*Peer
	parts      map[string]*Part // own registered parts, by sha hex
	remoteHave map[string]*Have // parts announced by peers, by sha hex
	fetches    map[string]*Fetch

	dataLn   net.Listener
	dataAddr string
	dataMu   sync.Mutex // guards dataConn
	dataConn net.Conn
	dataWMu  sync.Mutex // serializes whole frames on the data socket
}

func NewSidecar(cfg *Config, em *Emitter) (*Sidecar, error) {
	ctx, cancel := context.WithCancel(context.Background())
	sc := &Sidecar{cfg: cfg, em: em, ctx: ctx, cancel: cancel,
		pins: map[string]string{}, peers: map[string]*Peer{}, parts: map[string]*Part{},
		remoteHave: map[string]*Have{}, fetches: map[string]*Fetch{}, rejected: map[string]bool{}}
	if cfg.RelayURL != "" {
		sc.relay = NewRelay(cfg, em)
	}
	s, err := newSealer(cfg.SignalKey)
	if err != nil {
		return nil, err
	}
	sc.seal = s
	sg, err := newSigner(cfg.Node, cfg.SignKey, cfg.PeerKeys)
	if err != nil {
		return nil, err
	}
	sc.sign = sg
	// One long-lived certificate per run: its fingerprint is this node's identity (plan 5.2 item 1).
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, err
	}
	cert, err := webrtc.GenerateCertificate(key)
	if err != nil {
		return nil, err
	}
	sc.cert = *cert
	fps, err := cert.GetFingerprints()
	if err != nil || len(fps) == 0 || fps[0].Algorithm != "sha-256" {
		return nil, fmt.Errorf("certificate fingerprint: %v", err)
	}
	sc.fp = strings.ToUpper(fps[0].Value)

	if err := sc.buildAPIs(); err != nil {
		return nil, err
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	sc.dataLn = ln
	sc.dataAddr = ln.Addr().String()
	go sc.acceptData()
	return sc, nil
}

// buildAPIs makes sc.api (offerer) and sc.apiAnswer (answerer; TTL-primed when priming applies).
func (sc *Sidecar) buildAPIs() error {
	se, err := sc.settingEngine(false)
	if err != nil {
		return err
	}
	sc.api = webrtc.NewAPI(webrtc.WithSettingEngine(se))
	sc.apiAnswer = sc.api
	if primeTTL(sc.cfg) > 0 {
		sa, err := sc.settingEngine(true)
		if err != nil {
			return err
		}
		sc.apiAnswer = webrtc.NewAPI(webrtc.WithSettingEngine(sa))
	}
	return nil
}

// primeTTL is the answerer's priming TTL, or 0 when priming does not apply: off by config (ice_prime_ttl < 0), the
// loopback code test (no NAT), or relay-only (no direct path is tried).
func primeTTL(cfg *Config) int {
	if cfg.ICEPrimeTTL < 0 || cfg.Loopback || cfg.ICEPolicy == "relay" {
		return 0
	}
	if cfg.ICEPrimeTTL == 0 {
		return defaultICEPrimeTTL
	}
	return cfg.ICEPrimeTTL
}

// pionCheckInterval is pion/ice v4.4.4's connectivity-check tick while checking (agent_config.go:18
// defaultCheckInterval); the sidecar never changes it.
const pionCheckInterval = 200 * time.Millisecond

// iceMaxBindingRequests is how many binding requests ICE sends per candidate pair before marking it failed.
//
// 79e1986 raised this from pion's 7 (pion/ice v4.4.4 agent_config.go:52, ~1.4 s of checks) to the whole checking
// window (disconnected + failed timeouts = pion's initialCheckingTimeout, agent.go:741-752) on the theory that the two
// peers' check windows never overlapped. That theory was REFUTED on the real NATs the same morning (runs 3/4 failed
// with 201): the real cause is NAT poisoning by the answerer's early checks, fixed by prime.go. The long window is
// kept because the fix needs it: the primed answerer must still be checking (keeping its NAT open) when the
// offerer's first check arrives seconds later, and the offerer must keep checking until the answerer's flips to
// full TTL. Cost on a failing epoch: one ~100 B request per pair per 200 ms. cfg.ICEMaxBindingRequests > 0
// overrides (7 = pion's default).
func iceMaxBindingRequests(cfg *Config) uint16 {
	if cfg.ICEMaxBindingRequests > 0 {
		if cfg.ICEMaxBindingRequests > 65535 {
			return 65535
		}
		return uint16(cfg.ICEMaxBindingRequests)
	}
	window := secs(cfg.ICEDisconnectedS) + secs(cfg.ICEFailedS)
	n := int64(window/pionCheckInterval) + 1
	if n < 7 {
		n = 7
	}
	if n > 65535 {
		n = 65535
	}
	return uint16(n)
}

// settingEngine builds the pion SettingEngine from the config: everything that decides which sockets ICE opens,
// which candidates it keeps and how long it punches. Factored out of NewSidecar so tests can run the real policy
// on a virtual network (punch_test.go).
func (sc *Sidecar) settingEngine(prime bool) (webrtc.SettingEngine, error) {
	cfg := sc.cfg
	se := webrtc.SettingEngine{}
	base := sc.netBase
	if prime {
		if base == nil {
			n, err := stdnet.NewNet()
			if err != nil {
				return se, err
			}
			base = n
		}
		low := sc.lowSend
		if low == nil {
			low = ttlLowSender(primeTTL(cfg))
		}
		base = &primingNet{Net: base, servers: iceServerIPs(append(append([]string{}, cfg.STUN...), cfg.TURN...),
			net.LookupIP), low: low}
	}
	if base != nil {
		se.SetNet(base)
	}
	lf := logging.NewDefaultLoggerFactory()
	lf.Writer = os.Stderr // stdout is the JSON protocol
	lf.DefaultLogLevel = logging.LogLevelWarn
	switch cfg.LogLevel {
	case "trace":
		lf.DefaultLogLevel = logging.LogLevelTrace
	case "debug":
		lf.DefaultLogLevel = logging.LogLevelDebug
	}
	se.LoggerFactory = lf
	se.DetachDataChannels()
	se.SetSCTPMaxReceiveBufferSize(cfg.RwndBytes)
	// Largest message we accept (advertised as a=max-message-size): one chunk + header, or a 60 KiB ctl message.
	// pion's default is ~1 GiB; a smaller cap bounds what even an authenticated peer can make us reassemble.
	se.SetSCTPMaxMessageSize(uint32(cfg.ChunkBytes + 64<<10))
	if cfg.MinCwnd > 0 {
		se.SetSCTPMinCwnd(cfg.MinCwnd)
	}
	if cfg.CwndCAStep > 0 {
		se.SetSCTPCwndCAStep(cfg.CwndCAStep)
	}
	se.SetICEMulticastDNSMode(ice.MulticastDNSModeDisabled)
	var nts []webrtc.NetworkType
	for _, n := range cfg.NetworkTypes {
		switch n {
		case "udp4":
			nts = append(nts, webrtc.NetworkTypeUDP4)
		case "udp6":
			nts = append(nts, webrtc.NetworkTypeUDP6)
		default:
			return se, fmt.Errorf("unsupported network type %q", n)
		}
	}
	// TCP/TLS TURN needs NO tcp4/tcp6 network type here. (Corrected 2026-10-08: an earlier comment claimed
	// pion/ice drops a `turn:...?transport=tcp` or `turns:` URL unless a TCP network type is configured.
	// That is FALSE on the pinned pion/ice v4.4.4: gather.go:1028 filters TURN URLs by
	// turnNetworkTypesForURL(url, a.turnTransportProtocols), and pion/webrtc v4.2.22 never sets
	// turnTransportProtocols, so the filter is empty = every type allowed. Measured: TCP and TLS relay
	// runs passed with network_types ["udp4"]. Adding tcp4/tcp6 would only enable ICE-TCP HOST candidates,
	// which nothing here wants, so the case was removed and "tcp4" is now an unsupported network type.)
	se.SetNetworkTypes(nts)
	if sc.cfg.TURNCAFile != "" {
		// An explicit CA for turns: servers (turntls.go). Without it pion verifies against the platform
		// store only, which on Windows ignores SSL_CERT_FILE; with it no GODEBUG or env var is involved.
		d, err := newTURNTLSDialer(sc.cfg.TURNCAFile, sc.cfg.TURN)
		if err != nil {
			return se, err
		}
		if cfg.pin != nil {
			d.local = cfg.pin.IP
		}
		se.SetICEProxyDialer(d)
	} else if cfg.pin != nil {
		// pion dials TCP/TLS TURN with a nil local address; a pinned sidecar dials them itself (pin.go).
		d, err := newTURNPinnedDialer(sc.cfg.TURN, cfg.pin.IP)
		if err != nil {
			return se, err
		}
		if d != nil {
			se.SetICEProxyDialer(d)
		}
	}
	se.SetICETimeouts(secs(cfg.ICEDisconnectedS), secs(cfg.ICEFailedS), secs(cfg.ICEKeepaliveS))
	// Check for the whole checking window, not pion's 7 requests (~1.4 s): see iceMaxBindingRequests.
	se.SetICEMaxBindingRequests(iceMaxBindingRequests(cfg))
	deny := cfg.IfaceDeny
	if cfg.Loopback {
		deny = nil // the IP filter below restricts the code test to 127.0.0.1 ("lo" must stay usable on Linux)
	}
	se.SetInterfaceFilter(func(name string) bool { return !ifaceDenied(name, deny) })
	var pinIPFilter func(net.IP) bool
	if cfg.pin != nil {
		// PINNED: gather on exactly one adapter and one address (validate already refused loopback + pin and a
		// pin the deny list excludes). srflx and UDP-TURN sockets then bind to this address too (pin.go).
		var ifFilter func(string) bool
		ifFilter, pinIPFilter = pinFilters(cfg.pin)
		se.SetInterfaceFilter(ifFilter)
	}
	if cfg.Loopback {
		// CODE TEST ONLY: two sidecars on one PC. Every pair this mode can form is a loopback host pair, which
		// the driver labels VOID (the positive control of the no-LAN rule).
		se.SetIncludeLoopbackCandidate(true)
		se.SetIPFilter(func(ip net.IP) bool { return ip.IsLoopback() && ip.To4() != nil })
		se.SetRemoteIPFilter(func(ip net.IP) bool { return ip.IsLoopback() })
	} else {
		// Local gathering: keep the real LAN base (srflx needs it), drop link-local, loopback and 100.64/10
		// (Tailscale). Remote candidates: public addresses only, so no LAN or VPN path can even be tried.
		se.SetIPFilter(localIPKeep(pinIPFilter))
		se.SetRemoteIPFilter(remoteIPFilter(sc.cfg.ICEPolicy, sc.cfg.AllowPrivateRelay))
	}
	return se, nil
}

// localIPKeep is the non-loopback local gathering filter: keep the real LAN base (srflx needs it), drop
// link-local, loopback and 100.64/10 (Tailscale) -- except that a PINNED sidecar keeps a CGNAT-range pin address
// (a phone hotspot may hand one out). With a pin, nothing but the pinned address survives.
func localIPKeep(pinIPFilter func(net.IP) bool) func(net.IP) bool {
	return func(ip net.IP) bool {
		c := classifyIP(ip.String())
		if pinIPFilter != nil && !pinIPFilter(ip) {
			return false
		}
		return c == "public" || c == "private" || (pinIPFilter != nil && c == "cgnat")
	}
}

// remoteIPKeep is the non-loopback remote candidate filter with the PRODUCTION setting (allow_private_relay
// false): public addresses only, so no LAN or VPN path can even be tried. It deliberately knows nothing about this
// node's own public IP or egress IP: two peers behind different NATs on ONE PC must accept each other's srflx
// (punch_test.go pins this with run2_0823's exact addresses).
func remoteIPKeep(icePolicy string) func(net.IP) bool { return remoteIPFilter(icePolicy, false) }

// remoteIPFilter: public remote candidates always; PRIVATE ones only when BOTH allow_private_relay is set and
// ice_policy is relay. DIAGNOSTIC (2026-10-05): a relay-only run may be pointed at a TURN server sitting on this
// LAN, whose relayed candidates are private addresses; without accepting them the relay path cannot be exercised
// here. Until 2026-10-10 ice_policy relay alone enabled that; production now needs it off (design 2026-10-09
// sec. 3.2), so it takes the explicit switch. Loopback, CGNAT, link-local etc. are never accepted.
func remoteIPFilter(icePolicy string, allowPrivateRelay bool) func(net.IP) bool {
	return func(ip net.IP) bool {
		c := classifyIP(ip.String())
		if c == "public" {
			return true
		}
		return allowPrivateRelay && icePolicy == "relay" && c == "private"
	}
}

func secs(s float64) time.Duration { return time.Duration(s * float64(time.Second)) }

func (sc *Sidecar) pcConfig() webrtc.Configuration {
	c := webrtc.Configuration{Certificates: []webrtc.Certificate{sc.cert}}
	// DIAGNOSTIC (2026-10-05): a credentialled TURN relay, so a failed hole punch degrades to a relay
	// instead of failing. This is the zero-configuration path: the TURN client only ever opens OUTBOUND
	// connections (RFC 8656 s3.1 allows UDP, TCP, TLS-over-TCP and DTLS client-to-server), so no peer
	// needs an inbound firewall rule or any OS setting.
	var ice []webrtc.ICEServer
	if !sc.cfg.Loopback {
		if len(sc.cfg.STUN) > 0 {
			ice = append(ice, webrtc.ICEServer{URLs: sc.cfg.STUN})
		}
		if len(sc.cfg.TURN) > 0 {
			ice = append(ice, webrtc.ICEServer{URLs: sc.cfg.TURN, Username: sc.cfg.TURNUser,
				Credential: sc.cfg.TURNPass, CredentialType: webrtc.ICECredentialTypePassword})
		}
	}
	c.ICEServers = ice
	switch sc.cfg.ICEPolicy {
	case "nohost":
		c.ICETransportPolicy = webrtc.ICETransportPolicyNoHost
	case "relay":
		// DIAGNOSTIC ONLY: forces every candidate to be a relay, which makes the fallback provable --
		// it removes the punch entirely, so a connection can only have come through the relay.
		c.ICETransportPolicy = webrtc.ICETransportPolicyRelay
	}
	return c
}

// ---------------------------------------------------------------------------------------- data socket

// acceptData: one authenticated client at a time. The first line must be "AUTH <local_token>".
func (sc *Sidecar) acceptData() {
	for {
		c, err := sc.dataLn.Accept()
		if err != nil {
			return
		}
		go func(c net.Conn) {
			_ = c.SetReadDeadline(time.Now().Add(5 * time.Second))
			buf := make([]byte, 0, 256)
			one := make([]byte, 1)
			for len(buf) < 200 {
				n, err := c.Read(one)
				if err != nil || n == 0 {
					c.Close()
					return
				}
				if one[0] == '\n' {
					break
				}
				buf = append(buf, one[0])
			}
			got := strings.TrimSpace(string(buf))
			want := "AUTH " + sc.cfg.LocalToken
			if subtle.ConstantTimeCompare([]byte(got), []byte(want)) != 1 {
				logf("data socket: rejected a connection with a bad token")
				c.Close()
				return
			}
			_ = c.SetReadDeadline(time.Time{})
			if _, err := c.Write([]byte("OK\n")); err != nil {
				c.Close()
				return
			}
			sc.dataMu.Lock()
			if sc.dataConn != nil {
				sc.dataConn.Close()
			}
			sc.dataConn = c
			sc.dataMu.Unlock()
			sc.em.Emit("data_connected", nil)
		}(c)
	}
}

// deliver writes one frame: a JSON header line, then exactly len(data) raw bytes.
func (sc *Sidecar) deliver(hdr map[string]any, data []byte, wait time.Duration) error {
	h, _ := json.Marshal(hdr)
	deadline := time.Now().Add(wait)
	var c net.Conn
	for {
		sc.dataMu.Lock()
		c = sc.dataConn
		sc.dataMu.Unlock()
		if c != nil {
			break
		}
		if time.Now().After(deadline) {
			return errors.New("no data socket client")
		}
		time.Sleep(100 * time.Millisecond)
	}
	sc.dataWMu.Lock()
	defer sc.dataWMu.Unlock()
	_ = c.SetWriteDeadline(time.Now().Add(120 * time.Second))
	if _, err := c.Write(append(h, byte(10))); err != nil {
		return err
	}
	if _, err := c.Write(data); err != nil {
		return err
	}
	return nil
}

// ---------------------------------------------------------------------------------------- dispatch

func (sc *Sidecar) reply(c Cmd, ev string, f map[string]any) {
	if f == nil {
		f = map[string]any{}
	}
	f["id"] = c.ID
	sc.em.Emit(ev, f)
}

func (sc *Sidecar) fail(c Cmd, err error) {
	sc.em.Emit("error", map[string]any{"id": c.ID, "cmd": c.Cmd, "error": err.Error()})
}

func (sc *Sidecar) dispatch(c Cmd) {
	defer func() {
		if r := recover(); r != nil {
			sc.em.Emit("error", map[string]any{"id": c.ID, "cmd": c.Cmd, "error": fmt.Sprintf("panic: %v", r)})
		}
	}()
	switch c.Cmd {
	case "nat_probe":
		t := c.TimeoutS
		if t <= 0 {
			t = 3
		}
		bip, err := sc.probeBindIP(c.BindIP)
		if err != nil {
			sc.reply(c, "nat_probe", map[string]any{"result": NATResult{Class: "FAILED", Note: "bind: " + err.Error()}})
			return
		}
		res := natProbe(c.Servers, secs(t), bip)
		if sc.cfg.ASNLookup {
			for _, ip := range res.PublicIPs {
				info := lookupASN(ip)
				sc.reply(c, "asn", map[string]any{"ip": ip, "asn": info.ASN, "prefix": info.Prefix, "source": info.Source, "error": info.Err})
			}
		}
		sc.reply(c, "nat_probe", map[string]any{"result": res})
	case "relay_put":
		sc.cmdRelayPut(c)
	case "relay_get":
		sc.cmdRelayGet(c)
	case "publish_identity":
		sc.cmdPublishIdentity(c)
	case "connect":
		sc.cmdConnect(c)
	case "gather_probe":
		sc.cmdGatherProbe(c)
	case "register":
		sc.cmdRegister(c)
	case "unregister":
		sc.mu.Lock()
		p := sc.parts[strings.ToLower(c.Sha)]
		delete(sc.parts, strings.ToLower(c.Sha))
		sc.mu.Unlock()
		var served int64
		if p != nil {
			served = p.served.Load()
			p.close()
		}
		sc.reply(c, "unregistered", map[string]any{"sha": c.Sha, "found": p != nil, "served_bytes": served})
	case "fetch":
		sc.cmdFetch(c)
	case "fetch_cancel":
		sc.cmdFetchCancel(c)
	case "send_app":
		p := sc.peer(c.Peer)
		if p == nil {
			sc.fail(c, fmt.Errorf("no connected peer %q", c.Peer))
			return
		}
		if len(c.Msg) > 48<<10 {
			sc.fail(c, errors.New("app message above 48 KiB"))
			return
		}
		err := p.sendCtl(map[string]any{"t": "app", "msg": c.Msg})
		sc.reply(c, "sent_app", map[string]any{"ok": err == nil, "error": errStr(err)})
	case "clock_sync":
		p := sc.peer(c.Peer)
		if p == nil {
			sc.fail(c, fmt.Errorf("no connected peer %q", c.Peer))
			return
		}
		n := c.N
		if n <= 0 {
			n = 20
		}
		out, err := p.clockSync(n)
		if err != nil {
			sc.fail(c, err)
			return
		}
		sc.reply(c, "clock_sync", out)
	case "stats":
		p := sc.peer(c.Peer)
		if p == nil {
			sc.fail(c, fmt.Errorf("no connected peer %q", c.Peer))
			return
		}
		sc.reply(c, "stats", map[string]any{"peer": c.Peer, "stripes": p.stripeInfo(true)})
	case "classify":
		var out []map[string]any
		for _, ip := range c.IPs {
			m := map[string]any{"ip": ip, "class": classifyIP(ip)}
			if sc.cfg.ASNLookup && classifyIP(ip) == "public" {
				info := lookupASN(ip)
				m["asn"], m["asn_error"], m["source"] = info.ASN, info.Err, info.Source
			}
			out = append(out, m)
		}
		sc.reply(c, "classify", map[string]any{"results": out})
	case "close":
		if p := sc.peer(c.Peer); p != nil {
			p.close("closed by driver")
		}
		sc.reply(c, "closed", map[string]any{"peer": c.Peer})
	default:
		sc.fail(c, fmt.Errorf("unknown cmd %q", c.Cmd))
	}
}

func errStr(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func (sc *Sidecar) peer(name string) *Peer {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	return sc.peers[name]
}

func (sc *Sidecar) Shutdown() {
	sc.cancel()
	sc.mu.Lock()
	var ps []*Peer
	for _, p := range sc.peers {
		ps = append(ps, p)
	}
	for _, p := range sc.parts {
		p.close()
	}
	sc.mu.Unlock()
	for _, p := range ps {
		p.close("shutdown")
	}
	sc.dataLn.Close()
	sc.dataMu.Lock()
	if sc.dataConn != nil {
		sc.dataConn.Close()
	}
	sc.dataMu.Unlock()
}

// ---------------------------------------------------------------------------------------- relay records

// putRec signs v for name (when per-miner keys are set; sign.go), seals it (when a signal key is set) and stores it.
func (sc *Sidecar) putRec(ctx context.Context, name string, v any) (string, int, error) {
	wv, err := sc.sign.wrap(name, v)
	if err != nil {
		return "", 0, err
	}
	sv, err := sc.seal.seal(name, wv)
	if err != nil {
		return "", 0, err
	}
	return sc.relay.PutJSON(ctx, name, sv)
}

// waitRec polls name until a record that opens (authenticates) appears, ctx ends, or wait elapses
// (wait <= 0: one look). Records that fail authentication are reported once and ignored: never parsed, never
// pinned. A forger can at worst hide the real record (denial of service), not replace it.
func (sc *Sidecar) waitRec(ctx context.Context, name string, v any, wait time.Duration) (bool, string, error) {
	return sc.waitRecFrom(ctx, name, v, wait, "")
}

// waitRecFrom is waitRec that, with per-miner keys configured, accepts only a record signed by node `from`
// ("" = any configured key). Unsigned, foreign and mis-signed records take the same path as a failed seal:
// reported once, ignored, never parsed.
func (sc *Sidecar) waitRecFrom(ctx context.Context, name string, v any, wait time.Duration, from string) (bool, string, error) {
	deadline := time.Now().Add(wait)
	poll := secs(sc.cfg.PollS)
	var lastErr error
	for {
		var raw json.RawMessage
		found, sha, err := sc.relay.GetJSON(ctx, name, &raw)
		if err != nil {
			// every unusable read is evidence (junk under the name, an oversize record, a relay error): log it and
			// report it once per distinct error, so a withheld or stalled record leaves a trace
			lastErr = err
			key := name + "|" + err.Error()
			sc.rejMu.Lock()
			first := !sc.rejected[key] && len(sc.rejected) < 1000 // bounded
			if first {
				sc.rejected[key] = true
			}
			sc.rejMu.Unlock()
			if first {
				logf("relay read %s: %v", name, err)
				sc.em.Emit("relay_read_error", map[string]any{"name": name, "error": err.Error()})
			}
		}
		if found {
			var inner json.RawMessage
			oerr := sc.seal.open(name, raw, &inner)
			if oerr == nil {
				_, oerr = sc.sign.unwrap(name, inner, from, v)
			}
			if oerr == nil {
				return true, sha, nil
			} else {
				lastErr = oerr
				sc.rejMu.Lock()
				first := !sc.rejected[sha] && len(sc.rejected) < 1000 // bounded
				if first {
					sc.rejected[sha] = true
				}
				sc.rejMu.Unlock()
				if first {
					sc.em.Emit("record_rejected", map[string]any{"name": name, "sha": sha, "why": oerr.Error()})
				}
			}
		}
		if wait <= 0 || time.Now().After(deadline) {
			return false, "", lastErr
		}
		select {
		case <-ctx.Done():
			return false, "", ctx.Err()
		case <-time.After(poll):
		}
	}
}

func (sc *Sidecar) cmdRelayPut(c Cmd) {
	if sc.relay == nil {
		sc.fail(c, errors.New("no relay configured"))
		return
	}
	var v any
	if err := json.Unmarshal(c.Obj, &v); err != nil {
		sc.fail(c, fmt.Errorf("obj: %v", err))
		return
	}
	ctx, cancel := context.WithTimeout(sc.ctx, 90*time.Second)
	defer cancel()
	sha, n, err := sc.putRec(ctx, c.Name, v)
	if err != nil {
		sc.reply(c, "relay_put", map[string]any{"ok": false, "name": c.Name, "error": err.Error()})
		return
	}
	sc.reply(c, "relay_put", map[string]any{"ok": true, "name": c.Name, "sha": sha, "bytes": n, "sealed": sc.seal != nil})
}

func (sc *Sidecar) cmdRelayGet(c Cmd) {
	if sc.relay == nil {
		sc.fail(c, errors.New("no relay configured"))
		return
	}
	var v json.RawMessage
	found, sha, err := sc.waitRec(sc.ctx, c.Name, &v, secs(c.WaitS))
	f := map[string]any{"name": c.Name, "found": found, "sha": sha, "error": errStr(err), "signed": sc.sign != nil}
	if found {
		f["obj"] = v
	}
	sc.reply(c, "relay_get", f)
}

// ---------------------------------------------------------------------------------------- identity + pinning

// IdentityRec is <ns>/rtc_id_<node>. It is NOT part of init_meta (plan 5.1): per-node fingerprints differ.
type IdentityRec struct {
	V    int     `json:"v"`
	Node string  `json:"node"`
	Alg  string  `json:"alg"`
	FP   string  `json:"fp"`
	T    float64 `json:"t"`
}

func (sc *Sidecar) nsName(ns, leaf string) (string, error) {
	if !strings.HasPrefix(ns, sc.cfg.NamePrefix) || !nameCharsOK(ns) {
		return "", fmt.Errorf("namespace %q is outside the allowed prefix %q", ns, sc.cfg.NamePrefix)
	}
	return strings.TrimRight(ns, "/") + "/" + leaf, nil
}

func (sc *Sidecar) cmdPublishIdentity(c Cmd) {
	if sc.relay == nil {
		sc.fail(c, errors.New("no relay configured"))
		return
	}
	name, err := sc.nsName(c.NS, "rtc_id_"+sc.cfg.Node)
	if err != nil {
		sc.fail(c, err)
		return
	}
	rec := IdentityRec{V: 1, Node: sc.cfg.Node, Alg: "sha-256", FP: sc.fp, T: float64(time.Now().UnixNano()) / 1e9}
	ctx, cancel := context.WithTimeout(sc.ctx, 90*time.Second)
	defer cancel()
	sha, n, err := sc.putRec(ctx, name, rec)
	if err != nil {
		sc.fail(c, err)
		return
	}
	sc.reply(c, "identity_published", map[string]any{"name": name, "sha": sha, "bytes": n, "fp": sc.fp, "sealed": sc.seal != nil})
}

func fpOK(fp string) bool {
	if len(fp) != 95 {
		return false
	}
	for i, r := range fp {
		if i%3 == 2 {
			if r != ':' {
				return false
			}
		} else if !(r >= '0' && r <= '9' || r >= 'A' && r <= 'F' || r >= 'a' && r <= 'f') {
			return false
		}
	}
	return true
}

// pin records the first fingerprint seen for a peer; any different later value is a violation (VOID).
func (sc *Sidecar) pin(peer, fp, source string) error {
	fp = strings.ToUpper(fp)
	sc.mu.Lock()
	cur, ok := sc.pins[peer]
	if !ok {
		sc.pins[peer] = fp
	}
	sc.mu.Unlock()
	if !ok {
		sc.em.Emit("pinned", map[string]any{"peer": peer, "fp": fp, "source": source})
		return nil
	}
	if cur != fp {
		sc.em.Emit("pin_violation", map[string]any{"peer": peer, "pinned": cur, "seen": fp, "where": source})
		return fmt.Errorf("pin violation for %s: pinned %s, %s shows %s", peer, cur, source, fp)
	}
	return nil
}

func (sc *Sidecar) pinned(peer string) string {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	return sc.pins[peer]
}

// sdpFingerprints returns every a=fingerprint value in an SDP as "alg VALUE".
func sdpFingerprints(sdp string) [][2]string {
	var out [][2]string
	for _, ln := range strings.Split(sdp, "\n") {
		ln = strings.TrimSpace(ln)
		if !strings.HasPrefix(ln, "a=fingerprint:") {
			continue
		}
		f := strings.Fields(strings.TrimPrefix(ln, "a=fingerprint:"))
		if len(f) == 2 {
			out = append(out, [2]string{strings.ToLower(f[0]), strings.ToUpper(f[1])})
		} else {
			out = append(out, [2]string{"?", ln})
		}
	}
	return out
}

// checkSDPPinned: at least one fingerprint, every fingerprint sha-256 and equal to the pin. pion accepts a
// DTLS certificate that matches ANY listed fingerprint, so an extra unpinned line must be refused here.
func (sc *Sidecar) checkSDPPinned(peer, sdp, where string) error {
	pin := sc.pinned(peer)
	if pin == "" {
		return fmt.Errorf("no pinned identity for %s", peer)
	}
	fps := sdpFingerprints(sdp)
	if len(fps) == 0 {
		return fmt.Errorf("%s: SDP has no a=fingerprint", where)
	}
	for _, f := range fps {
		if f[0] != "sha-256" || f[1] != pin {
			sc.em.Emit("pin_violation", map[string]any{"peer": peer, "pinned": pin, "seen": f[0] + " " + f[1], "where": where})
			return fmt.Errorf("%s: SDP fingerprint %s %s is not the pinned %s", where, f[0], f[1], pin)
		}
	}
	return nil
}

func certFingerprint(der []byte) string {
	sum := sha256.Sum256(der)
	h := strings.ToUpper(hex.EncodeToString(sum[:]))
	var b strings.Builder
	for i := 0; i < len(h); i += 2 {
		if i > 0 {
			b.WriteByte(':')
		}
		b.WriteString(h[i : i+2])
	}
	return b.String()
}
