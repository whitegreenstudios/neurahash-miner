package main

// TTL-limited NAT priming for the ANSWERER's ICE sockets (2026-10-09).
//
// ROOT CAUSE this fixes (measured on this PC's two real NATs, raw sockets, scratchpad order_punch.py / ttl_punch.py):
// when one side's packet reaches the far NAT BEFORE that NAT's own side has sent anything to it, the pair is dead in
// BOTH directions for as long as the early side keeps sending (mobile-first 0/0 twice, home-first 0/0, simultaneous
// 50/50 three times and 0/0 once; the STUN mappings themselves are unchanged afterwards and no ICMP comes back). This
// is the classic Linux-conntrack hole-punch clash: the unsolicited inbound creates a conntrack entry, the inside
// host's later outbound collides with it and is SNATed to a different port, and the continued unsolicited packets keep
// the bogus entry alive. ICE ordering makes it deterministic: the answerer starts checking the moment it applies the
// offer (pion/webrtc peerconnection.go:1450), seconds before the offerer has the answer, so the answerer's checks
// always reach the offerer's NAT first and poison it. (79e1986's longer check window made this worse, not better: it
// keeps the bogus entry refreshed.)
//
// FIX: until the answerer's socket has received ANY packet from a non-server address, every packet it sends to a
// non-server address leaves with a small IP TTL (cfg.ICEPrimeTTL, default 6): enough to cross its own NAT(s) -- which
// opens their state toward the peer -- and to expire before the peer's NAT. The offerer checks at full TTL; its first
// check therefore meets an already-open answerer NAT, gets through, and the answerer's socket flips to full TTL
// before pion answers it. STUN/TURN server traffic is always full TTL. Measured TTL windows on this PC:
// mobile-as-primer works for 2..10 (fails at 1 and >= 12), home-as-primer for 4..8+ (fails at 1..3); 6 fits both.

import (
	"net"
	"net/url"
	"strings"
	"sync"

	"github.com/pion/transport/v5"
	"golang.org/x/net/ipv4"
)

const defaultICEPrimeTTL = 6

// serverSet decides whether a destination is a STUN/TURN server (full TTL, never "opens" the socket).
type serverSet map[string]bool

func (s serverSet) has(a net.Addr) bool {
	ip := addrIP(a)
	return ip != nil && s[ip.String()]
}

func addrIP(a net.Addr) net.IP {
	switch v := a.(type) {
	case *net.UDPAddr:
		return v.IP
	case *net.TCPAddr:
		return v.IP
	}
	h, _, err := net.SplitHostPort(a.String())
	if err != nil {
		return nil
	}
	return net.ParseIP(h)
}

// iceServerIPs resolves the host of every stun:/turn:/turns: URL (an IP literal needs no lookup). lookup is
// injectable for tests; a host that does not resolve is skipped (pion will fail to use it too).
func iceServerIPs(urls []string, lookup func(string) ([]net.IP, error)) serverSet {
	s := serverSet{}
	for _, u := range urls {
		rest := u
		if i := strings.IndexByte(rest, ':'); i >= 0 {
			rest = rest[i+1:]
		}
		if i := strings.IndexByte(rest, '?'); i >= 0 {
			rest = rest[:i]
		}
		host := rest
		if h, _, err := net.SplitHostPort(rest); err == nil {
			host = h
		}
		host, _ = url.PathUnescape(host)
		if ip := net.ParseIP(host); ip != nil {
			s[ip.String()] = true
			continue
		}
		ips, err := lookup(host)
		if err != nil {
			continue
		}
		for _, ip := range ips {
			s[ip.String()] = true
		}
	}
	return s
}

// lowSender sends one packet that must die after crossing this side's NAT(s); called with c.mu held. Production:
// IP TTL on the socket. The vnet test supplies its own (vnet has no TTL).
type lowSender func(c *primedConn, p []byte, addr net.Addr) (int, error)

// primingNet wraps a transport.Net so every UDP socket ICE opens is a primedConn.
type primingNet struct {
	transport.Net
	servers serverSet
	low     lowSender
}

func (n *primingNet) wrap(c transport.UDPConn) transport.UDPConn {
	return &primedConn{UDPConn: c, servers: n.servers, low: n.low}
}

func (n *primingNet) ListenUDP(network string, laddr *net.UDPAddr) (transport.UDPConn, error) {
	c, err := n.Net.ListenUDP(network, laddr)
	if err != nil {
		return nil, err
	}
	return n.wrap(c), nil
}

func (n *primingNet) ListenPacket(network, address string) (net.PacketConn, error) {
	c, err := n.Net.ListenPacket(network, address)
	if err != nil {
		return nil, err
	}
	if u, ok := c.(transport.UDPConn); ok {
		return n.wrap(u), nil
	}
	return c, nil
}

// primedConn: low TTL toward non-servers until the first packet from a non-server arrives.
type primedConn struct {
	transport.UDPConn
	servers serverSet
	low     lowSender

	mu     sync.Mutex
	opened bool
	ttlLow bool // production lowSender state: is the socket's TTL currently the low one?
	full   int  // the socket's original TTL (0 = not read yet)
}

func (c *primedConn) noteRead(a net.Addr, err error) {
	if err != nil || a == nil || c.servers.has(a) {
		return
	}
	c.mu.Lock()
	c.opened = true
	c.mu.Unlock()
}

func (c *primedConn) ReadFrom(p []byte) (int, net.Addr, error) {
	n, a, err := c.UDPConn.ReadFrom(p)
	c.noteRead(a, err)
	return n, a, err
}

func (c *primedConn) ReadFromUDP(p []byte) (int, *net.UDPAddr, error) {
	n, a, err := c.UDPConn.ReadFromUDP(p)
	if a != nil {
		c.noteRead(a, err)
	}
	return n, a, err
}

// WriteTo holds c.mu across the TTL decision AND the write, so a concurrent full-TTL write can never restore the TTL
// between a low-TTL write's setsockopt and its send (pion writes from the agent loop and from gathering goroutines).
func (c *primedConn) WriteTo(p []byte, a net.Addr) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.servers.has(a) || c.opened {
		if c.ttlLow {
			if u, ok := c.UDPConn.(*net.UDPConn); ok {
				_ = ipv4.NewConn(u).SetTTL(c.full)
			}
			c.ttlLow = false
		}
		return c.UDPConn.WriteTo(p, a)
	}
	return c.low(c, p, a)
}

func (c *primedConn) WriteToUDP(p []byte, a *net.UDPAddr) (int, error) { return c.WriteTo(p, a) }

// ttlLowSender is the production lowSender (called with c.mu held): set IP_TTL=ttl on the socket (once, remembering
// the original) and write. Not a *net.UDPConn (cannot happen with stdnet): write at full TTL rather than not at all.
func ttlLowSender(ttl int) lowSender {
	return func(c *primedConn, p []byte, a net.Addr) (int, error) {
		u, ok := c.UDPConn.(*net.UDPConn)
		if !ok {
			return c.UDPConn.WriteTo(p, a)
		}
		if !c.ttlLow {
			pc := ipv4.NewConn(u)
			if c.full == 0 {
				if t, err := pc.TTL(); err == nil && t > 0 {
					c.full = t
				} else {
					c.full = 64
				}
			}
			if err := pc.SetTTL(ttl); err == nil {
				c.ttlLow = true
			}
		}
		return u.WriteTo(p, a)
	}
}
