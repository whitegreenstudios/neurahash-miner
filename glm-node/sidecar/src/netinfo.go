package main

// NAT mapping probe (plan 6.3 pre-flight 1) and address classification / ASN lookup (plan 6.3 evidence).

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/pion/stun/v4"
)

type StunResult struct {
	Server     string  `json:"server"`
	ServerIP   string  `json:"server_ip"`
	MappedIP   string  `json:"mapped_ip,omitempty"`
	MappedPort int     `json:"mapped_port,omitempty"`
	RTTms      float64 `json:"rtt_ms,omitempty"`
	Err        string  `json:"error,omitempty"`
}

type NATResult struct {
	LocalAddr     string       `json:"local_addr"`
	BindIP        string       `json:"bind_ip,omitempty"` // set only when the probe socket was pinned (pin.go)
	LocalPort     int          `json:"local_port"`
	Results       []StunResult `json:"results"`
	Answered      int          `json:"answered_distinct_server_ips"`
	Class         string       `json:"class"` // EIM | EDM | FAILED
	PortPreserved bool         `json:"port_preserved"`
	PublicIPs     []string     `json:"public_ips"`
	Note          string       `json:"note"`
}

// natProbe sends STUN Binding requests from ONE UDP socket to every server. The same mapped IP:port from
// servers on different IPs = endpoint-independent mapping (EIM); a different mapping = endpoint-dependent (EDM,
// "symmetric"). Fewer than two answering server IPs = FAILED (cannot classify).
//
// bindIP nil = 0.0.0.0 (Windows picks the source by the default route). Non-nil = the socket is bound to that
// adapter address (pin.go: necessary, and with a host route via that adapter's gateway, sufficient).
func natProbe(servers []string, timeout time.Duration, bindIP net.IP) NATResult {
	res := NATResult{Class: "FAILED"}
	lip := net.IPv4zero
	if bindIP != nil {
		lip = bindIP
		res.BindIP = bindIP.String()
	}
	conn, err := net.ListenUDP("udp4", &net.UDPAddr{IP: lip, Port: 0})
	if err != nil {
		res.Note = "listen: " + err.Error()
		return res
	}
	defer conn.Close()
	la := conn.LocalAddr().(*net.UDPAddr)
	res.LocalAddr = la.String()
	res.LocalPort = la.Port
	buf := make([]byte, 2048)
	for _, s := range servers {
		s = strings.TrimPrefix(strings.TrimPrefix(s, "stun:"), "stun://")
		r := StunResult{Server: s}
		raddr, err := net.ResolveUDPAddr("udp4", s)
		if err != nil {
			r.Err = "resolve: " + err.Error()
			res.Results = append(res.Results, r)
			continue
		}
		r.ServerIP = raddr.IP.String()
		for attempt := 0; attempt < 3 && r.MappedIP == ""; attempt++ {
			m, err := stun.Build(stun.TransactionID, stun.BindingRequest, stun.Fingerprint)
			if err != nil {
				r.Err = "build: " + err.Error()
				break
			}
			t0 := time.Now()
			if _, err := conn.WriteToUDP(m.Raw, raddr); err != nil {
				r.Err = "send: " + err.Error()
				continue
			}
			_ = conn.SetReadDeadline(t0.Add(timeout))
			for {
				n, from, err := conn.ReadFromUDP(buf)
				if err != nil {
					r.Err = "timeout/read: " + err.Error()
					break
				}
				if !from.IP.Equal(raddr.IP) {
					continue
				}
				msg := &stun.Message{Raw: append([]byte(nil), buf[:n]...)}
				if msg.Decode() != nil || msg.TransactionID != m.TransactionID {
					continue
				}
				var xa stun.XORMappedAddress
				if err := xa.GetFrom(msg); err == nil {
					r.MappedIP, r.MappedPort = xa.IP.String(), xa.Port
				} else {
					var ma stun.MappedAddress
					if err2 := ma.GetFrom(msg); err2 == nil {
						r.MappedIP, r.MappedPort = ma.IP.String(), ma.Port
					} else {
						r.Err = "no mapped address in response"
					}
				}
				r.RTTms = float64(time.Since(t0).Microseconds()) / 1000
				if r.MappedIP != "" {
					r.Err = ""
				}
				break
			}
		}
		res.Results = append(res.Results, r)
	}
	// classify over answers from DISTINCT server IPs
	seenServer := map[string]bool{}
	mappings := map[string]bool{}
	pub := map[string]bool{}
	preserved := true
	for _, r := range res.Results {
		if r.MappedIP == "" || seenServer[r.ServerIP] {
			continue
		}
		seenServer[r.ServerIP] = true
		mappings[fmt.Sprintf("%s:%d", r.MappedIP, r.MappedPort)] = true
		pub[r.MappedIP] = true
		if r.MappedPort != res.LocalPort {
			preserved = false
		}
	}
	res.Answered = len(seenServer)
	for ip := range pub {
		res.PublicIPs = append(res.PublicIPs, ip)
	}
	switch {
	case res.Answered < 2:
		res.Class = "FAILED"
		res.Note = "fewer than 2 STUN servers on distinct IPs answered: mapping behaviour unknown"
	case len(mappings) == 1:
		res.Class = "EIM"
		res.PortPreserved = preserved
		res.Note = "same mapped address from every server: endpoint-independent mapping"
	default:
		res.Class = "EDM"
		res.Note = "mapped address differs by destination: endpoint-dependent (symmetric) mapping"
	}
	return res
}

var cgnatNet = mustCIDR("100.64.0.0/10")
var reservedNets = []*net.IPNet{
	mustCIDR("0.0.0.0/8"), mustCIDR("192.0.0.0/24"), mustCIDR("192.0.2.0/24"), mustCIDR("198.18.0.0/15"),
	mustCIDR("198.51.100.0/24"), mustCIDR("203.0.113.0/24"), mustCIDR("240.0.0.0/4"), mustCIDR("2001:db8::/32"),
}

func mustCIDR(s string) *net.IPNet {
	_, n, err := net.ParseCIDR(s)
	if err != nil {
		panic(err)
	}
	return n
}

// classifyIP: public | private | cgnat | loopback | linklocal | reserved | multicast | unspecified | invalid
func classifyIP(s string) string {
	if i := strings.IndexByte(s, '%'); i >= 0 {
		s = s[:i]
	}
	ip := net.ParseIP(s)
	switch {
	case ip == nil:
		return "invalid"
	case ip.IsUnspecified():
		return "unspecified"
	case ip.IsLoopback():
		return "loopback"
	case ip.IsLinkLocalUnicast():
		return "linklocal"
	case ip.IsMulticast() || ip.IsLinkLocalMulticast() || ip.IsInterfaceLocalMulticast():
		return "multicast"
	case ip.IsPrivate():
		return "private" // 10/8, 172.16/12, 192.168/16, fc00::/7
	case cgnatNet.Contains(ip):
		return "cgnat"
	}
	for _, n := range reservedNets {
		if n.Contains(ip) {
			return "reserved"
		}
	}
	return "public"
}

type ASNInfo struct {
	ASN    string `json:"asn,omitempty"`
	Prefix string `json:"prefix,omitempty"`
	CC     string `json:"cc,omitempty"`
	Source string `json:"source,omitempty"`
	Err    string `json:"error,omitempty"`
}

type asnCache struct {
	mu sync.Mutex
	m  map[string]ASNInfo
}

var asnc = &asnCache{m: map[string]ASNInfo{}}

// lookupASN: Team Cymru's DNS (origin.asn.cymru.com TXT), then RIPEstat over HTTPS. Public IPv4 only.
func lookupASN(ip string) ASNInfo {
	if classifyIP(ip) != "public" {
		return ASNInfo{Err: "not a public address"}
	}
	asnc.mu.Lock()
	if v, ok := asnc.m[ip]; ok {
		asnc.mu.Unlock()
		return v
	}
	asnc.mu.Unlock()
	info := cymru(ip)
	if info.ASN == "" {
		r := ripestat(ip)
		if r.ASN != "" {
			info = r
		} else {
			info.Err = info.Err + "; " + r.Err
		}
	}
	asnc.mu.Lock()
	asnc.m[ip] = info
	asnc.mu.Unlock()
	return info
}

func cymru(ipStr string) ASNInfo {
	ip := net.ParseIP(ipStr).To4()
	if ip == nil {
		return ASNInfo{Err: "cymru: ipv4 only"}
	}
	q := fmt.Sprintf("%d.%d.%d.%d.origin.asn.cymru.com", ip[3], ip[2], ip[1], ip[0])
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	txts, err := net.DefaultResolver.LookupTXT(ctx, q)
	if err != nil || len(txts) == 0 {
		return ASNInfo{Err: fmt.Sprintf("cymru: %v", err)}
	}
	f := strings.Split(txts[0], "|")
	if len(f) < 3 {
		return ASNInfo{Err: "cymru: bad answer"}
	}
	asn := strings.Fields(strings.TrimSpace(f[0]))
	if len(asn) == 0 {
		return ASNInfo{Err: "cymru: no asn"}
	}
	return ASNInfo{ASN: asn[0], Prefix: strings.TrimSpace(f[1]), CC: strings.TrimSpace(f[2]), Source: "cymru-dns"}
}

func ripestat(ip string) ASNInfo {
	c := &http.Client{Timeout: 8 * time.Second}
	resp, err := c.Get("https://stat.ripe.net/data/network-info/data.json?resource=" + ip)
	if err != nil {
		return ASNInfo{Err: "ripestat: " + err.Error()}
	}
	defer resp.Body.Close()
	var v struct {
		Data struct {
			ASNs   []string `json:"asns"`
			Prefix string   `json:"prefix"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&v); err != nil || len(v.Data.ASNs) == 0 {
		return ASNInfo{Err: fmt.Sprintf("ripestat: no asn (%v)", err)}
	}
	return ASNInfo{ASN: v.Data.ASNs[0], Prefix: v.Data.Prefix, Source: "ripestat"}
}

// AddrInfo describes one side of a selected ICE candidate pair.
type AddrInfo struct {
	Type    string `json:"type"`
	Addr    string `json:"addr"`
	Port    int    `json:"port"`
	Proto   string `json:"proto"`
	Class   string `json:"class"`
	ASN     string `json:"asn,omitempty"`
	ASNErr  string `json:"asn_error,omitempty"`
	RelAddr string `json:"rel_addr,omitempty"`
}

func describeAddr(typ, addr string, port int, proto, rel string, withASN bool) AddrInfo {
	a := AddrInfo{Type: typ, Addr: addr, Port: port, Proto: proto, Class: classifyIP(addr), RelAddr: rel}
	if withASN && a.Class == "public" {
		info := lookupASN(addr)
		a.ASN, a.ASNErr = info.ASN, info.Err
	}
	return a
}
