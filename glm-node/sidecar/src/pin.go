package main

// Network-adapter pinning (2026-10-09): `bind_iface` / `bind_ip` make EVERY packet this sidecar sends for the
// measured path leave through ONE adapter, so two peers can run on one PC, each behind a different NAT (the
// "home" peer on Ethernet, the "mobile" peer on Wi-Fi joined to a phone hotspot).
//
// What the pin covers (all bound to the pinned adapter's IPv4 address):
//   - ICE host / srflx / UDP-TURN gathering: SetInterfaceFilter (name == pin) + SetIPFilter (ip == pin). With a
//     filter set, pion/ice v4.4.4 binds each srflx and UDP-TURN socket to the filtered local address
//     (gather.go:861-871, 1003-1064) instead of 0.0.0.0.
//   - TCP / TLS TURN: pion dials those with a nil local address (gather.go:1110, 1177), so a pinned sidecar always
//     installs its own proxy dialer whose net.Dialer.LocalAddr is the pinned IP (turntls.go).
//   - The NAT probe socket (netinfo.go natProbe).
// What it does NOT cover (stated, not hidden): the HTTP signaling to the relay and the ASN lookups (DNS, RIPEstat)
// use the default route. They are control plane, not the measured path.
// Refused at config time: a pin that names no existing, up adapter with an IPv4; a pin plus loopback mode; a pin
// that iface_deny would exclude; a pin plus `turns:...?transport=udp` (pion dials DTLS TURN unbound, gather.go:1135).
//
// IMPORTANT, measured 2026-10-09 07:48-08:10 on the 5090 PC: binding a socket to the Wi-Fi address is NECESSARY but
// NOT SUFFICIENT on a Windows host whose interfaces have Forwarding=Enabled. Route lookup still picks the
// lowest-metric default route (Ethernet), and the packet is FORWARDED out of Ethernet with the Wi-Fi source address
// (`ping -S <wifi-ip> -i 1 1.1.1.1` is answered "TTL expired" by the PC ITSELF), where the home router drops it.
// The pinned path only works for destinations that have a host route via the pinned adapter's gateway; see
// p2p/webrtc/README.md "Same-PC two-NAT test". The pin never adds routes or changes any OS setting itself.

import (
	"errors"
	"fmt"
	"net"
	"sort"
	"strings"
)

// ifaceInfo is the part of net.Interface the pin needs; a slice of these is injectable for tests.
type ifaceInfo struct {
	Name string
	Up   bool
	IPv4 []net.IP
}

// bindPin is a resolved pin: the adapter's name and the one IPv4 address every socket binds to.
type bindPin struct {
	Iface string
	IP    net.IP
}

// ifaceLister is a variable so tests can supply a fake adapter table.
var ifaceLister = listIfaces

func listIfaces() ([]ifaceInfo, error) {
	ifs, err := net.Interfaces()
	if err != nil {
		return nil, err
	}
	var out []ifaceInfo
	for _, it := range ifs {
		ii := ifaceInfo{Name: it.Name, Up: it.Flags&net.FlagUp != 0}
		addrs, err := it.Addrs()
		if err == nil {
			for _, a := range addrs {
				var ip net.IP
				switch v := a.(type) {
				case *net.IPNet:
					ip = v.IP
				case *net.IPAddr:
					ip = v.IP
				}
				if ip4 := ip.To4(); ip4 != nil {
					ii.IPv4 = append(ii.IPv4, ip4)
				}
			}
		}
		out = append(out, ii)
	}
	return out, nil
}

// pinnableIP: an address a measured-path socket may bind to (not loopback / link-local / unspecified / multicast).
func pinnableIP(ip net.IP) bool {
	c := classifyIP(ip.String())
	return ip.To4() != nil && (c == "private" || c == "public" || c == "cgnat")
}

// ifaceDenied is the iface_deny rule exactly as sidecar.go has always applied it ("=name" exact, else substring,
// both case-insensitive). Factored out so the pin check and the ICE filter can never disagree.
func ifaceDenied(name string, deny []string) bool {
	ln := strings.ToLower(name)
	for _, d := range deny {
		d = strings.ToLower(d)
		if strings.HasPrefix(d, "=") {
			if ln == d[1:] {
				return true
			}
		} else if d != "" && strings.Contains(ln, d) {
			return true
		}
	}
	return false
}

func ifaceNames(ifs []ifaceInfo) string {
	var n []string
	for _, it := range ifs {
		n = append(n, fmt.Sprintf("%q", it.Name))
	}
	sort.Strings(n)
	return strings.Join(n, ", ")
}

// resolvePin turns (bind_iface, bind_ip) into a bindPin. Both empty = no pin (nil, nil): today's behaviour.
func resolvePin(ifs []ifaceInfo, iface, ipStr string, deny []string) (*bindPin, error) {
	iface, ipStr = strings.TrimSpace(iface), strings.TrimSpace(ipStr)
	if iface == "" && ipStr == "" {
		return nil, nil
	}
	var want net.IP
	if ipStr != "" {
		want = net.ParseIP(ipStr)
		if want == nil || want.To4() == nil {
			return nil, fmt.Errorf("bind_ip %q is not an IPv4 address", ipStr)
		}
		want = want.To4()
		if !pinnableIP(want) {
			return nil, fmt.Errorf("bind_ip %s is %s: not an address a WAN socket can leave from", ipStr, classifyIP(ipStr))
		}
	}
	var hit *ifaceInfo
	if iface != "" {
		for i := range ifs {
			if strings.EqualFold(ifs[i].Name, iface) {
				hit = &ifs[i]
				break
			}
		}
		if hit == nil {
			return nil, fmt.Errorf("bind_iface %q: no such adapter (have %s)", iface, ifaceNames(ifs))
		}
	} else {
		for i := range ifs {
			for _, ip := range ifs[i].IPv4 {
				if ip.Equal(want) {
					hit = &ifs[i]
				}
			}
		}
		if hit == nil {
			return nil, fmt.Errorf("bind_ip %s is not assigned to any adapter on this machine", ipStr)
		}
	}
	if !hit.Up {
		return nil, fmt.Errorf("adapter %q is down", hit.Name)
	}
	var usable []net.IP
	for _, ip := range hit.IPv4 {
		if pinnableIP(ip) {
			usable = append(usable, ip.To4())
		}
	}
	if len(usable) == 0 {
		return nil, fmt.Errorf("adapter %q has no usable IPv4 address (connected? DHCP done?)", hit.Name)
	}
	ip := usable[0]
	if want != nil {
		ip = nil
		for _, u := range usable {
			if u.Equal(want) {
				ip = u
			}
		}
		if ip == nil {
			return nil, fmt.Errorf("bind_ip %s is not an address of adapter %q", ipStr, hit.Name)
		}
	} else if len(usable) > 1 {
		return nil, fmt.Errorf("adapter %q has %d IPv4 addresses: also set bind_ip to choose one", hit.Name, len(usable))
	}
	if ifaceDenied(hit.Name, deny) {
		return nil, fmt.Errorf("adapter %q is excluded by iface_deny: a pin cannot override the deny list", hit.Name)
	}
	return &bindPin{Iface: hit.Name, IP: ip}, nil
}

// pinFilters returns the ICE interface and IP filters for a pin.
func pinFilters(p *bindPin) (func(string) bool, func(net.IP) bool) {
	return func(name string) bool { return strings.EqualFold(name, p.Iface) },
		func(ip net.IP) bool { return ip.Equal(p.IP) }
}

// checkPinCompat refuses combinations the pin cannot honour.
func checkPinCompat(c *Config) error {
	if c.pin == nil {
		return nil
	}
	if c.Loopback {
		return errors.New("bind_iface/bind_ip and loopback mode are exclusive (loopback pins to 127.0.0.1 itself)")
	}
	_, _, dtls, err := turnAddrs(c.TURN)
	if err != nil {
		return err
	}
	if dtls {
		return errors.New("bind_iface/bind_ip cannot pin turns:...?transport=udp (pion dials DTLS TURN with no " +
			"local address); use turns: over TCP or turn: over UDP/TCP")
	}
	return nil
}

// probeBindIP resolves the per-call `bind_ip` of a nat_probe (preflight probes every adapter) or falls back to the
// config pin. Returns nil = 0.0.0.0 (today's behaviour).
func (sc *Sidecar) probeBindIP(callIP string) (net.IP, error) {
	if strings.TrimSpace(callIP) != "" {
		ifs, err := ifaceLister()
		if err != nil {
			return nil, err
		}
		p, err := resolvePin(ifs, "", callIP, nil)
		if err != nil {
			return nil, err
		}
		return p.IP, nil
	}
	if sc.cfg.pin != nil {
		return sc.cfg.pin.IP, nil
	}
	return nil, nil
}
