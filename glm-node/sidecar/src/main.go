// p2psidecar: a pion/webrtc v4 data-channel sidecar driven from Python.
//
// Control: one JSON object per line on stdin (commands), one JSON object per line on stdout (events).
// Bulk bytes to Python: one 127.0.0.1 TCP socket, authenticated with a per-run token passed on stdin.
// Human logs go to stderr and never contain the relay token.
//
// Design: docs/research/P2P_WEBRTC_TRANSPORT_PLAN_2026-10-04.md sections 3.2, 4.1, 5 and 6.3.
// Scope of this build: the first 2-peer WAN test (random payloads only; no model, pool, mint or wallet code).
package main

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"
)

const sidecarVersion = "p2psidecar/0.1.0 (pion/webrtc v4.2.22)"

// Config is the first stdin line. RelayToken is a secret: it is never echoed, logged or written anywhere.
type Config struct {
	Cmd              string   `json:"cmd"`
	ID               string   `json:"id"`
	Node             string   `json:"node"`
	RelayURL         string   `json:"relay_url"`
	RelayToken       string   `json:"relay_token"`
	SignalKey        string   `json:"signal_key"` // secret: 64 hex chars; seals every signaling record (seal.go)
	NamePrefix       string   `json:"name_prefix"`
	MinerID          string   `json:"miner_id"`
	LocalToken       string   `json:"local_token"`
	STUN             []string `json:"stun"`
	TURN             []string `json:"turn"`
	TURNUser         string   `json:"turn_user"`
	TURNPass         string   `json:"turn_pass"`    // secret: never echoed, logged or written anywhere
	TURNCAFile       string   `json:"turn_ca_file"` // PEM CA bundle trusted for turns: (TLS) servers; see turntls.go
	ICEPolicy        string   `json:"ice_policy"`
	Loopback         bool     `json:"loopback"`
	NetworkTypes     []string `json:"network_types"`
	IfaceDeny        []string `json:"iface_deny"`
	BindIface        string   `json:"bind_iface"` // pin every measured-path socket to this adapter (pin.go)
	BindIP           string   `json:"bind_ip"`    // ... or to this local IPv4 (with bind_iface: which of its IPs)
	Stripes          int      `json:"stripes"`
	RwndBytes        uint32   `json:"sctp_rwnd_bytes"`
	MinCwnd          uint32   `json:"sctp_min_cwnd_bytes"`
	CwndCAStep       uint32   `json:"sctp_cwnd_ca_step_bytes"`
	ChunkBytes       int      `json:"chunk_bytes"`
	WindowBytes      int      `json:"window_bytes"`
	HighWater        int      `json:"send_high_water_bytes"`
	MaxPartBytes     int64    `json:"max_part_bytes"`
	NoProgressS      float64  `json:"no_progress_timeout_s"`
	ReqTimeoutS      float64  `json:"chunk_request_timeout_s"`
	ICEDisconnectedS float64  `json:"ice_disconnected_s"`
	ICEFailedS       float64  `json:"ice_failed_s"`
	ICEKeepaliveS    float64  `json:"ice_keepalive_s"`
	// binding requests per candidate pair; 0 = punch for the whole checking window (sidecar.go
	// iceMaxBindingRequests). 7 = pion's default, i.e. the pre-2026-10-09 behaviour.
	ICEMaxBindingRequests int `json:"ice_max_binding_requests"`
	// IP TTL of the answerer's checks until its socket hears from the peer (prime.go); 0 = default 6, -1 = off.
	ICEPrimeTTL    int     `json:"ice_prime_ttl"`
	ASNLookup      bool    `json:"asn_lookup"`
	MaxRecordBytes int     `json:"max_record_bytes"`
	PollS          float64 `json:"poll_s"`
	MaxServeQueue  int     `json:"max_serve_queue"`
	LogLevel       string  `json:"log_level"`
	// Production switch (P2P_IN_DILOCO_RUNNER_DESIGN_2026-10-09 sec. 3.2): accept PRIVATE remote relay candidates
	// (a TURN server on this LAN). Default false: a relay pair can then only be public, so no LAN path can carry a
	// training byte (owner WAN-only rule). Only meaningful with ice_policy relay; DIAGNOSTIC profiles set it.
	AllowPrivateRelay bool `json:"allow_private_relay"`
	// fetch_progress cadence while a fetch runs (xfer.go); 0 = default 1 s, < 0 = no progress events.
	ProgressS float64 `json:"progress_interval_s"`
	// Per-miner signed signalling (sign.go): this node's Ed25519 private seed (secret, 64 hex; never echoed) and
	// the peers' public keys (node -> 64 hex). Both or neither; with both, every signalling record is signed and
	// a record not signed by the expected peer's key is ignored.
	SignKey  string            `json:"sign_key"`
	PeerKeys map[string]string `json:"peer_pubkeys"`

	pin *bindPin // resolved bind_iface/bind_ip; nil = unpinned (validate fills it)
}

func (c *Config) applyDefaults() {
	if c.ICEPolicy == "" {
		c.ICEPolicy = "nohost"
	}
	if len(c.NetworkTypes) == 0 {
		c.NetworkTypes = []string{"udp4"}
	}
	if c.IfaceDeny == nil {
		c.IfaceDeny = []string{"tailscale", "wireguard", "zerotier", "vethernet", "vmware", "virtualbox",
			"docker", "bluetooth", "openvpn", "tap-windows", "wintun", "nordlynx", "mullvad", "warp", "proton",
			"=wg0", "=lo", "=tun0", "=tap0"}
	}
	if c.Stripes <= 0 {
		c.Stripes = 4
	}
	if c.RwndBytes == 0 {
		c.RwndBytes = 8 << 20
	}
	if c.ChunkBytes <= 0 {
		c.ChunkBytes = 256 << 10
	}
	if c.WindowBytes <= 0 {
		c.WindowBytes = 8 << 20
	}
	if c.HighWater <= 0 {
		c.HighWater = 8 << 20
	}
	if c.MaxPartBytes <= 0 {
		c.MaxPartBytes = 128 << 20
	}
	if c.NoProgressS <= 0 {
		c.NoProgressS = 120
	}
	if c.ReqTimeoutS <= 0 {
		c.ReqTimeoutS = 60
	}
	if c.ICEDisconnectedS <= 0 {
		c.ICEDisconnectedS = 10
	}
	if c.ICEFailedS <= 0 {
		c.ICEFailedS = 30
	}
	if c.ICEKeepaliveS <= 0 {
		c.ICEKeepaliveS = 2
	}
	if c.MaxRecordBytes <= 0 {
		c.MaxRecordBytes = 64 << 10
	}
	if c.PollS <= 0 {
		c.PollS = 1
	}
	if c.MaxServeQueue <= 0 {
		c.MaxServeQueue = 512
	}
	if c.ProgressS == 0 {
		c.ProgressS = 1
	}
}

func (c *Config) validate() error {
	if !nodeNameOK(c.Node) {
		return fmt.Errorf("bad node name %q (want [a-z0-9_-]{1,32})", c.Node)
	}
	if c.ChunkBytes < 16<<10 || c.ChunkBytes > 1<<20 {
		return fmt.Errorf("chunk_bytes %d outside [16 KiB, 1 MiB]", c.ChunkBytes)
	}
	if c.ICEMaxBindingRequests < 0 || c.ICEMaxBindingRequests > 65535 {
		return fmt.Errorf("ice_max_binding_requests %d outside [0, 65535]", c.ICEMaxBindingRequests)
	}
	if c.ICEPrimeTTL < -1 || c.ICEPrimeTTL > 64 {
		return fmt.Errorf("ice_prime_ttl %d outside [-1, 64] (-1 = off, 0 = default %d)", c.ICEPrimeTTL, defaultICEPrimeTTL)
	}
	if c.ICEPolicy != "nohost" && c.ICEPolicy != "all" && c.ICEPolicy != "relay" {
		return fmt.Errorf("ice_policy must be nohost, all or relay")
	}
	if c.ICEPolicy == "relay" && len(c.TURN) == 0 {
		return errors.New("ice_policy relay needs at least one TURN URL, or ICE gathers zero candidates")
	}
	if len(c.TURN) > 0 && c.TURNUser == "" {
		return errors.New("a TURN URL without username/credential is rejected by pion; set turn_user/turn_pass")
	}
	if c.TURNCAFile != "" {
		if _, err := newTURNTLSDialer(c.TURNCAFile, c.TURN); err != nil {
			return err
		}
	}
	if len(c.LocalToken) < 32 {
		return errors.New("local_token must be >= 32 chars")
	}
	if c.RelayURL != "" {
		if !strings.HasPrefix(c.NamePrefix, "p2ptest/") || strings.Contains(c.NamePrefix, "..") {
			return errors.New("name_prefix must start with p2ptest/ (this build writes only test records)")
		}
	}
	if c.AllowPrivateRelay && c.ICEPolicy != "relay" {
		return errors.New("allow_private_relay needs ice_policy relay (private relay candidates are a LAN-TURN diagnostic)")
	}
	if (c.SignKey == "") != (len(c.PeerKeys) == 0) {
		return errors.New("signed signalling needs both sign_key and peer_pubkeys (or neither)")
	}
	if c.Loopback && c.ICEPolicy != "all" {
		return errors.New("loopback mode needs ice_policy all (loopback pairs are host candidates)")
	}
	if c.BindIface != "" || c.BindIP != "" {
		ifs, err := ifaceLister()
		if err != nil {
			return fmt.Errorf("list adapters: %v", err)
		}
		p, err := resolvePin(ifs, c.BindIface, c.BindIP, c.IfaceDeny)
		if err != nil {
			return err
		}
		c.pin = p
	}
	return checkPinCompat(c)
}

func nodeNameOK(s string) bool {
	if len(s) < 1 || len(s) > 32 {
		return false
	}
	for _, r := range s {
		if !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '_' || r == '-') {
			return false
		}
	}
	return true
}

// Cmd is any stdin command after config. Unused fields stay zero.
type Cmd struct {
	Cmd        string          `json:"cmd"`
	ID         string          `json:"id"`
	Name       string          `json:"name"`
	NS         string          `json:"ns"`
	Obj        json.RawMessage `json:"obj"`
	WaitS      float64         `json:"wait_s"`
	Servers    []string        `json:"servers"`
	TimeoutS   float64         `json:"timeout_s"`
	Peer       string          `json:"peer"`
	Epoch      int             `json:"epoch"`
	ConnectS   float64         `json:"connect_timeout_s"`
	Sha        string          `json:"sha"`
	Size       int64           `json:"size"`
	Path       string          `json:"path"`
	Meta       json.RawMessage `json:"meta"`
	Deliver    string          `json:"deliver"`
	Msg        json.RawMessage `json:"msg"`
	N          int             `json:"n"`
	IPs        []string        `json:"ips"`
	ExpectSize int64           `json:"expect_size"`
	BindIP     string          `json:"bind_ip"` // nat_probe only: probe through this local IPv4 (preflight, pin.go)
}

// Emitter serializes stdout events; each event is one line.
type Emitter struct {
	mu    sync.Mutex
	w     *bufio.Writer
	start time.Time
}

func (e *Emitter) Emit(ev string, fields map[string]any) {
	if fields == nil {
		fields = map[string]any{}
	}
	now := time.Now()
	fields["ev"] = ev
	fields["ts"] = float64(now.UnixNano()) / 1e9
	fields["mono"] = now.Sub(e.start).Seconds()
	b, err := json.Marshal(fields)
	if err != nil {
		b = []byte(fmt.Sprintf(`{"ev":"error","error":%q}`, "marshal: "+err.Error()))
	}
	e.mu.Lock()
	e.w.Write(b)
	e.w.WriteByte('\n')
	e.w.Flush()
	e.mu.Unlock()
}

func logf(format string, a ...any) {
	fmt.Fprintf(os.Stderr, time.Now().Format("15:04:05.000")+" p2psidecar: "+format+"\n", a...)
}

func randHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	return hex.EncodeToString(b)
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "keygen" { // p2p2 2026-10-13: keygen.go
		os.Exit(keygenMain(os.Args[2:]))
	}
	em := &Emitter{w: bufio.NewWriterSize(os.Stdout, 1<<16), start: time.Now()}
	in := bufio.NewReaderSize(os.Stdin, 1<<20)

	line, err := readLine(in)
	if err != nil {
		em.Emit("error", map[string]any{"error": "no config line: " + err.Error()})
		os.Exit(2)
	}
	var cfg Config
	if err := json.Unmarshal(line, &cfg); err != nil || cfg.Cmd != "config" {
		em.Emit("error", map[string]any{"error": "first line must be {\"cmd\":\"config\",...}"})
		os.Exit(2)
	}
	cfg.applyDefaults()
	if err := cfg.validate(); err != nil {
		em.Emit("error", map[string]any{"id": cfg.ID, "error": "config: " + err.Error()})
		os.Exit(2)
	}
	sc, err := NewSidecar(&cfg, em)
	if err != nil {
		em.Emit("error", map[string]any{"id": cfg.ID, "error": "init: " + err.Error()})
		os.Exit(2)
	}
	ready := map[string]any{
		"id": cfg.ID, "version": sidecarVersion, "node": cfg.Node, "fp_alg": "sha-256", "fp": sc.fp,
		"data_addr": sc.dataAddr, "stripes": cfg.Stripes, "chunk_bytes": cfg.ChunkBytes,
		"window_bytes": cfg.WindowBytes, "sctp_rwnd_bytes": cfg.RwndBytes, "sctp_min_cwnd_bytes": cfg.MinCwnd,
		"sctp_cwnd_ca_step_bytes": cfg.CwndCAStep, "ice_policy": cfg.ICEPolicy, "loopback": cfg.Loopback,
		"network_types": cfg.NetworkTypes, "stun": cfg.STUN, "relay": cfg.RelayURL != "",
		"relay_token_set": cfg.RelayToken != "", "signal_sealed": sc.seal != nil, "max_part_bytes": cfg.MaxPartBytes,
		"iface_deny": cfg.IfaceDeny, "ice_max_binding_requests": iceMaxBindingRequests(&cfg),
		"ice_prime_ttl": primeTTL(&cfg),
	}
	if cfg.AllowPrivateRelay { // key only when set: a production ready event is unchanged
		ready["allow_private_relay"] = true
	}
	if sc.sign != nil {
		ready["signal_signed"], ready["sign_pubkey"] = true, sc.sign.pubHex()
	}
	if cfg.pin != nil { // keys only when pinned: an unpinned ready event is byte-identical to before
		ready["bind_iface"], ready["bind_ip"] = cfg.pin.Iface, cfg.pin.IP.String()
	}
	em.Emit("ready", ready)
	logf("ready node=%s fp=%s data=%s stripes=%d", cfg.Node, sc.fp, sc.dataAddr, cfg.Stripes)

	for {
		line, err := readLine(in)
		if err != nil {
			if !errors.Is(err, io.EOF) {
				logf("stdin: %v", err)
			}
			break
		}
		if len(strings.TrimSpace(string(line))) == 0 {
			continue
		}
		var c Cmd
		if err := json.Unmarshal(line, &c); err != nil {
			em.Emit("error", map[string]any{"error": "bad command json: " + err.Error()})
			continue
		}
		if c.Cmd == "shutdown" {
			break
		}
		go sc.dispatch(c)
	}
	sc.Shutdown()
	em.Emit("bye", nil)
}

func readLine(r *bufio.Reader) ([]byte, error) {
	var buf []byte
	for {
		part, isPrefix, err := r.ReadLine()
		if err != nil {
			return nil, err
		}
		buf = append(buf, part...)
		if len(buf) > 4<<20 {
			return nil, errors.New("command line longer than 4 MiB")
		}
		if !isPrefix {
			return buf, nil
		}
	}
}
