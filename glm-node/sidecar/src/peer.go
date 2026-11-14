package main

// One peer = k PeerConnections ("stripes"): k independent SCTP associations (k windows, k congestion
// windows) for long-RTT paths (plan 3.2, R1 Q6). Stripe 0 also carries the ordered "ctl" channel (HAVE, pings,
// driver messages). Every stripe carries one unordered reliable "bulk" channel for 256 KiB chunks.
// Channels are pre-negotiated (fixed ids), so neither side needs DCEP or OnDataChannel.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pion/datachannel"
	"github.com/pion/webrtc/v4"
)

const (
	ctlID  = uint16(0)
	bulkID = uint16(1)
)

type Stripe struct {
	idx      int
	pc       *webrtc.PeerConnection
	bulk     *webrtc.DataChannel
	bulkRW   datachannel.ReadWriteCloser
	ctl      *webrtc.DataChannel
	ctlRW    datachannel.ReadWriteCloser
	lowCh    chan struct{}
	sendMu   sync.Mutex
	ctlMu    sync.Mutex
	bulkOpen chan struct{}
	ctlOpen  chan struct{}
	failed   chan struct{}
	failOnce sync.Once
	state    atomic.Value // string
	serveQ   chan serveReq
	rxBytes  atomic.Int64
	txBytes  atomic.Int64
	// lastPair caches the most recent selected candidate pair ([2]AddrInfo). pion drops the pair when ICE fails
	// or the connection closes, so labelling from the live pair alone would lose a stripe's path evidence
	// (security review 2026-10-04, HIGH).
	lastPair atomic.Value
	pairMu   sync.Mutex // guards pairSeq + lastPair together: a slow lookup never overwrites a newer pair
	pairSeq  uint64
}

type Peer struct {
	sc       *Sidecar
	name     string
	epoch    int
	stripes  []*Stripe
	closed   chan struct{}
	once     sync.Once
	verified chan struct{} // closed once every stripe's DTLS certificate matched the pin; serving waits for it

	pingMu  sync.Mutex
	pingSeq uint64
	pings   map[uint64]chan pongMsg
}

type pongMsg struct {
	T0, T1 int64
	At     time.Time
}

// SignalRec is an offer or an answer: <ns>/rtc/<a>-<b>/o<epoch> or a<epoch> (plan 5.1). Vanilla ICE: each SDP
// already carries all gathered candidates.
type SignalRec struct {
	V     int      `json:"v"`
	Kind  string   `json:"kind"`
	From  string   `json:"from"`
	To    string   `json:"to"`
	Epoch int      `json:"epoch"`
	FP    string   `json:"fp"`
	SDP   []string `json:"sdp"`
	T     float64  `json:"t"`
}

func pairNames(a, b string) (lo, hi string) {
	if a < b {
		return a, b
	}
	return b, a
}

func (sc *Sidecar) newStripe(p *Peer, idx int, offerer bool) (*Stripe, error) {
	api := sc.apiAnswer // the answerer checks first, so its sockets are TTL-primed (prime.go)
	if offerer {
		api = sc.api
	}
	pc, err := api.NewPeerConnection(sc.pcConfig())
	if err != nil {
		return nil, err
	}
	st := &Stripe{idx: idx, pc: pc, lowCh: make(chan struct{}, 1), bulkOpen: make(chan struct{}),
		ctlOpen: make(chan struct{}), failed: make(chan struct{}), serveQ: make(chan serveReq, sc.cfg.MaxServeQueue)}
	st.state.Store("new")
	neg := true
	unordered := false
	bid := bulkID
	bulk, err := pc.CreateDataChannel("bulk", &webrtc.DataChannelInit{Negotiated: &neg, ID: &bid, Ordered: &unordered})
	if err != nil {
		pc.Close()
		return nil, err
	}
	st.bulk = bulk
	bulk.SetBufferedAmountLowThreshold(uint64(sc.cfg.HighWater / 2))
	bulk.OnBufferedAmountLow(func() {
		select {
		case st.lowCh <- struct{}{}:
		default:
		}
	})
	bulk.OnOpen(func() {
		rw, err := bulk.Detach()
		if err != nil {
			logf("stripe %d: bulk detach: %v", idx, err)
			st.fail()
			return
		}
		st.bulkRW = rw
		close(st.bulkOpen)
		go p.readBulk(st)
		go p.serveLoop(st)
	})
	if idx == 0 {
		ordered := true
		cid := ctlID
		ctl, err := pc.CreateDataChannel("ctl", &webrtc.DataChannelInit{Negotiated: &neg, ID: &cid, Ordered: &ordered})
		if err != nil {
			pc.Close()
			return nil, err
		}
		st.ctl = ctl
		ctl.OnOpen(func() {
			rw, err := ctl.Detach()
			if err != nil {
				logf("ctl detach: %v", err)
				st.fail()
				return
			}
			st.ctlRW = rw
			close(st.ctlOpen)
			go p.readCtl(st)
		})
	} else {
		close(st.ctlOpen)
	}
	pc.OnConnectionStateChange(func(s webrtc.PeerConnectionState) {
		st.state.Store(s.String())
		sc.em.Emit("peer_state", map[string]any{"peer": p.name, "epoch": p.epoch, "stripe": idx, "state": s.String()})
		if s == webrtc.PeerConnectionStateFailed || s == webrtc.PeerConnectionStateClosed {
			st.fail()
		}
	})
	pc.SCTP().Transport().ICETransport().OnSelectedCandidatePairChange(func(pair *webrtc.ICECandidatePair) {
		if pair == nil || pair.Local == nil || pair.Remote == nil {
			return
		}
		l, r := *pair.Local, *pair.Remote
		st.pairMu.Lock()
		st.pairSeq++
		seq := st.pairSeq
		st.pairMu.Unlock()
		go func() { // ASN lookups may take seconds: never block pion's callback goroutine
			ap := [2]AddrInfo{
				describeAddr(l.Typ.String(), l.Address, int(l.Port), l.Protocol.String(), l.RelatedAddress, sc.cfg.ASNLookup),
				describeAddr(r.Typ.String(), r.Address, int(r.Port), r.Protocol.String(), r.RelatedAddress, sc.cfg.ASNLookup),
			}
			st.pairMu.Lock()
			if st.pairSeq == seq {
				st.lastPair.Store(ap)
			}
			st.pairMu.Unlock()
			sc.em.Emit("pair_change", map[string]any{"peer": p.name, "epoch": p.epoch, "stripe": idx,
				"local": ap[0], "remote": ap[1]})
		}()
	})
	return st, nil
}

func (st *Stripe) fail() { st.failOnce.Do(func() { close(st.failed) }) }

// gather sets the local description and waits for ICE gathering to complete (vanilla ICE, no trickle).
func gather(pc *webrtc.PeerConnection, desc webrtc.SessionDescription, timeout time.Duration) (string, error) {
	done := webrtc.GatheringCompletePromise(pc)
	if err := pc.SetLocalDescription(desc); err != nil {
		return "", err
	}
	select {
	case <-done:
	case <-time.After(timeout):
		return "", errors.New("ICE gathering timed out")
	}
	ld := pc.LocalDescription()
	if ld == nil {
		return "", errors.New("no local description")
	}
	return ld.SDP, nil
}

func (sc *Sidecar) cmdConnect(c Cmd) {
	t0 := time.Now()
	peerName := c.Peer
	if !nodeNameOK(peerName) || peerName == sc.cfg.Node {
		sc.fail(c, fmt.Errorf("bad peer %q", peerName))
		return
	}
	if sc.relay == nil {
		sc.fail(c, errors.New("no relay configured"))
		return
	}
	wait := secs(c.WaitS)
	if wait <= 0 {
		wait = 10 * time.Minute
	}
	connectTimeout := secs(c.ConnectS)
	if connectTimeout <= 0 {
		connectTimeout = 90 * time.Second
	}
	ctx, cancel := context.WithTimeout(sc.ctx, wait+connectTimeout+time.Minute)
	defer cancel()
	// stage tells the driver WHERE a connect failed: only "ice" (signaling completed, the path did not come up)
	// can mean "no direct path" (BAD); identity / signaling / dtls_pin / setup failures VOID the run.
	// sig records WHEN each signaling record was published, seen and applied (wall clock, unix s), plus the remote
	// record's own creation time and candidate counts. The verdict uses it to refuse "no direct path" when the
	// signaling itself was late (a withheld or stalled record), which is not a path result (re-review MEDIUM).
	sig := map[string]any{"role": map[bool]string{true: "offerer", false: "answerer"}[sc.cfg.Node < peerName]}
	stamp := func(k string) { sig[k] = float64(time.Now().UnixNano()) / 1e9 }
	failf := func(stage, format string, a ...any) {
		sc.reply(c, "connect_failed", map[string]any{"peer": peerName, "epoch": c.Epoch, "stage": stage,
			"error": fmt.Sprintf(format, a...), "elapsed_s": time.Since(t0).Seconds(), "sig": sig})
	}

	// 1. identity: pin the peer's fingerprint on first sight (plan 5.2 item 1); sealed records only (seal.go)
	idName, err := sc.nsName(c.NS, "rtc_id_"+peerName)
	if err != nil {
		failf("setup", "%v", err)
		return
	}
	var id IdentityRec
	found, _, err := sc.waitRecFrom(ctx, idName, &id, wait, peerName)
	if !found {
		failf("identity", "no valid identity record %s within %.0f s (%v)", idName, wait.Seconds(), err)
		return
	}
	if id.V != 1 || id.Node != peerName || id.Alg != "sha-256" || !fpOK(id.FP) {
		failf("identity", "malformed identity record for %s", peerName)
		return
	}
	if err := sc.pin(peerName, id.FP, "identity:"+idName); err != nil {
		failf("identity", "%v", err)
		return
	}
	tID := time.Now()

	lo, hi := pairNames(sc.cfg.Node, peerName)
	offerer := sc.cfg.Node == lo
	offerName, err := sc.nsName(c.NS, fmt.Sprintf("rtc/%s-%s/o%d", lo, hi, c.Epoch))
	if err != nil {
		failf("setup", "%v", err)
		return
	}
	answerName, _ := sc.nsName(c.NS, fmt.Sprintf("rtc/%s-%s/a%d", lo, hi, c.Epoch))
	k := sc.cfg.Stripes
	p := &Peer{sc: sc, name: peerName, epoch: c.Epoch, closed: make(chan struct{}), verified: make(chan struct{}),
		pings: map[uint64]chan pongMsg{}}
	cleanup := func() {
		for _, st := range p.stripes {
			if st != nil {
				st.pc.Close()
			}
		}
		p.once.Do(func() { close(p.closed) }) // releases goroutines waiting on p.verified (never registered)
	}
	p.stripes = make([]*Stripe, k)
	for i := 0; i < k; i++ {
		st, err := sc.newStripe(p, i, offerer)
		if err != nil {
			cleanup()
			failf("setup", "new stripe %d: %v", i, err)
			return
		}
		p.stripes[i] = st
	}
	gatherT := 20 * time.Second
	sdps := make([]string, k)
	errs := make([]error, k)
	var wg sync.WaitGroup
	var remote SignalRec
	publish := func() []string { // what goes on the relay: raddr/rport zeroed (sanitizeSDP)
		out := make([]string, k)
		for i := range sdps {
			out[i] = sanitizeSDP(sdps[i])
		}
		return out
	}
	var published []string

	if offerer {
		for i := 0; i < k; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				off, err := p.stripes[i].pc.CreateOffer(nil)
				if err != nil {
					errs[i] = err
					return
				}
				sdps[i], errs[i] = gather(p.stripes[i].pc, off, gatherT)
			}(i)
		}
		wg.Wait()
		for i, e := range errs {
			if e != nil {
				cleanup()
				failf("signaling", "offer stripe %d: %v", i, e)
				return
			}
		}
		published = publish()
		rec := SignalRec{V: 1, Kind: "offer", From: sc.cfg.Node, To: peerName, Epoch: c.Epoch, FP: sc.fp, SDP: published,
			T: float64(time.Now().UnixNano()) / 1e9}
		if _, _, err := sc.putRec(ctx, offerName, rec); err != nil {
			cleanup()
			failf("signaling", "put offer: %v", err)
			return
		}
		stamp("t_offer_put")
		sig["offer_candidates"] = len(sdpCandidates(published))
		found, _, err := sc.waitRecFrom(ctx, answerName, &remote, wait, peerName)
		stamp("t_answer_seen")
		if !found {
			cleanup()
			failf("signaling", "no valid answer %s (%v)", answerName, err)
			return
		}
		if err := sc.checkSignal(&remote, "answer", peerName, c.Epoch, k); err != nil {
			cleanup()
			failf("signaling", "%v", err)
			return
		}
		sig["answer_rec_t"] = remote.T
		sig["answer_candidates"] = len(sdpCandidates(remote.SDP))
		for i := 0; i < k; i++ {
			if err := p.stripes[i].pc.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeAnswer, SDP: remote.SDP[i]}); err != nil {
				cleanup()
				failf("signaling", "set answer stripe %d: %v", i, err)
				return
			}
		}
		stamp("t_answer_applied")
	} else {
		found, _, err := sc.waitRecFrom(ctx, offerName, &remote, wait, peerName)
		stamp("t_offer_seen")
		if !found {
			cleanup()
			failf("signaling", "no valid offer %s (%v)", offerName, err)
			return
		}
		if err := sc.checkSignal(&remote, "offer", peerName, c.Epoch, k); err != nil {
			cleanup()
			failf("signaling", "%v", err)
			return
		}
		sig["offer_rec_t"] = remote.T
		sig["offer_candidates"] = len(sdpCandidates(remote.SDP))
		for i := 0; i < k; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				pc := p.stripes[i].pc
				if err := pc.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeOffer, SDP: remote.SDP[i]}); err != nil {
					errs[i] = err
					return
				}
				ans, err := pc.CreateAnswer(nil)
				if err != nil {
					errs[i] = err
					return
				}
				sdps[i], errs[i] = gather(pc, ans, gatherT)
			}(i)
		}
		wg.Wait()
		for i, e := range errs {
			if e != nil {
				cleanup()
				failf("signaling", "answer stripe %d: %v", i, e)
				return
			}
		}
		published = publish()
		rec := SignalRec{V: 1, Kind: "answer", From: sc.cfg.Node, To: peerName, Epoch: c.Epoch, FP: sc.fp, SDP: published,
			T: float64(time.Now().UnixNano()) / 1e9}
		if _, _, err := sc.putRec(ctx, answerName, rec); err != nil {
			cleanup()
			failf("signaling", "put answer: %v", err)
			return
		}
		stamp("t_answer_put")
		sig["answer_candidates"] = len(sdpCandidates(published))
	}
	tSig := time.Now()
	offeredLocal, offeredRemote := sdpCandidates(published), sdpCandidates(remote.SDP)
	sc.em.Emit("offered", map[string]any{"id": c.ID, "peer": peerName, "epoch": c.Epoch,
		"local": offeredLocal, "remote": offeredRemote})

	// 2. wait until every stripe's channels are open (ICE + DTLS + SCTP up). pion's DTLS handshake already refuses
	// a certificate that does not match the (pin-checked) remote SDP; step 3 re-checks the presented certificate.
	deadline := time.After(connectTimeout)
	for i, st := range p.stripes {
		for _, ch := range []chan struct{}{st.bulkOpen, st.ctlOpen} {
			select {
			case <-ch:
			case <-st.failed:
				cleanup()
				failf("ice", "stripe %d failed during connect (state %v)", i, st.state.Load())
				return
			case <-deadline:
				cleanup()
				failf("ice", "stripe %d not open after %.0f s (state %v)", i, connectTimeout.Seconds(), st.state.Load())
				return
			}
		}
	}
	tICE := time.Now()

	// 3. the DTLS certificate actually presented must hash to the pinned fingerprint, on every stripe.
	// Nothing is served and no ctl message is processed before this passes (p.verified).
	pin := sc.pinned(peerName)
	for i, st := range p.stripes {
		der := st.pc.SCTP().Transport().GetRemoteCertificate()
		got := certFingerprint(der)
		if len(der) == 0 || got != pin {
			sc.em.Emit("pin_violation", map[string]any{"peer": peerName, "pinned": pin, "seen": got, "where": fmt.Sprintf("dtls stripe %d", i)})
			cleanup()
			failf("dtls_pin", "stripe %d: DTLS certificate %s is not the pinned %s", i, got, pin)
			return
		}
	}
	close(p.verified)

	sc.mu.Lock()
	if old := sc.peers[peerName]; old != nil {
		go old.close("replaced by epoch " + fmt.Sprint(c.Epoch))
	}
	sc.peers[peerName] = p
	sc.mu.Unlock()
	go p.watch()
	// announce every part we already hold
	sc.mu.Lock()
	var mine []*Part
	for _, pt := range sc.parts {
		mine = append(mine, pt)
	}
	sc.mu.Unlock()
	for _, pt := range mine {
		_ = p.sendHave(pt)
	}
	sc.reply(c, "connected", map[string]any{
		"peer": peerName, "epoch": c.Epoch, "offerer": offerer, "stripes": p.stripeInfo(true),
		"t_identity_s": tID.Sub(t0).Seconds(), "t_signal_s": tSig.Sub(tID).Seconds(), "t_ice_s": tICE.Sub(tSig).Seconds(),
		"remote_cert_pinned": true, "pinned_fp": pin, "local_fp": sc.fp, "signal_sealed": sc.seal != nil,
		"sdp_bytes_local": totalLen(published), "sdp_bytes_remote": totalLen(remote.SDP),
		"offered_local": offeredLocal, "offered_remote": offeredRemote, "sig": sig,
	})
}

// sdpCandidates lists every a=candidate line of the SDPs with its address class (plan 6.3 pre-flight 3: no offered
// candidate may sit in 10/8, 100.64/10, 169.254/16, 172.16/12, 192.168/16 or fc00::/7 during this test).
func sdpCandidates(sdps []string) []map[string]any {
	var out []map[string]any
	for i, sdp := range sdps {
		for _, ln := range strings.Split(sdp, "\n") {
			ln = strings.TrimSpace(ln)
			if !strings.HasPrefix(ln, "a=candidate:") {
				continue
			}
			f := strings.Fields(strings.TrimPrefix(ln, "a=candidate:"))
			if len(f) < 8 || f[6] != "typ" {
				out = append(out, map[string]any{"stripe": i, "unparsed": true, "class": "invalid"})
				continue
			}
			m := map[string]any{"stripe": i, "component": f[1], "proto": strings.ToLower(f[2]), "addr": f[4], "port": f[5],
				"type": f[7], "class": classifyIP(f[4])}
			for j := 8; j+1 < len(f); j += 2 {
				if f[j] == "raddr" {
					m["raddr"] = f[j+1]
				}
			}
			out = append(out, m)
		}
	}
	return out
}

// sanitizeSDP zeroes the related address of every candidate line ("raddr <ip> rport <port>" becomes
// "raddr 0.0.0.0 rport 0", as browsers publish it). The raddr of a srflx candidate is this PC's private LAN base;
// the peer does not need it for ICE, and relay records are readable by anyone who knows the name.
func sanitizeSDP(sdp string) string {
	lines := strings.Split(sdp, "\n")
	for i, ln := range lines {
		body := strings.TrimSuffix(ln, "\r")
		if !strings.HasPrefix(body, "a=candidate:") {
			continue
		}
		f := strings.Split(body, " ")
		for j := 0; j+1 < len(f); j++ {
			switch f[j] {
			case "raddr":
				f[j+1] = "0.0.0.0"
			case "rport":
				f[j+1] = "0"
			}
		}
		out := strings.Join(f, " ")
		if strings.HasSuffix(ln, "\r") {
			out += "\r"
		}
		lines[i] = out
	}
	return strings.Join(lines, "\n")
}

func totalLen(ss []string) int {
	n := 0
	for _, s := range ss {
		n += len(s)
	}
	return n
}

// cmdGatherProbe: one PeerConnection with this run's ICE settings, gathered, then closed. No signaling and no peer:
// it shows which candidates this PC would offer (plan 6.3 pre-flight 3), using the configured STUN servers only.
func (sc *Sidecar) cmdGatherProbe(c Cmd) {
	t0 := time.Now()
	pc, err := sc.api.NewPeerConnection(sc.pcConfig())
	if err != nil {
		sc.fail(c, err)
		return
	}
	defer pc.Close()
	neg := true
	bid := bulkID
	if _, err := pc.CreateDataChannel("bulk", &webrtc.DataChannelInit{Negotiated: &neg, ID: &bid}); err != nil {
		sc.fail(c, err)
		return
	}
	off, err := pc.CreateOffer(nil)
	if err != nil {
		sc.fail(c, err)
		return
	}
	sdp, err := gather(pc, off, 20*time.Second)
	if err != nil {
		sc.fail(c, err)
		return
	}
	sc.reply(c, "gather_probe", map[string]any{"candidates": sdpCandidates([]string{sdp}), "gather_s": time.Since(t0).Seconds(),
		"sdp_bytes": len(sdp), "fingerprints": sdpFingerprints(sdp), "ice_policy": sc.cfg.ICEPolicy})
}

func (sc *Sidecar) checkSignal(r *SignalRec, kind, peer string, epoch, k int) error {
	if r.V != 1 || r.Kind != kind || r.From != peer || r.To != sc.cfg.Node || r.Epoch != epoch {
		return fmt.Errorf("%s record does not match (from %q to %q epoch %d kind %q)", kind, r.From, r.To, r.Epoch, r.Kind)
	}
	if len(r.SDP) != k {
		return fmt.Errorf("%s carries %d SDPs, want %d (stripes must match on both sides)", kind, len(r.SDP), k)
	}
	if strings.ToUpper(r.FP) != sc.pinned(peer) {
		sc.em.Emit("pin_violation", map[string]any{"peer": peer, "pinned": sc.pinned(peer), "seen": r.FP, "where": kind + " record"})
		return fmt.Errorf("%s record fingerprint is not the pinned one", kind)
	}
	for i, s := range r.SDP {
		if len(s) > 32<<10 {
			return fmt.Errorf("%s SDP %d too large", kind, i)
		}
		if err := sc.checkSDPPinned(peer, s, fmt.Sprintf("%s sdp %d", kind, i)); err != nil {
			return err
		}
	}
	return nil
}

func (p *Peer) watch() {
	for _, st := range p.stripes {
		go func(st *Stripe) {
			select {
			case <-st.failed:
				p.sc.em.Emit("stripe_failed", map[string]any{"peer": p.name, "epoch": p.epoch, "stripe": st.idx})
				if st.idx == 0 {
					p.close("stripe 0 (ctl) failed")
				}
			case <-p.closed:
			}
		}(st)
	}
}

func (p *Peer) close(reason string) {
	p.once.Do(func() {
		close(p.closed)
		for _, st := range p.stripes {
			st.pc.Close()
		}
		p.sc.mu.Lock()
		if p.sc.peers[p.name] == p {
			delete(p.sc.peers, p.name)
		}
		p.sc.mu.Unlock()
		p.sc.em.Emit("peer_closed", map[string]any{"peer": p.name, "epoch": p.epoch, "reason": reason})
	})
}

func (p *Peer) liveStripes() []*Stripe {
	var out []*Stripe
	for _, st := range p.stripes {
		select {
		case <-st.failed:
			continue
		default:
		}
		select {
		case <-st.bulkOpen:
			out = append(out, st)
		default:
		}
	}
	return out
}

// stripeInfo: the selected candidate pair of every stripe, with address classes and ASNs (plan 6.3 evidence).
func (p *Peer) stripeInfo(withASN bool) []map[string]any {
	var out []map[string]any
	for _, st := range p.stripes {
		m := map[string]any{"stripe": st.idx, "state": st.state.Load(), "rx_bytes": st.rxBytes.Load(), "tx_bytes": st.txBytes.Load()}
		it := st.pc.SCTP().Transport().ICETransport()
		st.pairMu.Lock()
		seq0 := st.pairSeq // a pair change during the lookups below wins over this (older) read
		st.pairMu.Unlock()
		if pair, err := it.GetSelectedCandidatePair(); err == nil && pair != nil && pair.Local != nil && pair.Remote != nil {
			ap := [2]AddrInfo{
				describeAddr(pair.Local.Typ.String(), pair.Local.Address, int(pair.Local.Port), pair.Local.Protocol.String(), pair.Local.RelatedAddress, withASN && p.sc.cfg.ASNLookup),
				describeAddr(pair.Remote.Typ.String(), pair.Remote.Address, int(pair.Remote.Port), pair.Remote.Protocol.String(), pair.Remote.RelatedAddress, withASN && p.sc.cfg.ASNLookup),
			}
			st.pairMu.Lock()
			if st.pairSeq == seq0 { // no pair change since the read: cache it, superseding lookups in flight
				st.pairSeq++
				st.lastPair.Store(ap)
			}
			st.pairMu.Unlock()
			m["local"], m["remote"] = ap[0], ap[1]
		} else if ap, ok := st.lastPair.Load().([2]AddrInfo); ok {
			// the stripe lost its pair (ICE failed or closed): report the last pair it actually used
			m["local"], m["remote"], m["pair_from_cache"] = ap[0], ap[1], true
		} else {
			m["selected_error"] = errStr(err)
		}
		if ps, ok := it.GetSelectedCandidatePairStats(); ok {
			m["rtt_s"] = ps.CurrentRoundTripTime
			m["pair_bytes_sent"] = ps.BytesSent
			m["pair_bytes_received"] = ps.BytesReceived
			m["nominated"] = ps.Nominated
		}
		out = append(out, m)
	}
	return out
}

// ---------------------------------------------------------------------------------------- ctl channel

func (p *Peer) ctlStripe() *Stripe {
	if len(p.stripes) == 0 {
		return nil
	}
	return p.stripes[0]
}

func (p *Peer) sendCtl(v any) error {
	st := p.ctlStripe()
	if st == nil || st.ctlRW == nil {
		return errors.New("ctl channel not open")
	}
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if len(b) > 60<<10 {
		return errors.New("ctl message above 60 KiB")
	}
	st.ctlMu.Lock()
	defer st.ctlMu.Unlock()
	_, err = st.ctlRW.WriteDataChannel(b, true)
	return err
}

type ctlMsg struct {
	T      string          `json:"t"`
	Sha    string          `json:"sha"`
	Size   int64           `json:"size"`
	Chunk  int             `json:"chunk"`
	Hashes []string        `json:"hashes"`
	Meta   json.RawMessage `json:"meta"`
	N      uint64          `json:"n"`
	T0     int64           `json:"t0"`
	T1     int64           `json:"t1"`
	Msg    json.RawMessage `json:"msg"`
}

func (p *Peer) readCtl(st *Stripe) {
	// process nothing from the peer before its DTLS certificate has been checked against the pin
	select {
	case <-p.verified:
	case <-p.closed:
		return
	}
	buf := make([]byte, 128<<10)
	for {
		n, isStr, err := st.ctlRW.ReadDataChannel(buf)
		if err != nil {
			p.close("ctl read: " + err.Error())
			return
		}
		if !isStr {
			continue
		}
		var m ctlMsg
		if err := json.Unmarshal(buf[:n], &m); err != nil {
			logf("ctl from %s: bad json", p.name)
			continue
		}
		switch m.T {
		case "ping":
			_ = p.sendCtl(map[string]any{"t": "pong", "n": m.N, "t0": m.T0, "t1": time.Now().UnixNano()})
		case "pong":
			p.pingMu.Lock()
			ch := p.pings[m.N]
			delete(p.pings, m.N)
			p.pingMu.Unlock()
			if ch != nil {
				ch <- pongMsg{T0: m.T0, T1: m.T1, At: time.Now()}
			}
		case "have":
			p.sc.onHave(p, &m)
		case "want":
			p.sc.onWant(p, &m)
		case "app":
			p.sc.em.Emit("peer_app", map[string]any{"peer": p.name, "msg": m.Msg})
		default:
			logf("ctl from %s: unknown type %q", p.name, m.T)
		}
	}
}

// clockSync: NTP-style pings over the ctl channel; the sample with the smallest RTT gives the offset
// (peer wall clock minus ours) with an error bound of rtt/2.
func (p *Peer) clockSync(n int) (map[string]any, error) {
	type sample struct {
		rtt, off int64
	}
	var ss []sample
	for i := 0; i < n; i++ {
		p.pingMu.Lock()
		p.pingSeq++
		seq := p.pingSeq
		ch := make(chan pongMsg, 1)
		p.pings[seq] = ch
		p.pingMu.Unlock()
		t0 := time.Now()
		if err := p.sendCtl(map[string]any{"t": "ping", "n": seq, "t0": t0.UnixNano()}); err != nil {
			return nil, err
		}
		select {
		case pm := <-ch:
			rtt := pm.At.Sub(t0).Nanoseconds()
			mid := t0.UnixNano() + rtt/2
			ss = append(ss, sample{rtt: rtt, off: pm.T1 - mid})
		case <-time.After(5 * time.Second):
			p.pingMu.Lock()
			delete(p.pings, seq)
			p.pingMu.Unlock()
		case <-p.closed:
			return nil, errors.New("peer closed")
		}
		time.Sleep(50 * time.Millisecond)
	}
	if len(ss) == 0 {
		return nil, errors.New("no pong received")
	}
	sort.Slice(ss, func(i, j int) bool { return ss[i].rtt < ss[j].rtt })
	var rtts []float64
	for _, s := range ss {
		rtts = append(rtts, float64(s.rtt)/1e6)
	}
	return map[string]any{"peer": p.name, "offset_ns": ss[0].off, "rtt_ns": ss[0].rtt, "samples": len(ss),
		"rtt_ms_sorted": rtts, "median_rtt_ms": rtts[len(rtts)/2]}, nil
}
