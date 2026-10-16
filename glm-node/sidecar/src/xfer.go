package main

// Chunk protocol (plan 1.2, 4.1): a part is fetched as 256 KiB chunk requests over the bulk channels of all
// stripes. Each chunk is checked against the per-chunk sha256 list from the holder's HAVE, the whole part
// against its sha256, and the driver re-checks the whole part again after the localhost delivery.
//
// Wire (binary data-channel messages):
//   REQ  0x01 | sha[32] | idx u32 BE | req u64 BE                    (45 B)
//   DATA 0x02 | sha[32] | idx u32 BE | req u64 BE | payload           (45 B + chunk)
//   ERR  0x03 | sha[32] | idx u32 BE | req u64 BE | code u8           (46 B)
// A node serves ONLY parts its own driver registered (never arbitrary files).

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	msgReq  = 0x01
	msgData = 0x02
	msgErr  = 0x03
	hdrLen  = 45

	maxRemoteHave        = 256 // announcements kept (oldest evicted)
	maxConcurrentFetches = 6   // part buffers allocated at once
)

type Part struct {
	shaHex  string
	size    int64
	chunk   int
	n       int
	hashHex []string
	meta    json.RawMessage
	mu      sync.Mutex
	f       *os.File
	served  atomic.Int64
}

func (pt *Part) close() {
	pt.mu.Lock()
	if pt.f != nil {
		pt.f.Close()
		pt.f = nil
	}
	pt.mu.Unlock()
}

func (pt *Part) readChunk(idx int, dst []byte) error {
	pt.mu.Lock()
	f := pt.f
	pt.mu.Unlock()
	if f == nil {
		return errors.New("part closed")
	}
	_, err := f.ReadAt(dst, int64(idx)*int64(pt.chunk))
	if err == io.EOF {
		err = nil
	}
	return err
}

type Have struct {
	peer   string
	shaHex string
	size   int64
	chunk  int
	n      int
	hashes [][32]byte
	meta   json.RawMessage
	gotAt  time.Time
}

type serveReq struct {
	sha [32]byte
	idx uint32
	req uint64
}

type arrival struct {
	idx    int
	stripe int
	ok     bool
	reason string
}

type Fetch struct {
	id        string // the driver's fetch command id (carried on fetch_progress events)
	h         *Have
	peer      *Peer
	tStart    time.Time
	rxBytes   atomic.Int64 // accepted (chunk-verified) payload bytes so far
	cancelCh  chan struct{}
	cancelMu  sync.Once
	buf       []byte
	done      []atomic.Bool
	arrivals  chan arrival
	firstByte atomic.Int64 // unix ns of the first accepted chunk
	stripeRx  []atomic.Int64
	reqSeq    atomic.Uint64
}

func chunkLen(size int64, chunk, idx int) int {
	off := int64(idx) * int64(chunk)
	if rem := size - off; rem < int64(chunk) {
		return int(rem)
	}
	return chunk
}

func isHex64(s string) bool {
	if len(s) != 64 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

// ---------------------------------------------------------------------------------------- register (own parts)

func (sc *Sidecar) cmdRegister(c Cmd) {
	t0 := time.Now()
	sha := strings.ToLower(c.Sha)
	if !isHex64(sha) {
		sc.fail(c, errors.New("register: sha must be 64 hex"))
		return
	}
	f, err := os.Open(c.Path)
	if err != nil {
		sc.fail(c, fmt.Errorf("register: %v", err))
		return
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		sc.fail(c, fmt.Errorf("register: stat: %v", err))
		return
	}
	if st.Size() <= 0 || st.Size() > sc.cfg.MaxPartBytes || (c.Size > 0 && st.Size() != c.Size) {
		f.Close()
		sc.fail(c, fmt.Errorf("register: bad size %d (want %d, cap %d)", st.Size(), c.Size, sc.cfg.MaxPartBytes))
		return
	}
	size := st.Size()
	chunk := sc.cfg.ChunkBytes
	n := int((size + int64(chunk) - 1) / int64(chunk))
	whole := sha256.New()
	hashes := make([]string, n)
	buf := make([]byte, chunk)
	for i := 0; i < n; i++ {
		l := chunkLen(size, chunk, i)
		if _, err := io.ReadFull(f, buf[:l]); err != nil {
			f.Close()
			sc.fail(c, fmt.Errorf("register: read: %v", err))
			return
		}
		whole.Write(buf[:l])
		h := sha256.Sum256(buf[:l])
		hashes[i] = hex.EncodeToString(h[:])
	}
	if got := hex.EncodeToString(whole.Sum(nil)); got != sha {
		f.Close()
		sc.fail(c, fmt.Errorf("register: file sha256 %s != declared %s", got, sha))
		return
	}
	if len(c.Meta) > 2048 {
		f.Close()
		sc.fail(c, errors.New("register: meta above 2 KiB"))
		return
	}
	pt := &Part{shaHex: sha, size: size, chunk: chunk, n: n, hashHex: hashes, meta: c.Meta, f: f}
	sc.mu.Lock()
	if old := sc.parts[sha]; old != nil {
		old.close()
	}
	sc.parts[sha] = pt
	var ps []*Peer
	for _, p := range sc.peers {
		ps = append(ps, p)
	}
	sc.mu.Unlock()
	tHashed := time.Now()
	var sent []string
	for _, p := range ps {
		if err := p.sendHave(pt); err == nil {
			sent = append(sent, p.name)
		}
	}
	sc.reply(c, "registered", map[string]any{"sha": sha, "size": size, "nchunks": n, "hash_s": tHashed.Sub(t0).Seconds(),
		"have_sent_to": sent, "t_have_sent": float64(time.Now().UnixNano()) / 1e9})
}

func (p *Peer) sendHave(pt *Part) error {
	return p.sendCtl(map[string]any{"t": "have", "sha": pt.shaHex, "size": pt.size, "chunk": pt.chunk,
		"hashes": pt.hashHex, "meta": pt.meta})
}

// onHave validates a holder's announcement (it carries the per-chunk hash list, plan 4.1).
func (sc *Sidecar) onHave(p *Peer, m *ctlMsg) {
	sha := strings.ToLower(m.Sha)
	bad := func(why string) {
		sc.em.Emit("bad_have", map[string]any{"peer": p.name, "sha": m.Sha, "why": why})
	}
	if !isHex64(sha) {
		bad("sha")
		return
	}
	if m.Size <= 0 || m.Size > sc.cfg.MaxPartBytes {
		bad("size")
		return
	}
	if m.Chunk < 16<<10 || m.Chunk > 1<<20 {
		bad("chunk")
		return
	}
	n := int((m.Size + int64(m.Chunk) - 1) / int64(m.Chunk))
	if len(m.Hashes) != n {
		bad("hash count")
		return
	}
	hs := make([][32]byte, n)
	for i, s := range m.Hashes {
		b, err := hex.DecodeString(s)
		if err != nil || len(b) != 32 {
			bad("hash")
			return
		}
		copy(hs[i][:], b)
	}
	if len(m.Meta) > 2048 {
		bad("meta")
		return
	}
	h := &Have{peer: p.name, shaHex: sha, size: m.Size, chunk: m.Chunk, n: n, hashes: hs, meta: m.Meta, gotAt: time.Now()}
	sc.mu.Lock()
	if _, dup := sc.remoteHave[sha]; !dup && len(sc.remoteHave) >= maxRemoteHave {
		// bounded table: evict the oldest announcement (security review 2026-10-04, MEDIUM: HAVE flood)
		oldest, ot := "", time.Time{}
		for k, v := range sc.remoteHave {
			if oldest == "" || v.gotAt.Before(ot) {
				oldest, ot = k, v.gotAt
			}
		}
		delete(sc.remoteHave, oldest)
	}
	sc.remoteHave[sha] = h
	sc.mu.Unlock()
	sc.em.Emit("peer_have", map[string]any{"peer": p.name, "sha": sha, "size": m.Size, "chunk": m.Chunk, "nchunks": n, "meta": m.Meta})
}

// ---------------------------------------------------------------------------------------- serving

func (p *Peer) readBulk(st *Stripe) {
	buf := make([]byte, hdrLen+(1<<20)+64)
	for {
		n, isStr, err := st.bulkRW.ReadDataChannel(buf)
		if err != nil {
			if !errors.Is(err, io.EOF) {
				logf("stripe %d bulk read: %v", st.idx, err)
			}
			st.fail()
			return
		}
		st.rxBytes.Add(int64(n))
		if isStr || n < hdrLen {
			continue
		}
		switch buf[0] {
		case msgReq:
			if n != hdrLen {
				continue
			}
			var r serveReq
			copy(r.sha[:], buf[1:33])
			r.idx = binary.BigEndian.Uint32(buf[33:37])
			r.req = binary.BigEndian.Uint64(buf[37:45])
			select {
			case st.serveQ <- r:
			default:
				// queue full: drop; the fetcher re-requests after its timeout. Never block the reader.
			}
		case msgData:
			p.sc.onData(p, st, buf[:n])
		case msgErr:
			// informational: the fetcher's timeout re-requests
		}
	}
}

func (p *Peer) serveLoop(st *Stripe) {
	// serve nothing before the peer's DTLS certificate has been checked against the pin
	select {
	case <-p.verified:
	case <-p.closed:
		return
	}
	for {
		var r serveReq
		select {
		case r = <-st.serveQ:
		case <-st.failed:
			return
		case <-p.closed:
			return
		}
		shaHex := hex.EncodeToString(r.sha[:])
		p.sc.mu.Lock()
		pt := p.sc.parts[shaHex]
		p.sc.mu.Unlock()
		if pt == nil || int(r.idx) >= pt.n {
			msg := make([]byte, hdrLen+1)
			msg[0] = msgErr
			copy(msg[1:33], r.sha[:])
			binary.BigEndian.PutUint32(msg[33:37], r.idx)
			binary.BigEndian.PutUint64(msg[37:45], r.req)
			msg[45] = 1
			_ = p.sendBulk(st, msg)
			continue
		}
		l := chunkLen(pt.size, pt.chunk, int(r.idx))
		msg := make([]byte, hdrLen+l)
		msg[0] = msgData
		copy(msg[1:33], r.sha[:])
		binary.BigEndian.PutUint32(msg[33:37], r.idx)
		binary.BigEndian.PutUint64(msg[37:45], r.req)
		if err := pt.readChunk(int(r.idx), msg[hdrLen:]); err != nil {
			logf("serve %s[%d]: %v", shaHex[:12], r.idx, err)
			continue
		}
		if err := p.sendBulk(st, msg); err != nil {
			return
		}
		pt.served.Add(int64(l))
	}
}

// sendBulk applies backpressure: never more than HighWater bytes buffered in the SCTP send queue.
func (p *Peer) sendBulk(st *Stripe, msg []byte) error {
	hw := uint64(p.sc.cfg.HighWater)
	for st.bulk.BufferedAmount() > hw {
		select {
		case <-st.lowCh:
		case <-time.After(250 * time.Millisecond):
		case <-st.failed:
			return errors.New("stripe failed")
		case <-p.closed:
			return errors.New("peer closed")
		}
	}
	st.sendMu.Lock()
	_, err := st.bulkRW.WriteDataChannel(msg, false)
	st.sendMu.Unlock()
	if err == nil {
		st.txBytes.Add(int64(len(msg)))
	}
	return err
}

// ---------------------------------------------------------------------------------------- fetching

func (sc *Sidecar) fetchFor(shaHex, peer string) *Fetch {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	f := sc.fetches[shaHex]
	if f == nil || f.peer.name != peer {
		return nil
	}
	return f
}

func (sc *Sidecar) onData(p *Peer, st *Stripe, msg []byte) {
	shaHex := hex.EncodeToString(msg[1:33])
	f := sc.fetchFor(shaHex, p.name)
	if f == nil {
		return // stray (late duplicate after completion)
	}
	idx := int(binary.BigEndian.Uint32(msg[33:37]))
	if idx < 0 || idx >= f.h.n {
		sendArrival(f, arrival{idx: -1, stripe: st.idx, ok: false, reason: "index out of range"})
		return
	}
	payload := msg[hdrLen:]
	if len(payload) != chunkLen(f.h.size, f.h.chunk, idx) {
		sendArrival(f, arrival{idx: idx, stripe: st.idx, ok: false, reason: "wrong chunk length"})
		return
	}
	if sha256.Sum256(payload) != f.h.hashes[idx] {
		sendArrival(f, arrival{idx: idx, stripe: st.idx, ok: false, reason: "chunk sha256 mismatch"})
		return
	}
	if !f.done[idx].CompareAndSwap(false, true) {
		return // duplicate of an already accepted chunk
	}
	copy(f.buf[int64(idx)*int64(f.h.chunk):], payload)
	f.firstByte.CompareAndSwap(0, time.Now().UnixNano())
	f.rxBytes.Add(int64(len(payload)))
	if st.idx < len(f.stripeRx) {
		f.stripeRx[st.idx].Add(int64(len(payload)))
	}
	sendArrival(f, arrival{idx: idx, stripe: st.idx, ok: true})
}

func sendArrival(f *Fetch, a arrival) {
	select {
	case f.arrivals <- a:
	case <-time.After(5 * time.Second):
		logf("fetch %s: arrival queue stuck", f.h.shaHex[:12])
	}
}

// wantWait bounds the wait for a holder to answer a `want` (one ctl round trip over a live link).
const wantWait = 3 * time.Second

// onWant answers a peer that asks whether we hold a part. A successful fetch CONSUMES the announcement
// (cmdFetch deletes it below) and remoteHave keeps only ONE holder per sha, so without this a fetcher
// could not re-fetch an object it had already received, nor fetch from a holder other than the most
// recent announcer -- both of which a multi-source swarm depends on.
func (sc *Sidecar) onWant(p *Peer, m *ctlMsg) {
	sha := strings.ToLower(m.Sha)
	if !isHex64(sha) {
		return
	}
	sc.mu.Lock()
	pt := sc.parts[sha]
	sc.mu.Unlock()
	if pt == nil {
		return // we do not hold it: silence is the answer, and the fetcher falls over to the next holder
	}
	if err := p.sendHave(pt); err != nil {
		logf("ctl to %s: answer want: %v", p.name, err)
	}
}

// wanted blocks until the named peer's announcement for sha arrives, or the deadline passes.
func (sc *Sidecar) wanted(sha, peer string, d time.Duration) *Have {
	deadline := time.Now().Add(d)
	for {
		sc.mu.Lock()
		h := sc.remoteHave[sha]
		sc.mu.Unlock()
		if h != nil && (peer == "" || h.peer == peer) {
			return h
		}
		if time.Now().After(deadline) {
			return nil
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func (sc *Sidecar) cmdFetch(c Cmd) {
	sha := strings.ToLower(c.Sha)
	sc.mu.Lock()
	h := sc.remoteHave[sha]
	_, busy := sc.fetches[sha]
	sc.mu.Unlock()
	if h == nil || (c.Peer != "" && c.Peer != h.peer) {
		// Ask the peer we were told to use (or every connected peer when none was named). This is what
		// makes a SECOND fetch of the same object work after the first consumed the announcement, and
		// what lets a fetcher choose its holder instead of being pinned to the last announcer.
		var targets []*Peer
		if c.Peer != "" {
			if p := sc.peer(c.Peer); p != nil {
				targets = append(targets, p)
			}
		} else {
			sc.mu.Lock()
			for _, p := range sc.peers {
				targets = append(targets, p)
			}
			sc.mu.Unlock()
		}
		for _, p := range targets {
			_ = p.sendCtl(map[string]any{"t": "want", "sha": sha})
		}
		if len(targets) > 0 {
			h = sc.wanted(sha, c.Peer, wantWait)
		}
	}
	if h == nil {
		sc.fail(c, fmt.Errorf("fetch: no HAVE for %s", sha))
		return
	}
	if c.Peer != "" && c.Peer != h.peer {
		sc.fail(c, fmt.Errorf("fetch: %s was announced by %s, not %s", sha, h.peer, c.Peer))
		return
	}
	if c.ExpectSize > 0 && h.size != c.ExpectSize {
		sc.fail(c, fmt.Errorf("fetch: announced size %d != expected %d", h.size, c.ExpectSize))
		return
	}
	if busy {
		sc.fail(c, errors.New("fetch: already in progress"))
		return
	}
	p := sc.peer(h.peer)
	if p == nil {
		sc.fail(c, fmt.Errorf("fetch: peer %s not connected", h.peer))
		return
	}
	// bounded memory: at most maxConcurrentFetches part buffers at once (each <= MaxPartBytes)
	sc.mu.Lock()
	if _, busy2 := sc.fetches[sha]; busy2 || len(sc.fetches) >= maxConcurrentFetches {
		sc.mu.Unlock()
		sc.fail(c, fmt.Errorf("fetch: already in progress or %d fetches running", maxConcurrentFetches))
		return
	}
	tStart := time.Now()
	f := &Fetch{id: c.ID, h: h, peer: p, tStart: tStart, buf: make([]byte, h.size), done: make([]atomic.Bool, h.n),
		arrivals: make(chan arrival, 4096), stripeRx: make([]atomic.Int64, len(p.stripes)), cancelCh: make(chan struct{})}
	sc.fetches[sha] = f
	sc.mu.Unlock()
	defer func() {
		sc.mu.Lock()
		delete(sc.fetches, sha)
		sc.mu.Unlock()
	}()

	retries, misses, err := sc.runFetch(f)
	tDone := time.Now()
	base := map[string]any{"sha": sha, "peer": h.peer, "size": h.size, "nchunks": h.n, "meta": h.meta,
		"t_have": float64(h.gotAt.UnixNano()) / 1e9, "t_start": float64(tStart.UnixNano()) / 1e9,
		"t_done": float64(tDone.UnixNano()) / 1e9, "dur_s": tDone.Sub(tStart).Seconds(),
		"retries": retries, "chunk_misses": misses}
	if fb := f.firstByte.Load(); fb > 0 {
		base["t_first_byte"] = float64(fb) / 1e9
		base["ttfb_s"] = float64(fb-tStart.UnixNano()) / 1e9
	}
	var per []int64
	for i := range f.stripeRx {
		per = append(per, f.stripeRx[i].Load())
	}
	base["per_stripe_bytes"] = per
	base["stripes"] = p.stripeInfo(true)
	if err != nil {
		base["error"] = err.Error()
		if errors.Is(err, errFetchCancelled) {
			base["cancelled"] = true
		}
		sc.reply(c, "fetch_failed", base)
		return
	}
	if got := sha256.Sum256(f.buf); hex.EncodeToString(got[:]) != sha {
		base["error"] = "whole-part sha256 mismatch"
		sc.reply(c, "fetch_failed", base)
		return
	}
	base["verified"] = true
	sc.mu.Lock()
	delete(sc.remoteHave, sha) // fetched: the announcement is no longer needed
	sc.mu.Unlock()
	// p2p2 2026-10-13: a sub-tick fetch (a tiny part on one PC; Windows timer granularity) has dur 0, and +Inf made
	// json.Marshal fail, so the `fetched` reply was never emitted (the driver saw only "unsupported value: +Inf").
	if d := tDone.Sub(tStart).Seconds(); d > 0 {
		base["mbps"] = float64(h.size) * 8 / d / 1e6
	}
	switch c.Deliver {
	case "", "socket":
		derr := sc.deliver(map[string]any{"sha": sha, "size": h.size, "id": c.ID}, f.buf, 60*time.Second)
		base["delivered"] = derr == nil
		base["deliver_error"] = errStr(derr)
		base["t_delivered"] = float64(time.Now().UnixNano()) / 1e9
	case "none":
		base["delivered"] = false
	default:
		base["delivered"] = false
		base["deliver_error"] = "unknown deliver mode"
	}
	sc.reply(c, "fetched", base)
}

// runFetch: windowed chunk requests over all live stripes; re-request on timeout; fail on 3 integrity misses
// or on NoProgressS without an accepted chunk (the plan's "hang").
func (sc *Sidecar) runFetch(f *Fetch) (retries, misses int, err error) {
	h := f.h
	p := f.peer
	k := len(p.stripes)
	window := int64(sc.cfg.WindowBytes)
	reqTimeout := secs(sc.cfg.ReqTimeoutS)
	noProgress := secs(sc.cfg.NoProgressS)
	state := make([]uint8, h.n) // 0 pending, 1 requested, 2 done
	reqStripe := make([]int, h.n)
	reqAt := make([]time.Time, h.n)
	out := make([]int64, k)
	pending := make([]int, 0, h.n)
	for i := 0; i < h.n; i++ {
		pending = append(pending, i)
	}
	doneN := 0
	lastProgress := time.Now()
	progOn := sc.cfg.ProgressS > 0
	progEvery := secs(sc.cfg.ProgressS)
	lastProgEv := time.Now()
	firstSent := false
	rr := 0
	tick := time.NewTicker(100 * time.Millisecond)
	defer tick.Stop()
	shaRaw, _ := hex.DecodeString(h.shaHex)

	request := func(st *Stripe, idx int) error {
		msg := make([]byte, hdrLen)
		msg[0] = msgReq
		copy(msg[1:33], shaRaw)
		binary.BigEndian.PutUint32(msg[33:37], uint32(idx))
		binary.BigEndian.PutUint64(msg[37:45], f.reqSeq.Add(1))
		st.sendMu.Lock()
		_, err := st.bulkRW.WriteDataChannel(msg, false)
		st.sendMu.Unlock()
		return err
	}

	liveSet := make([]bool, k)
	for doneN < h.n {
		live := p.liveStripes()
		if len(live) == 0 {
			return retries, misses, errors.New("no live stripe")
		}
		for i := range liveSet {
			liveSet[i] = false
		}
		for _, st := range live {
			liveSet[st.idx] = true
		}
		// fill every live stripe's window, round robin
		for progressed := true; progressed && len(pending) > 0; {
			progressed = false
			for j := 0; j < len(live) && len(pending) > 0; j++ {
				st := live[(rr+j)%len(live)]
				idx := pending[0]
				l := int64(chunkLen(h.size, h.chunk, idx))
				if out[st.idx]+l > window {
					continue
				}
				if err := request(st, idx); err != nil {
					st.fail()
					continue
				}
				pending = pending[1:]
				state[idx], reqStripe[idx], reqAt[idx] = 1, st.idx, time.Now()
				out[st.idx] += l
				progressed = true
			}
			rr++
		}
		select {
		case a := <-f.arrivals:
			if !a.ok {
				misses++
				if misses >= 3 {
					return retries, misses, fmt.Errorf("integrity: %d chunk failures (last: %s)", misses, a.reason)
				}
				if a.idx >= 0 && state[a.idx] == 1 {
					out[reqStripe[a.idx]] -= int64(chunkLen(h.size, h.chunk, a.idx))
					state[a.idx] = 0
					pending = append(pending, a.idx)
				}
				continue
			}
			if state[a.idx] == 1 {
				out[reqStripe[a.idx]] -= int64(chunkLen(h.size, h.chunk, a.idx))
			} else if state[a.idx] == 0 {
				// arrived after a timeout re-queue: drop it from pending
				for i, v := range pending {
					if v == a.idx {
						pending = append(pending[:i], pending[i+1:]...)
						break
					}
				}
			}
			if state[a.idx] != 2 {
				state[a.idx] = 2
				doneN++
			}
			lastProgress = time.Now()
			if progOn && !firstSent {
				// the first verified chunk is the signal a hedging driver waits for: report it at once
				firstSent = true
				lastProgEv = lastProgress
				sc.emitFetchProgress(f, doneN, "first_byte")
			}
		case <-tick.C:
			now := time.Now()
			if progOn && now.Sub(lastProgEv) >= progEvery {
				lastProgEv = now
				sc.emitFetchProgress(f, doneN, "periodic")
			}
			for idx := 0; idx < h.n; idx++ {
				// re-request after the timeout, or at once if the stripe that carried the request has died
				if state[idx] == 1 && (now.Sub(reqAt[idx]) > reqTimeout || !liveSet[reqStripe[idx]]) {
					out[reqStripe[idx]] -= int64(chunkLen(h.size, h.chunk, idx))
					state[idx] = 0
					pending = append(pending, idx)
					retries++
				}
			}
			if now.Sub(lastProgress) > noProgress {
				return retries, misses, fmt.Errorf("hang: no chunk accepted for %.0f s (%d of %d done)", noProgress.Seconds(), doneN, h.n)
			}
		case <-p.closed:
			return retries, misses, errors.New("peer closed during fetch")
		case <-f.cancelCh:
			return retries, misses, errFetchCancelled
		}
	}
	return retries, misses, nil
}

// ---------------------------------------------------------------------------------------- progress + cancel
// (P2P_TRANSPORT_AUTOSWITCH_DESIGN_2026-10-10 sec. 2.4 / sec. 6 step 2: a driver hedging a P2P fetch against the
// relay must not wait for NoProgressS (120 s) to learn that nothing is arriving, and must be able to stop the
// losing fetch.)

var errFetchCancelled = errors.New("cancelled by driver (fetch_cancel)")

func (f *Fetch) cancel() { f.cancelMu.Do(func() { close(f.cancelCh) }) }

// emitFetchProgress: one fetch_progress event. bytes counts chunk-verified payload only; t_first_byte / ttfb_s are
// present once the first verified chunk has arrived (absent = nothing yet, the hedge signal).
func (sc *Sidecar) emitFetchProgress(f *Fetch, chunksDone int, trigger string) {
	now := time.Now()
	el := now.Sub(f.tStart).Seconds()
	b := f.rxBytes.Load()
	ev := map[string]any{"id": f.id, "sha": f.h.shaHex, "peer": f.h.peer, "size": f.h.size, "nchunks": f.h.n,
		"chunks_done": chunksDone, "bytes": b, "t_start": float64(f.tStart.UnixNano()) / 1e9, "elapsed_s": el,
		"trigger": trigger}
	if fb := f.firstByte.Load(); fb > 0 {
		ev["t_first_byte"] = float64(fb) / 1e9
		ev["ttfb_s"] = float64(fb-f.tStart.UnixNano()) / 1e9
	}
	if el > 0 {
		ev["mbps"] = float64(b) * 8 / el / 1e6
	}
	sc.em.Emit("fetch_progress", ev)
}

// cmdFetchCancel aborts the in-flight fetch of sha (optionally only if it is from peer). The cancelled fetch ends
// with its own fetch_failed reply carrying "cancelled": true; this op replies fetch_cancel at once. found=false
// means no fetch of that sha was in flight (not started yet, still asking holders, or already finished).
func (sc *Sidecar) cmdFetchCancel(c Cmd) {
	sha := strings.ToLower(c.Sha)
	if !isHex64(sha) {
		sc.fail(c, errors.New("fetch_cancel: sha must be 64 hex"))
		return
	}
	sc.mu.Lock()
	f := sc.fetches[sha]
	sc.mu.Unlock()
	if f != nil && c.Peer != "" && f.peer.name != c.Peer {
		f = nil
	}
	out := map[string]any{"sha": sha, "found": f != nil}
	if f != nil {
		out["peer"] = f.peer.name
		out["fetch_id"] = f.id
		out["bytes"] = f.rxBytes.Load()
		f.cancel()
	}
	sc.reply(c, "fetch_cancel", out)
}
