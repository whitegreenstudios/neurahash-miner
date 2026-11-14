package main

// Signaling through the relay's name records (plan 5.1). Same API the GLM runner uses:
//   PUT /o/<sha256(body)>  + X-Name: <name>  + X-Auth: <token>   (the relay refuses a body whose sha256 != path)
//   GET /n/<name>  -> {"sha256": ..., "size": ..., "name": ...}  or 404 {"error": "no such name"}
//   GET /o/<sha>   -> the bytes
// Records are tiny JSON (identity ~150 B, offer/answer ~2-8 KB). This client refuses any record above
// MaxRecordBytes, so the relay can never carry bulk through this code. Every PUT name must start with the
// configured prefix, which must start with "p2ptest/". The token is sent only on PUT and never logged.

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Relay struct {
	base      string
	token     string
	prefix    string
	minerID   string
	maxRecord int
	client    *http.Client
	em        *Emitter
}

var errNoSuchName = errors.New("no such name")

func NewRelay(cfg *Config, em *Emitter) *Relay {
	return &Relay{
		base:      strings.TrimRight(cfg.RelayURL, "/"),
		token:     cfg.RelayToken,
		prefix:    cfg.NamePrefix,
		minerID:   cfg.MinerID,
		maxRecord: cfg.MaxRecordBytes,
		em:        em,
		client: &http.Client{
			Timeout: 30 * time.Second,
			// Proxy: nil. The default transport honours HTTP_PROXY, and a proxy would see the plain-HTTP X-Auth.
			Transport: &http.Transport{Proxy: nil, MaxIdleConns: 4, IdleConnTimeout: 30 * time.Second,
				TLSHandshakeTimeout: 10 * time.Second, ResponseHeaderTimeout: 30 * time.Second},
			// Never follow redirects: a redirect could carry request headers to another host, and our tiny
			// records are never moved to the relay's cold tier.
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		},
	}
}

func nameCharsOK(name string) bool {
	if len(name) == 0 || len(name) > 200 || strings.Contains(name, "..") || strings.Contains(name, "//") {
		return false
	}
	for _, r := range name {
		if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || strings.ContainsRune("_./:-", r)) {
			return false
		}
	}
	return true
}

func (r *Relay) checkName(name string) error {
	if r.base == "" {
		return errors.New("no relay configured")
	}
	if !nameCharsOK(name) {
		return fmt.Errorf("bad record name %q", name)
	}
	if !strings.HasPrefix(r.prefix, "p2ptest/") || !strings.HasPrefix(name, r.prefix) {
		return fmt.Errorf("record name %q is outside the allowed prefix %q", name, r.prefix)
	}
	return nil
}

func (r *Relay) do(ctx context.Context, method, path string, body []byte, hdr map[string]string) (int, []byte, error) {
	var rd io.Reader
	if body != nil {
		rd = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, r.base+path, rd)
	if err != nil {
		return 0, nil, err
	}
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	if r.minerID != "" {
		req.Header.Set("X-Miner-Id", r.minerID)
	}
	resp, err := r.client.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(io.LimitReader(resp.Body, int64(r.maxRecord)+4096))
	return resp.StatusCode, b, err
}

func short(b []byte) string {
	s := string(b)
	if len(s) > 200 {
		s = s[:200]
	}
	return strings.Map(func(r rune) rune {
		if r < 32 || r > 126 {
			return '?'
		}
		return r
	}, s)
}

// PutJSON stores v under name and checks that the name now resolves to it.
func (r *Relay) PutJSON(ctx context.Context, name string, v any) (string, int, error) {
	if err := r.checkName(name); err != nil {
		return "", 0, err
	}
	body, err := json.Marshal(v)
	if err != nil {
		return "", 0, err
	}
	if len(body) > r.maxRecord {
		return "", 0, fmt.Errorf("record %d B exceeds the %d B cap (the relay never carries bulk)", len(body), r.maxRecord)
	}
	sum := sha256.Sum256(body)
	sha := hex.EncodeToString(sum[:])
	hdr := map[string]string{"Content-Type": "application/octet-stream", "X-Name": name}
	if r.token != "" {
		hdr["X-Auth"] = r.token
	}
	var lastErr error
	for attempt := 0; attempt < 4; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return "", 0, ctx.Err()
			case <-time.After(time.Duration(attempt) * time.Second):
			}
		}
		st, rb, err := r.do(ctx, http.MethodPut, "/o/"+sha, body, hdr)
		if err != nil {
			lastErr = err
			continue
		}
		if st == 401 || st == 403 {
			return "", 0, fmt.Errorf("relay refused the PUT: HTTP %d %s", st, short(rb))
		}
		if st != 200 && st != 201 {
			lastErr = fmt.Errorf("relay PUT HTTP %d %s", st, short(rb))
			continue
		}
		got, _, gerr := r.resolve(ctx, name)
		if gerr != nil {
			lastErr = fmt.Errorf("resolve after PUT: %v", gerr)
			continue
		}
		if got != sha {
			lastErr = fmt.Errorf("name %s resolves to %s after PUT of %s", name, got, sha)
			continue
		}
		return sha, len(body), nil
	}
	return "", 0, lastErr
}

func (r *Relay) resolve(ctx context.Context, name string) (string, int64, error) {
	st, b, err := r.do(ctx, http.MethodGet, "/n/"+name, nil, nil)
	if err != nil {
		return "", 0, err
	}
	if st == 404 {
		if bytes.Contains(b, []byte("no such name")) {
			return "", 0, errNoSuchName
		}
		return "", 0, fmt.Errorf("unexpected 404 body %s", short(b))
	}
	if st != 200 {
		return "", 0, fmt.Errorf("GET /n HTTP %d %s", st, short(b))
	}
	var ent struct {
		Sha  string `json:"sha256"`
		Size int64  `json:"size"`
		Name string `json:"name"`
	}
	if err := json.Unmarshal(b, &ent); err != nil {
		return "", 0, fmt.Errorf("bad /n answer: %v", err)
	}
	if ent.Name != name {
		return "", 0, fmt.Errorf("relay echoed name %q for %q", ent.Name, name)
	}
	if len(ent.Sha) != 64 {
		return "", 0, errors.New("bad sha in /n answer")
	}
	return strings.ToLower(ent.Sha), ent.Size, nil
}

// GetJSON reads the record bound to name into v. found=false means the name does not exist (yet).
func (r *Relay) GetJSON(ctx context.Context, name string, v any) (bool, string, error) {
	if r.base == "" {
		return false, "", errors.New("no relay configured")
	}
	if !nameCharsOK(name) {
		return false, "", fmt.Errorf("bad record name %q", name)
	}
	sha, size, err := r.resolve(ctx, name)
	if errors.Is(err, errNoSuchName) {
		return false, "", nil
	}
	if err != nil {
		return false, "", err
	}
	if size > int64(r.maxRecord) {
		return false, "", fmt.Errorf("record %s is %d B, above the %d B cap", name, size, r.maxRecord)
	}
	st, b, err := r.do(ctx, http.MethodGet, "/o/"+sha, nil, nil)
	if err != nil {
		return false, "", err
	}
	if st != 200 {
		return false, "", fmt.Errorf("GET /o HTTP %d", st)
	}
	sum := sha256.Sum256(b)
	if hex.EncodeToString(sum[:]) != sha {
		return false, "", fmt.Errorf("record %s: body sha256 mismatch", name)
	}
	if err := json.Unmarshal(b, v); err != nil {
		return false, "", fmt.Errorf("record %s: bad json: %v", name, err)
	}
	return true, sha, nil
}

// WaitJSON polls until name exists, ctx ends, or wait elapses (wait <= 0: one try).
func (r *Relay) WaitJSON(ctx context.Context, name string, v any, wait, poll time.Duration) (bool, string, error) {
	deadline := time.Now().Add(wait)
	for {
		found, sha, err := r.GetJSON(ctx, name, v)
		if found {
			return true, sha, nil
		}
		if err != nil {
			logf("relay get %s: %v (will retry)", name, err)
		}
		if wait <= 0 || time.Now().After(deadline) {
			return false, "", err
		}
		select {
		case <-ctx.Done():
			return false, "", ctx.Err()
		case <-time.After(poll):
		}
	}
}
