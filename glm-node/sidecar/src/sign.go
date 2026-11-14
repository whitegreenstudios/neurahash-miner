package main

// Per-miner signed signalling (owner decision 2026-10-10, design P2P_TRANSPORT_AUTOSWITCH sec. 8 Q4).
// A shared signal key (seal.go) is not a secret at 1000 miners: anyone holding it can forge any node's identity,
// offer or answer. With per-miner keys, each node signs every signalling record it writes with its OWN Ed25519 key,
// and a reader accepts a record only if it is signed by the key configured for the node it expects to have written
// it. The signature covers the algorithm, the record NAME and the signer, so a record cannot be replayed under
// another name or re-attributed to another node. Signing sits INSIDE the seal: sign -> seal -> relay.
// Unsigned, foreign (signer without a configured key), wrong-signer and mis-signed records are errors; the caller
// (waitRecFrom) reports them once and ignores them, exactly like a record that fails the seal.
// Residual (documented, not solved here): a peer's OWN earlier record for the SAME name can be re-bound by anyone
// with relay write access; names carry the run namespace and epoch, and pin-on-first-sight still applies.

import (
	"crypto/ed25519"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
)

const signAlg = "ed25519/p2p-signal-1"

type signedRec struct {
	V    int             `json:"v"`
	Alg  string          `json:"alg"`
	Node string          `json:"node"`
	Body json.RawMessage `json:"body"`
	Sig  string          `json:"sig"`
}

type signer struct {
	node  string
	priv  ed25519.PrivateKey
	peers map[string]ed25519.PublicKey // includes this node's own key
}

// newSigner returns nil (unsigned records) when no key is configured. seedHex is this node's 32-byte Ed25519 seed.
func newSigner(node, seedHex string, peers map[string]string) (*signer, error) {
	if seedHex == "" && len(peers) == 0 {
		return nil, nil
	}
	if seedHex == "" || len(peers) == 0 {
		return nil, errors.New("signed signalling needs both sign_key and peer_pubkeys")
	}
	seed, err := hex.DecodeString(seedHex)
	if err != nil || len(seed) != ed25519.SeedSize {
		return nil, errors.New("sign_key must be 64 hex chars (an Ed25519 seed)")
	}
	s := &signer{node: node, priv: ed25519.NewKeyFromSeed(seed), peers: map[string]ed25519.PublicKey{}}
	own := s.priv.Public().(ed25519.PublicKey)
	for n, kh := range peers {
		if !nodeNameOK(n) {
			return nil, fmt.Errorf("peer_pubkeys: bad node name %q", n)
		}
		k, err := hex.DecodeString(kh)
		if err != nil || len(k) != ed25519.PublicKeySize {
			return nil, fmt.Errorf("peer_pubkeys[%s] must be 64 hex chars (an Ed25519 public key)", n)
		}
		if n == node && !own.Equal(ed25519.PublicKey(k)) {
			return nil, fmt.Errorf("peer_pubkeys[%s] (this node) does not match sign_key", n)
		}
		s.peers[n] = ed25519.PublicKey(k)
	}
	s.peers[node] = own
	return s, nil
}

func (s *signer) pubHex() string { return hex.EncodeToString(s.priv.Public().(ed25519.PublicKey)) }

// signedBytes is what the signature covers. name and node have restricted charsets (no '|'), body is last.
func signedBytes(name, node string, body []byte) []byte {
	b := []byte(signAlg + "|" + name + "|" + node + "|")
	return append(b, body...)
}

// wrap returns the value to store under name: v itself without keys, else the signed envelope.
func (s *signer) wrap(name string, v any) (any, error) {
	if s == nil {
		return v, nil
	}
	body, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	sig := ed25519.Sign(s.priv, signedBytes(name, s.node, body))
	return signedRec{V: 3, Alg: signAlg, Node: s.node, Body: body, Sig: base64.StdEncoding.EncodeToString(sig)}, nil
}

// unwrap decodes the record stored under name into v and returns its signer. from != "" requires that signer.
func (s *signer) unwrap(name string, raw []byte, from string, v any) (string, error) {
	if s == nil {
		return "", json.Unmarshal(raw, v)
	}
	var env signedRec
	if err := json.Unmarshal(raw, &env); err != nil || env.V != 3 || env.Alg != signAlg || len(env.Body) == 0 {
		return "", errors.New("record is not signed (per-miner signed signalling is on)")
	}
	pub, ok := s.peers[env.Node]
	if !ok {
		return "", fmt.Errorf("record signed by %q, which has no configured public key", env.Node)
	}
	if from != "" && env.Node != from {
		return "", fmt.Errorf("record signed by %q, expected %q", env.Node, from)
	}
	sig, err := base64.StdEncoding.DecodeString(env.Sig)
	if err != nil || len(sig) != ed25519.SignatureSize {
		return "", errors.New("malformed signature")
	}
	if !ed25519.Verify(pub, signedBytes(name, env.Node, env.Body), sig) {
		return "", fmt.Errorf("signature by %q does not verify for this record name", env.Node)
	}
	if err := json.Unmarshal(env.Body, v); err != nil {
		return "", fmt.Errorf("signed record: bad json: %v", err)
	}
	return env.Node, nil
}
