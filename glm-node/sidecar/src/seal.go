package main

// Sealed signaling records (security review 2026-10-04, MEDIUM "anyone can read and write the signaling").
// Relay names are world-readable (GET /n, GET /manifest) and the public-tier token can rebind any name outside
// pin/ and eph/, so a plaintext offer leaks the ICE credentials and a forged lobby/identity/offer/answer could be
// pinned first. With a shared test key, every record is AES-256-GCM sealed with the record NAME as associated data:
// a record that fails authentication is ignored (never parsed, never pinned), and a sealed record cannot be
// replayed under another name. Pin-on-first-sight (plan 5.2) still applies on top.

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
)

const sealAlg = "aes-256-gcm/p2ptest-signal-1"

type sealedRec struct {
	V     int    `json:"v"`
	Alg   string `json:"alg"`
	Nonce string `json:"nonce"`
	CT    string `json:"ct"`
}

type sealer struct {
	aead cipher.AEAD
}

func newSealer(keyHex string) (*sealer, error) {
	if keyHex == "" {
		return nil, nil
	}
	key, err := hex.DecodeString(keyHex)
	if err != nil || len(key) != 32 {
		return nil, errors.New("signal_key must be 64 hex chars (32 bytes)")
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	return &sealer{aead: aead}, nil
}

func aad(name string) []byte { return []byte(sealAlg + "|" + name) }

// seal returns the value to store under name: v itself without a key, else the sealed envelope.
func (s *sealer) seal(name string, v any) (any, error) {
	if s == nil {
		return v, nil
	}
	pt, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	nonce := make([]byte, s.aead.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return nil, err
	}
	ct := s.aead.Seal(nil, nonce, pt, aad(name))
	return sealedRec{V: 2, Alg: sealAlg, Nonce: base64.StdEncoding.EncodeToString(nonce),
		CT: base64.StdEncoding.EncodeToString(ct)}, nil
}

// open decodes the record stored under name into v. With a key, anything that is not a sealed record for THIS
// name is an error (the caller ignores it and keeps waiting).
func (s *sealer) open(name string, raw []byte, v any) error {
	if s == nil {
		return json.Unmarshal(raw, v)
	}
	var env sealedRec
	if err := json.Unmarshal(raw, &env); err != nil || env.V != 2 || env.Alg != sealAlg {
		return errors.New("record is not sealed with the run's signal key")
	}
	nonce, err1 := base64.StdEncoding.DecodeString(env.Nonce)
	ct, err2 := base64.StdEncoding.DecodeString(env.CT)
	if err1 != nil || err2 != nil || len(nonce) != s.aead.NonceSize() {
		return errors.New("malformed sealed record")
	}
	pt, err := s.aead.Open(nil, nonce, ct, aad(name))
	if err != nil {
		return errors.New("sealed record failed authentication")
	}
	if err := json.Unmarshal(pt, v); err != nil {
		return fmt.Errorf("sealed record: bad json: %v", err)
	}
	return nil
}
