package main

// p2p2 (2026-10-13): `p2psidecar keygen <path>` -- the per-miner signing key of owner decision Q4, made ON the box that
// will use it: when <path> does not exist, a fresh Ed25519 seed (64 hex) is written there (exclusive create, 0600);
// either way the PUBLIC key is printed as "PUBKEY <64 hex>". The seed is never printed. Exit 0 on success, 2 on error.

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func keygenMain(args []string) int {
	if len(args) != 1 || args[0] == "" {
		fmt.Fprintln(os.Stderr, "usage: p2psidecar keygen <seed file path>")
		return 2
	}
	path := args[0]
	raw, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		seed := make([]byte, ed25519.SeedSize)
		if _, err := rand.Read(seed); err != nil {
			fmt.Fprintln(os.Stderr, "keygen: rand:", err)
			return 2
		}
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			fmt.Fprintln(os.Stderr, "keygen: mkdir:", err)
			return 2
		}
		f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if err != nil {
			fmt.Fprintln(os.Stderr, "keygen: create:", err)
			return 2
		}
		if _, err := f.WriteString(hex.EncodeToString(seed)); err != nil {
			f.Close()
			fmt.Fprintln(os.Stderr, "keygen: write:", err)
			return 2
		}
		if err := f.Close(); err != nil {
			fmt.Fprintln(os.Stderr, "keygen: close:", err)
			return 2
		}
		fmt.Println("CREATED", path)
		raw = []byte(hex.EncodeToString(seed))
	} else if err != nil {
		fmt.Fprintln(os.Stderr, "keygen: read:", err)
		return 2
	} else {
		fmt.Println("EXISTS", path)
	}
	seed, err := hex.DecodeString(strings.TrimSpace(string(raw)))
	if err != nil || len(seed) != ed25519.SeedSize {
		fmt.Fprintln(os.Stderr, "keygen: the file does not hold a 64-hex Ed25519 seed")
		return 2
	}
	fmt.Println("PUBKEY", hex.EncodeToString(ed25519.NewKeyFromSeed(seed).Public().(ed25519.PublicKey)))
	return 0
}
