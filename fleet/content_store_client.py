"""Tiny client for the VPS content store (`content_store.py` on 47.84.93.96:8710) -- the genuinely-public,
already-running WAN relay this project already uses for corpus distribution. Reused here so ANY fleet node
(4060, Colab, eventually 3070) can push/pull its per-round delta over the REAL internet, not the LAN.

API (see the VPS's content_store.py): GET /health, GET /o/<sha256>, GET /manifest (all public/unauthenticated),
PUT /o/<sha256> (needs X-Auth: <token> + the body's sha256 to match the path; optional X-Name registers a
friendly name -> {sha256,size} in the public manifest so a fetcher doesn't need to already know the hash).

No dependencies beyond stdlib (urllib) so this drops into Colab / any miner env unchanged."""
import hashlib
import json
import urllib.request


def put(base_url, token, data: bytes, name=None, timeout=120):
    """Upload `data`, registering it under the friendly `name` if given. Returns the sha256 hex."""
    h = hashlib.sha256(data).hexdigest()
    req = urllib.request.Request(base_url.rstrip("/") + "/o/" + h, data=data, method="PUT")
    req.add_header("X-Auth", token)
    if name:
        req.add_header("X-Name", name)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    if not resp.get("ok"):
        raise RuntimeError("content_store PUT failed: %r" % resp)
    return resp["sha256"]


def get_manifest(base_url, timeout=20):
    with urllib.request.urlopen(base_url.rstrip("/") + "/manifest", timeout=timeout) as r:
        return json.loads(r.read())


def get_by_hash(base_url, sha256hex, timeout=120):
    with urllib.request.urlopen(base_url.rstrip("/") + "/o/" + sha256hex, timeout=timeout) as r:
        return r.read()


def get_by_name(base_url, name, timeout=120):
    """Look up `name` in the public manifest, fetch its bytes. Returns (sha256, bytes) or (None, None)."""
    man = get_manifest(base_url, timeout=timeout)
    entry = man.get(name)
    if not entry:
        return None, None
    return entry["sha256"], get_by_hash(base_url, entry["sha256"], timeout=timeout)
