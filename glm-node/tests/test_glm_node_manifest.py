"""glm-node: the shipped bytes are the fleet's bytes, nothing is missing or extra, and nothing secret ships.

1. MANIFEST.sha256 names EVERY file of glm-node/ (except itself) and every sha256 matches -- so a checkout that
   Git rewrote (line endings!) or a stray local file is caught before a node starts.
2. The files the fleet compares by sha256 carry the fleet's sha256 (manifests/fleet_pins.json: runner, P2P
   transport, wire codec, golden check, libraries, sidecar).
3. glm-node/.gitattributes keeps every file byte-exact on every OS (`* -text`), and Git agrees.
4. A secrets / privacy scan: key blocks, API tokens, token-shaped hex, password and secret-named literals, e-mail
   addresses, private (RFC 1918) and carrier-grade-NAT IPs, Windows user-profile paths and account path segments,
   user@host logins, ssh key files, remote home dirs, X-Auth literals, paths into the private repo's directory. The
   pinned fleet files cannot be edited without changing their sha256, so their known hits are allowlisted by (file,
   kind, sha256 of the matched text) with an exact count and the reason -- a new hit anywhere, or one more in those
   files, fails. (The operator's own account / PC names are checked by the operator's private scan before
   publishing, so this public file does not have to carry them.)
5. The same scan over values that never appear whole in the source: the JOIN check folds every split literal
   ("ab" + "cd", adjacent literals, "".join([...]), constant f-strings; no allowlist), and the GLUED-CALL check glues
   every call of constant pieces (a helper, os.path.join, a constant format; hash allowlist with exact counts).

Run: python -m pytest glm-node/tests/test_glm_node_manifest.py -q
"""
import ast
import hashlib
import io
import json
import os
import re
import subprocess
import tokenize

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "MANIFEST.sha256")
PINS = os.path.join(ROOT, "manifests", "fleet_pins.json")

# (file, kind, first 16 hex of sha256(matched text)) -> (count, why). Hashes, not the strings: this file must not
# itself carry what it guards against.
_FIXTURE = ("the Stage-0 runner's (s0-redact) selftest: an INVENTED path / login / key name that its path stripper "
            "and argv redactor must remove (the selftest asserts they are gone from the public records)")
_REPO_DIR = ("the private repo's DIRECTORY name in a sha-pinned module's docstring / comment / default (no account, "
             "key, address or secret; a node passes every path it uses explicitly)")
ALLOWED = {
    # the old public demo token (owner decision 8: blank it only together with the token rotation)
    ("tools/glm_arc_eval.py", "hex32", "4c2c9eaabf4761b7"): (1,
        "LANE_TOKEN (tools/glm_arc_eval.py:99): the old demo lane's token. The SAME value (same sha256) is already "
        "public in this repository's main branch as PUBLIC_LANE_TOKEN in tools/sharddiloco_glm_contributor.py "
        "(line 2049 at 8a3a96a); the Stage-0 runner refuses to start without the node's own NEURAHASH_CONTENT_TOKEN "
        "(--allow-demo-token only on purpose)"),
    ("tools/glm_arc_eval.py", "secret-assign", "ec0dce3ef3667b6d"): (1, "the same LANE_TOKEN line"),
    # golden check (sha-pinned be4b161f): its docstring's usage example for the fleet's 8 GB node
    ("lib/runner/diloco_golden_check.py", "user-path", "094d2f0047842e07"): (1,
        "a usage example in the docstring (a generic Windows user name)"),
    ("lib/runner/diloco_golden_check.py", "account-path", "b512d97e7cbf97c2"): (1, "the same docstring line"),
    # the Stage-0 runner (sha-pinned): generic, INVENTED test inputs of its redaction / LAN-refusal selftests (a
    # value split into pieces is checked by the JOIN check below, which has no allowlist)
    # (runner 87b73d0c, rebuilt for the 2026-10-15 BLOCKER: every fixture is an invented 'Alex Example' / example_key
    # / documentation-range value, and no owner value is in it whole, split or glued -- the operator's scan reads 0)
    ("lib/runner/glm47_h1x_chain.py", "private-ip", "3779952acebd8a95"): (3,
        "(d-s18) selftest: a GENERIC 192.168.0.x address asserted REFUSED as the data-plane anchor (WAN-only), and "
        "the runner's own privacy scan's allowance of that same synthetic fixture (S0_PRIV_ALLOW_IPS)"),
    ("lib/runner/glm47_h1x_chain.py", "user-path", "99ab01280efac5a2"): (1,
        "a docstring example of a user path with a space (the invented 'Alex Example'), cut by the path stripper"),
    ("lib/runner/glm47_h1x_chain.py", "user-path", "1027d1a7140c8d31"): (3, _FIXTURE),
    ("lib/runner/glm47_h1x_chain.py", "user-path", "65a9e65018a6449e"): (2, _FIXTURE),
    ("lib/runner/glm47_h1x_chain.py", "user-path", "6e1152faf1e0046e"): (1, _FIXTURE),
    ("lib/runner/glm47_h1x_chain.py", "user-path", "7e17970639af410c"): (1, _FIXTURE),
    ("lib/runner/glm47_h1x_chain.py", "user-path", "e22b98c808289450"): (2, _FIXTURE),
    ("lib/runner/glm47_h1x_chain.py", "login", "a41e078c697e735a"): (1, _FIXTURE),
    ("lib/runner/glm47_h1x_chain.py", "ssh-key", "faeffc26071f88d2"): (1, _FIXTURE),
    ("lib/runner/glm47_h1x_chain.py", "ssh-key", "4d9ce3d2a37fcb06"): (2, _FIXTURE + "; and a PIECE of (s0-privacy)'s "
                                                                       "invented example_key fixture"),
    ("lib/runner/glm47_h1x_chain.py", "remote-home", "3d3786f24963f162"): (1, _FIXTURE),
    ("lib/runner/glm47_h1x_chain.py", "remote-home", "c0bd14e65ed265f2"): (1, _FIXTURE),
    ("lib/runner/glm47_h1x_chain.py", "remote-home", "f9e409c5c16ecfdd"): (1,
        "a PIECE of (s0-privacy)'s invented example_key fixture (a one-letter home dir)"),
    ("lib/runner/glm47_h1x_chain.py", "private-repo-path", "e21345e1484e6806"): (1, _FIXTURE),
    # the PUBLIC WG variant (e5b993b4): its owner ssh defaults are gone; what remains names the private repo's dir in
    # the OLMoE chain's own defaults / comments (role_chain, which no node runs)
    ("lib/wan_gradient_chain_ckpt.py", "private-repo-path", "9bfb42878123b162"): (1, _REPO_DIR),
    ("lib/wan_gradient_chain_ckpt.py", "private-repo-path", "e21345e1484e6806"): (3, _REPO_DIR),
    ("tools/sharddiloco_glm_expert.py", "private-repo-path", "e21345e1484e6806"): (4, _REPO_DIR),
    # p2p_transport (sha-pinned eca89c4d): its module docstring names the sidecar's build worktree (<repo dir>_p2p_wt)
    ("lib/runner/p2p_transport.py", "private-repo-path", "e21345e1484e6806"): (1, _REPO_DIR),
}

SCAN = [
    ("key-block", re.compile(rb"-----BEGIN [A-Z ]*(PRIVATE KEY|CERTIFICATE)-----")),
    ("api-token", re.compile(rb"\bhf_[A-Za-z0-9]{30,}\b|\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b|"
                             rb"github_pat_[A-Za-z0-9_]{40,}|\bAKIA[0-9A-Z]{16}\b|xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("hex32", re.compile(rb"(?<![0-9a-fA-F])[0-9a-f]{32}(?![0-9a-fA-F])")),
    ("password", re.compile(rb"(?i)\b(password|passwd|turn_pass|secret|token)\s*[:=]\s*[\"'][^\"'\s{}%]{6,}[\"']")),
    ("email", re.compile(rb"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.(com|net|org|io|ai|dev|co)\b")),
    ("private-ip", re.compile(rb"\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|"
                              rb"172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b")),
    ("user-path", re.compile(rb"(?i)\b[A-Z]:(\\\\|\\|/)+Users(\\\\|\\|/)+[A-Za-z0-9_.-]+")),
    ("login", re.compile(rb"\b[a-z_][a-z0-9_-]*@\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
    # 2026-10-15 publish brief: carrier-grade NAT addresses (a CIDR range constant is not an address), ssh key FILES,
    # remote home dirs, X-Auth header literals, a Windows account as a path segment, absolute paths into the private
    # repo's directory, and secret-named assignments of long literals (an env-var NAME is not a secret)
    ("cgnat-ip", re.compile(rb"(?<![\d.])100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}(?![\d])(?!/\d)")),
    ("ssh-key", re.compile(rb"(?i)\.ssh(\\\\|\\|/)|\bid_(rsa|dsa|ecdsa|ed25519)\b|(?<=[\\/\"'])[A-Za-z0-9_-]+_ed25519\b|"
                           rb"\.pem\b|\.ppk\b")),
    ("remote-home", re.compile(rb"/home/[a-z_][a-z0-9_-]*")),
    ("x-auth", re.compile(rb"(?i)x-auth[\"']?\s*[:=,]\s*[\"'][^\"'\s{}%<>]{8,}[\"']")),
    ("account-path", re.compile(rb"(?<=[\\/])User(?=[\\/\"'])")),
    # no trailing \b: a sibling dir such as <drive>:/<repo dir>_p2p_wt (a worktree) names the private repo too
    ("private-repo-path", re.compile(rb"(?i)\b[A-Z]:(\\\\|\\|/)+aiCrypto|/[a-z]/aiCrypto")),
    # case-insensitive NAME only: the env-var-NAME exclusion (an ALL-CAPS identifier value) must stay case-sensitive
    ("secret-assign", re.compile(rb"\b[\w-]*(?i:token|secret|psk|hmac|passw(?:or)?d|api[_-]?key|signal[_-]?key)[\w-]*"
                                 rb"[\"']?\s*[:=]\s*[\"'](?![A-Z][A-Z0-9_]{3,}[\"'])[^\"'\s{}%<>]{12,}[\"']")),
]


def _files(root=ROOT):
    out = []
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if x not in (".git", "__pycache__", ".pytest_cache")]
        for f in fs:
            out.append(os.path.relpath(os.path.join(d, f), root).replace("\\", "/"))
    return sorted(out)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _manifest():
    rows = {}
    with open(MANIFEST, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                sha, rel = ln.split(None, 1)
                rows[rel.lstrip("*")] = sha
    return rows


def test_manifest_covers_every_file_and_every_hash_matches():
    if not os.path.isfile(MANIFEST):
        pytest.skip("MANIFEST.sha256 not generated yet")
    rows = _manifest()
    files = [f for f in _files() if f != "MANIFEST.sha256"]
    missing = [f for f in files if f not in rows]
    extra = [f for f in rows if f not in files]
    bad = [f for f in files if f in rows and _sha(os.path.join(ROOT, *f.split("/"))) != rows[f]]
    assert not missing and not extra and not bad, (
        "MANIFEST.sha256 vs disk: %d unlisted %s, %d listed but absent %s, %d hash mismatches %s -- a hash mismatch "
        "right after a clone means Git rewrote line endings: keep glm-node/.gitattributes"
        % (len(missing), missing[:5], len(extra), extra[:5], len(bad), bad[:5]))


def _pins():
    with open(PINS, encoding="utf-8") as f:
        return json.load(f)["files"]


def test_the_fleet_compared_files_carry_the_fleet_hashes():
    pins = _pins()
    assert "lib/runner/glm47_h1x_chain.py" in pins and "lib/runner/p2p_transport.py" in pins
    bad = dict((f, (_sha(os.path.join(ROOT, *f.split("/")))[:16]
                    if os.path.isfile(os.path.join(ROOT, *f.split("/"))) else "MISSING"))
               for f, ent in pins.items()
               if not os.path.isfile(os.path.join(ROOT, *f.split("/")))
               or _sha(os.path.join(ROOT, *f.split("/"))) != ent["sha256"])
    assert not bad, "files differ from what the fleet runs (a version skew the run would refuse): %s" % bad


def test_the_example_card_names_the_shipped_runner():
    with open(os.path.join(ROOT, "examples", "run_card.example.json"), encoding="utf-8") as f:
        card = json.load(f)
    pins = _pins()
    assert card["runner_sha256"] == pins["lib/runner/glm47_h1x_chain.py"]["sha256"]
    assert card["p2p_transport_sha256"] == pins["lib/runner/p2p_transport.py"]["sha256"]


def test_gitattributes_keeps_every_byte():
    with open(os.path.join(ROOT, ".gitattributes"), encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    assert "* -text" in lines, "glm-node/.gitattributes must say `* -text` (no end-of-line conversion)"


def test_git_would_not_rewrite_line_endings():
    """With core.autocrlf=true (Git for Windows' default) the attribute must win for every pinned file."""
    pins = sorted(_pins())
    try:
        out = subprocess.run(["git", "-C", ROOT, "check-attr", "text", "--"] + pins,
                             capture_output=True, encoding="utf-8", errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git not available")
    if out.returncode != 0:
        pytest.skip("not inside a git checkout")
    unset = [ln for ln in out.stdout.splitlines() if not ln.rstrip().endswith(": unset")]
    assert not unset, "git would treat these as text (and may rewrite them): %s" % unset


def test_no_go_test_files_ship():
    """The sidecar's Go tests carried the operator's own network addresses; they are not needed to build it."""
    shipped = [f for f in _files() if f.startswith("sidecar/") and f.endswith("_test.go")]
    assert not shipped, shipped


def scan(root=ROOT):
    """-> [(file, line, kind, sha16 of the matched text)] for every hit in every non-binary file."""
    hits = []
    for rel in _files(root):
        if root == ROOT and rel == "tests/test_glm_node_manifest.py":
            continue
        with open(os.path.join(root, *rel.split("/")), "rb") as fh:
            data = fh.read()
        if b"\0" in data[:8192]:
            continue                                        # binaries: the sidecar exe is pinned by its sha256
        for kind, rx in SCAN:
            for m in rx.finditer(data):
                if kind == "hex32" and re.match(rb"^[0-9]+$", m.group(0)):
                    continue
                hits.append((rel, data.count(b"\n", 0, m.start()) + 1, kind,
                             hashlib.sha256(m.group(0)).hexdigest()[:16]))
    return hits


def _unexplained(hits, allowed):
    counts = {}
    for rel, line, kind, h in hits:
        counts.setdefault((rel, kind, h), []).append(line)
    bad = []
    for key, lines in sorted(counts.items()):
        want = allowed.get(key, (0, ""))[0]
        if len(lines) != want:
            bad.append("%s: %s x%d (allowed %d) at lines %s" % (key[0], key[1], len(lines), want, lines[:6]))
    return bad, counts


# ---- the JOIN check. A value split into pieces ("ab" + "cd", adjacent literals "ab" "cd", "".join(["ab", "cd"]),
# f"ab{'cd'}") never appears whole in the source text, so the substring scan above only sees fragments. Every
# constant-only string expression built from 2+ pieces is folded with the AST; a SCAN (or JOIN_EXTRA) hit in the
# folded value that lies inside no single piece fails. There is NO allowlist for it: a pinned file that carries one
# must be rebuilt.
JOIN_EXTRA = [   # PARTS of a login / LAN address are suspicious once glued back together
    ("login-prefix", re.compile(rb"\b[a-z_][a-z0-9_.-]*@\d{1,3}\.\d{1,3}(\.\d{1,3})?\b")),
    ("lan-prefix", re.compile(rb"(?<![\d.])(10\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3})"
                              rb"\.?(?![\d])")),
]
_ADJACENT = re.compile(r"""['"]\s*(?:\\\r?\n\s*)*[rRbBuUfF]{0,2}['"]""")


def _fold(node, memo):
    """-> (folded str, [piece strings]) for a constant-only string expression, else None."""
    if id(node) in memo:
        return memo[id(node)]
    r = None
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
        v = node.value if isinstance(node.value, str) else node.value.decode("latin-1")
        r = (v, [v])
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a = _fold(node.left, memo)
        b = _fold(node.right, memo) if a else None
        r = (a[0] + b[0], a[1] + b[1]) if a and b else None
    elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join"
          and isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str)
          and len(node.args) == 1 and not node.keywords and isinstance(node.args[0], (ast.List, ast.Tuple))):
        parts = [_fold(e, memo) for e in node.args[0].elts]
        if parts and all(parts):
            r = (node.func.value.value.join(p[0] for p in parts), [c for p in parts for c in p[1]])
    elif isinstance(node, ast.JoinedStr):
        vals, pieces = [], []
        for v in node.values:
            if isinstance(v, ast.FormattedValue):
                f = _fold(v.value, memo) if v.conversion in (-1, None) and v.format_spec is None else None
            else:
                f = _fold(v, memo)
            if not f:
                vals = None
                break
            vals.append(f[0])
            pieces += f[1]
        r = ("".join(vals), pieces) if vals is not None else None
    memo[id(node)] = r
    return r


def _string_tokens(segment):
    try:
        return [t.string for t in tokenize.generate_tokens(io.StringIO(segment).readline) if t.type == tokenize.STRING]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []


def _folded(src):
    """-> [(line, folded value, [pieces])] for every MAXIMAL constant-only string expression of 2+ pieces."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return []
    blines = [ln.encode("utf-8") for ln in src.splitlines(keepends=True)]

    def segment(n):                                         # col offsets are UTF-8 byte offsets
        if n.lineno == n.end_lineno:
            b = blines[n.lineno - 1][n.col_offset:n.end_col_offset]
        else:
            b = b"".join([blines[n.lineno - 1][n.col_offset:]] + blines[n.lineno:n.end_lineno - 1]
                         + [blines[n.end_lineno - 1][:n.end_col_offset]])
        return b.decode("utf-8", "replace")
    parent = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            parent[id(ch)] = node
    memo, out = {}, []
    for node in ast.walk(tree):
        f = _fold(node, memo)
        if not f or (id(node) in parent and _fold(parent[id(node)], memo)):
            continue                                       # not foldable, or not the maximal expression
        pieces = f[1]
        if isinstance(node, ast.Constant):                 # one Constant: pieces only if 2+ adjacent literals
            seg = segment(node)
            toks = _string_tokens(seg) if _ADJACENT.search(seg) else []
            if len(toks) < 2:
                continue
            try:
                pieces = [ast.literal_eval(t) for t in toks]
            except (ValueError, SyntaxError):
                continue
            pieces = [p.decode("latin-1") if isinstance(p, bytes) else p for p in pieces]
        elif len(pieces) < 2:
            continue
        out.append((node.lineno, f[0], pieces))
    return out


def join_scan(root=ROOT):
    """-> [(file, line, 'joined:<kind>', sha16 of the glued match)] over every .py file of a tree."""
    hits = []
    for rel in _files(root):
        if not rel.endswith(".py") or (root == ROOT and rel == "tests/test_glm_node_manifest.py"):
            continue
        with open(os.path.join(root, *rel.split("/")), encoding="utf-8", errors="replace") as fh:
            src = fh.read()
        for line, val, pieces in _folded(src):
            q = ('"' + val + '"').encode("utf-8", "replace")
            bp = [p.encode("utf-8", "replace") for p in pieces]
            for kind, rx in SCAN + JOIN_EXTRA:
                for m in rx.finditer(q):
                    txt = m.group(0).strip(b'"')
                    if txt and not any(txt in p for p in bp) and not (kind == "hex32" and txt.isdigit()):
                        hits.append((rel, line, "joined:" + kind, hashlib.sha256(txt).hexdigest()[:16]))
    return hits


def test_no_private_data_hides_in_split_literals():
    hits = join_scan()
    assert not hits, ("values split into pieces that join into a private-data shape (no allowlist: rebuild the file "
                      "without them):\n  " + "\n  ".join("%s:%d %s %s" % h for h in hits))


def test_the_join_check_catches_planted_split_values(tmp_path):
    """POSITIVE CONTROL for the JOIN check: each split shape is glued back and caught (INVENTED values only); a value
    already whole in ONE piece is the substring scan's job, and a harmless join is not a hit."""
    planted = {
        "a_plus.py": ('R = "/ho" + "me/someone/x"\n', "joined:remote-home"),
        "b_adjacent.py": ('L = ("root"\n     "@203.0.113.9")\n', "joined:login"),
        "c_join.py": ('K = "".join(["id_", "ed25519"])\n', "joined:ssh-key"),
        "d_fstring.py": ('N = f"192.{\'168\'}.1.20"\n', "joined:private-ip"),
        "e_login_part.py": ('W = "ops" + "@203.0"\n', "joined:login-prefix"),
        "f_lan_part.py": ('S = "192.168." + "77."\n', "joined:lan-prefix"),
        "g_repo_dir.py": ('P = "E:/ai" + "Crypto/x"\n', "joined:private-repo-path"),
    }
    harmless = ['A = "abc" + "def"\n', 'B = "192.168.1.20" + "x"\n', 'C = "py" "thon"\n',
                'D = ", ".join(["a", "b"])\n', 'E = f"{x}-y"\n']
    for name, (src, _k) in planted.items():
        (tmp_path / name).write_text(src, encoding="utf-8")
    for i, src in enumerate(harmless):
        (tmp_path / ("z%d_harmless.py" % i)).write_text(src, encoding="utf-8")
    hits = join_scan(str(tmp_path))
    missed = [n for n, (_s, k) in planted.items() if k not in [h[2] for h in hits if h[0] == n]]
    false = [h for h in hits if h[0].endswith("_harmless.py")]
    assert not missed and not false, ("missed %s, false hits %s" % (missed, false))


# ---- the GLUED-CALL check. A call that glues constant pieces at RUN TIME (glue("192.168.", "1.20"),
# os.path.join("C:/", "Users", "x"), "{}{}".format(..), "%s%s" % (..)) never holds the value whole either, and the JOIN
# check above does not follow calls. Every call whose 2+ positional arguments are all constant-only strings (or such
# calls), and every constant format with constant arguments, is glued with "" and with "/" (and evaluated for a
# format); a SCAN / JOIN_EXTRA hit in a glued value that lies inside no single piece fails unless it is listed in
# GLUE_ALLOWED by (file, kind, sha16 of the glued match) with its exact count. The Stage-0 runner (sha-pinned) builds
# the VISIBLY FICTIONAL "allowed" fixtures of its own privacy selftest (s0-privacy) this way on purpose, so that its own
# literal scan never sees them whole; those, and only those, are listed. (The operator's private pre-publish scan
# glues every such call too and checks it against the operator's actual values: 0.)
_GLUE_FIXTURE = ("the Stage-0 runner's (s0-privacy) selftest: a visibly FICTIONAL value (example names, a documentation "
                 "address) glued at run time by _s0_cat, which its own scan must NOT flag (its negative control)")
GLUE_ALLOWED = {   # (s0-privacy)'s "allowed" list: the SAME invented values the selftests above use whole
    ("lib/runner/glm47_h1x_chain.py", "glued:private-ip", "3779952acebd8a95"): (1,
        _GLUE_FIXTURE + ": the generic 192.168.0.x fixture of (d-s18), = its ALLOWED hash above"),
    ("lib/runner/glm47_h1x_chain.py", "glued:lan-prefix", "99844cc00ff1c9a2"): (1, "the same 192.168.0.x fixture"),
    ("lib/runner/glm47_h1x_chain.py", "glued:user-path", "1027d1a7140c8d31"): (1,
        _GLUE_FIXTURE + ": the invented 'Alex Example' profile, = its ALLOWED hash above"),
    ("lib/runner/glm47_h1x_chain.py", "glued:login", "a41e078c697e735a"): (1,
        _GLUE_FIXTURE + ": a login on a documentation address (TEST-NET-3), = its ALLOWED hash above"),
    ("lib/runner/glm47_h1x_chain.py", "glued:login-prefix", "6c35c6dcc090cbd9"): (1, "the same TEST-NET-3 login"),
}


def _glue_parts(node, memo):
    """-> [(value, [pieces]), ...] for the 2+ constant positional arguments of a call (nested such calls glued with
    ""), else None."""
    key = ("glue", id(node))
    if key in memo:
        return memo[key]
    r = None
    if (isinstance(node, ast.Call) and not node.keywords and len(node.args) >= 2
            and not any(isinstance(a, ast.Starred) for a in node.args)):
        parts = []
        for a in node.args:
            f = _fold(a, memo)
            if not f:
                sub = _glue_parts(a, memo)
                f = ("".join(p[0] for p in sub), [c for p in sub for c in p[1]]) if sub else None
            if not f:
                parts = None
                break
            parts.append(f)
        r = parts
    memo[key] = r
    return r


def _format_value(node, memo):
    """-> (value, [pieces]) for "fmt" % (constants) or "fmt".format(constants) with a constant format, else None."""
    fmt, args = None, None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        f = _fold(node.left, memo)
        if f:
            elts = node.right.elts if isinstance(node.right, ast.Tuple) else [node.right]
            vals = [_fold(e, memo) for e in elts]
            if vals and all(vals):
                fmt, args = f, vals
    elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format"
          and not node.keywords and node.args):
        f = _fold(node.func.value, memo)
        vals = [_fold(a, memo) for a in node.args]
        if f and all(vals):
            fmt, args = f, vals
    if fmt is None or len(fmt[0]) > 2000 or re.search(r"%\d{4,}|\{[^}]*:[^}]*\d{4,}", fmt[0]):
        return None
    try:
        v = (fmt[0] % tuple(a[0] for a in args) if isinstance(node, ast.BinOp)
             else fmt[0].format(*[a[0] for a in args]))
    except (TypeError, ValueError, KeyError, IndexError):
        return None
    return v, fmt[1] + [c for a in args for c in a[1]]


def _glued(src):
    """-> [(line, [candidate values], [pieces])] for every MAXIMAL glued call / constant format of a source."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return []
    parent = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            parent[id(ch)] = node
    memo, out = {}, []

    def cands(node):                                       # in order: glued with "", the format's value, with "/"
        c = []
        parts = _glue_parts(node, memo)
        if parts:
            c.append(("".join(p[0] for p in parts), [x for p in parts for x in p[1]]))
        fv = _format_value(node, memo)
        if fv:
            c.append(fv)
        if parts:
            c.append(("/".join(p[0] for p in parts), [x for p in parts for x in p[1]]))
        return c
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Call, ast.BinOp)):
            continue
        c = cands(node)
        if not c:
            continue
        p = parent.get(id(node))
        if p is not None and isinstance(p, (ast.Call, ast.BinOp)) and cands(p):
            continue                                       # not maximal: the enclosing call glues it in
        out.append((node.lineno, [v for v, _p in c], sorted(set(x for _v, ps in c for x in ps))))
    return out


def glue_scan(root=ROOT):
    """-> [(file, line, 'glued:<kind>', sha16 of the glued match)], one per (call, kind, match) over every .py file."""
    hits = []
    for rel in _files(root):
        if not rel.endswith(".py") or (root == ROOT and rel == "tests/test_glm_node_manifest.py"):
            continue
        with open(os.path.join(root, *rel.split("/")), encoding="utf-8", errors="replace") as fh:
            src = fh.read()
        for line, values, pieces in _glued(src):
            bp = [p.encode("utf-8", "replace") for p in pieces]
            seen = set()
            for val in values:                              # a later spelling adds only kinds not yet found
                kinds_before = set(k for k, _h in seen)
                q = ('"' + val + '"').encode("utf-8", "replace")
                for kind, rx in SCAN + JOIN_EXTRA:
                    if kind in kinds_before:
                        continue
                    for m in rx.finditer(q):
                        txt = m.group(0).strip(b'"')
                        if txt and not any(txt in p for p in bp) and not (kind == "hex32" and txt.isdigit()):
                            seen.add((kind, hashlib.sha256(txt).hexdigest()[:16]))
            hits += [(rel, line, "glued:" + k, h) for k, h in sorted(seen)]
    return hits


def test_no_private_data_hides_in_glued_calls():
    bad, counts = _unexplained(glue_scan(), GLUE_ALLOWED)
    stale = [k for k, (n, _w) in GLUE_ALLOWED.items() if len(counts.get(k, [])) != n]
    assert not bad and not stale, ("values glued by a call into a private-data shape:\n  " + "\n  ".join(bad)
                                   + ("\nstale GLUE_ALLOWED entries: %s" % stale if stale else ""))


def test_the_glue_check_catches_planted_calls(tmp_path):
    """POSITIVE CONTROL for the GLUED-CALL check (INVENTED values only): a helper call, a nested call, a path join,
    a format and a % format are glued back and caught; a call with a variable piece and a harmless call are not."""
    planted = {
        "a_helper.py": ('X = glue("192.168.", "77.20")\n', "glued:private-ip"),
        "b_nested.py": ('X = glue(glue("ro", "ot"), "@203.0.113.9")\n', "glued:login"),
        "c_path.py": ('X = os.path.join("C:/", "Users", "someone")\n', "glued:user-path"),
        "d_format.py": ('X = "{}{}".format("/ho", "me/someone")\n', "glued:remote-home"),
        "e_percent.py": ('X = "%s.%s" % ("10.1", "2.3")\n', "glued:private-ip"),
    }
    harmless = ['X = glue("192.168.", y)\n', 'X = print("abc", "def")\n', 'X = os.path.join("a", "b")\n',
                'X = "%s-%s" % ("a", "b")\n']
    for name, (src, _k) in planted.items():
        (tmp_path / name).write_text(src, encoding="utf-8")
    for i, src in enumerate(harmless):
        (tmp_path / ("z%d_harmless.py" % i)).write_text(src, encoding="utf-8")
    hits = glue_scan(str(tmp_path))
    missed = [n for n, (_s, k) in planted.items() if k not in [h[2] for h in hits if h[0] == n]]
    false = [h for h in hits if h[0].endswith("_harmless.py")]
    assert not missed and not false, ("missed %s, false hits %s" % (missed, false))
    # and an allowlisted glued hit with the wrong count is reported
    key = [(h[0], h[2], h[3]) for h in hits if h[0] == "a_helper.py" and h[2] == "glued:private-ip"][0]
    bad, _ = _unexplained(glue_scan(str(tmp_path)), {key: (2, "")})
    assert any("a_helper.py: glued:private-ip x1 (allowed 2)" in b for b in bad), bad


def test_nothing_secret_or_private_ships():
    bad, _ = _unexplained(scan(), ALLOWED)
    assert not bad, "possible secrets / private data:\n  " + "\n  ".join(bad)


def test_every_allowlisted_hit_still_exists():
    """A stale allowlist entry is a pre-approved hole: each must still match exactly its count of hits."""
    _, counts = _unexplained(scan(), ALLOWED)
    stale = [k for k, (n, _w) in ALLOWED.items() if len(counts.get(k, [])) != n]
    assert not stale, "stale ALLOWED entries (fix their counts or delete them): %s" % stale


def test_the_scan_catches_planted_secrets(tmp_path):
    """POSITIVE CONTROL: each pattern must fire on a planted example (built at run time so this file holds none)."""
    t = "".join(["ab", "cd"] * 8)
    planted = {
        "a.py": "K = '" + "-----BEGIN " + "PRIVATE KEY-----'\n",
        "b.py": "T = 'hf_" + "x" * 34 + "'\n",
        "c.py": "TOKEN = '" + t + "'\n",
        "d.py": "password = '" + "hunter22" + "'\n",
        "e.md": "mail me at someone" + "@example.com\n",
        "f.py": "LAN = '192." + "168.1.20'\n",
        "g.py": "P = r'C:" + "\\Users\\someone\\x'\n",
        "h.py": "SSH = 'root" + "@203.0.113.9'\n",
        "i.py": "CG = '100." + "64.3.4'\n",
        "j.py": "K = '~/." + "ssh/x'\n",
        "k.py": "R = '/home/" + "someone/x'\n",
        "l.py": "H = {'X-" + "Auth': 'abcdefgh12345'}\n",
        "m.py": "P = 'D:/x/" + "User/y'\n",
        "n.py": "P = 'E:/" + "aiCrypto/x'\n",
        "o.py": "api_" + "key = 'abcdefABCDEF123456'\n",
    }
    for name, body in planted.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    kinds = sorted(set(k for _r, _l, k, _h in scan(str(tmp_path))))
    assert kinds == sorted(["key-block", "api-token", "hex32", "password", "email", "private-ip", "user-path",
                            "login", "cgnat-ip", "ssh-key", "remote-home", "x-auth", "account-path",
                            "private-repo-path", "secret-assign"]), kinds
    # NEGATIVE shapes the new kinds must NOT match (a range constant, a flag name, an env-var name, the public anchor)
    neg = tmp_path / "neg"
    neg.mkdir()
    (neg / "z.py").write_text("NET = '100.64.0.0/10'\nhave_ed25519 = True\n'turn_password_env': 'NEURAHASH_TURN_PASSWORD'\n"
                              "A = 'http://47.84.93.96:8710'\nU = 'User-Agent'\n", encoding="utf-8")
    assert scan(str(neg)) == [], scan(str(neg))
    # and an allowlisted (file, kind, hash) with the wrong count is reported
    bad, _ = _unexplained(scan(str(tmp_path)), {("c.py", "hex32", hashlib.sha256(t.encode()).hexdigest()[:16]): (2, "")})
    assert any("c.py: hex32 x1 (allowed 2)" in b for b in bad), bad


def test_no_control_bytes_in_shipped_text():
    """A stray control byte (an escape that went wrong in an edit) once turned a regex into a silent no-op."""
    bad = []
    for rel in _files():
        with open(os.path.join(ROOT, *rel.split("/")), "rb") as fh:
            data = fh.read()
        if b"\0" in data[:8192]:
            continue
        hits = [i for i, b in enumerate(data) if b < 32 and b not in (9, 10, 13)]
        if hits:
            bad.append("%s: %d control byte(s), first at line %d" % (rel, len(hits), data.count(b"\n", 0, hits[0]) + 1))
    assert not bad, bad
