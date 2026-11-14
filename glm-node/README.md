# glm-node — join the GLM-4.7-Flash training trial as a node

**What this is.** An ENGINEERING trial of decentralized training: several home PCs on different internet
connections train one model together (GLM-4.7-Flash, a 30B mixture-of-experts LLM; each node trains a LoRA
adapter with DiLoCo — local steps, then a shared average once per round). Peers exchange their round updates over
the internet through a relay and, when it is faster, directly peer-to-peer (WebRTC, through a TURN server).

**What this is NOT.** Not a paid or mining lane: nothing is minted, no wallet is involved, no reward is promised. It
is experimental software run by one operator; runs are scheduled with the operator, and a run can be stopped,
voided or restarted at any time. Join only if you are happy to lend a GPU for testing.

**What your PC does.** It reads the model's 55.6 GB of expert weights from your SSD about twice per training step
(~100 GB per step, measured), trains on a public science-QA corpus, and uploads one lossless-compressed update
(~37 MB) per round to the operator's relay. Details:
[What the node sends](#what-the-node-sends-and-to-whom).

---

## Requirements

| | minimum | why / measured on the fleet's own 8 GB node (RTX 4060, Windows 11) |
|---|---|---|
| GPU | NVIDIA, **8 GB VRAM** (RTX 3070 / 4060 class or bigger) | the node runs with a 6.5 GiB VRAM cap; measured peaks 5.6-5.8 GiB |
| NVIDIA driver | **570 or newer** (CUDA 12.8) | torch 2.9.1+cu128 |
| RAM | **16 GB** (32 GB recommended) | 4 GiB is pinned for read-ahead; free RAM caches packs = faster steps |
| Pagefile (Windows) | **"System managed size"** (the Windows default) | a node commits ~18.5 GB (RAM + pagefile): a 16 GB PC fits only with a system-managed pagefile; with the pagefile off or too small the node dies mid-run ("the paging file is too small", os error 1455). The self-check's `commit` line WARNs when the commit limit is too small |
| Disk | **~75 GB free on an NVMe SSD** | 55.6 GB packs + 5.7 GB trunk + run files; every step reads ~100-110 GB from it (a 4060 node: 46 s per step measured with the read-ahead settings the launcher uses, ~75 s without) |
| OS | Windows 10/11 64-bit (tested); Linux works (see [Linux](#linux-notes)) | |
| Software | **Python 3.10–3.13 (64-bit)**, **Git**, the NVIDIA driver | nothing else needs to be pre-installed |
| Network | outbound only: HTTP to `47.84.93.96:8710` (relay), TCP+UDP `3478` (TURN/STUN), HTTPS to `huggingface.co` (one-time ~65 GB download) | no inbound ports, no port forwarding |

If you are unsure, run the self-check ([section 4](#4-self-check)) first: it measures all of this and prints PASS/FAIL.

## Quick start (Windows, PowerShell)

```powershell
git clone https://github.com/whitegreenstudios/neurahash-miner.git D:\glmnode\neurahash-miner
cd D:\glmnode\neurahash-miner
git checkout glm-node-trial-1          # the tag the operator names for the run
python -m venv D:\glmnode\venv
D:\glmnode\venv\Scripts\Activate.ps1
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r glm-node\requirements-node.txt
$env:NEURAHASH_GLM_HOME = "D:\glmnode"   # where packs / base / corpus / runs go (pick an NVMe drive)
python glm-node\node_selfcheck.py        # 1. is this PC suitable?        (paste the report to the operator)
python glm-node\build_packs.py           # 2. model files, ~65 GB download (resumable, verified byte-for-byte)
python glm-node\fetch_corpus.py --rebuild   # 3. the training corpus (rebuilt from public datasets, sha-checked)
python glm-node\glm_node.py keygen       # 4. your P2P key; send the PUBKEY line to the operator
# ... the operator sends you a run card and your credentials ...
python glm-node\glm_node.py preflight --card run_card.json
python glm-node\glm_node.py run --card run_card.json
```

If PowerShell refuses `Activate.ps1` ("running scripts is disabled"), call the venv's interpreter directly
(`D:\glmnode\venv\Scripts\python.exe` wherever this page says `python`), or activate from cmd.exe with
`D:\glmnode\venv\Scripts\activate.bat`.

Keep the clone, the venv and `NEURAHASH_GLM_HOME` outside your user profile (e.g. under `D:\glmnode`, not
`C:\Users\<you>\...`): the node's run record (TRAIN.json) includes its command line, its interpreter path and its
local paths, and it is uploaded to the relay.

## 1. Install

1. Install **Python 3.10–3.13 (64-bit)** from python.org (tick "Add python.exe to PATH") and **Git** from
   git-scm.com. Install or update the NVIDIA driver (570+).
2. Clone **without** `--depth` and check out the tag the operator gives you:
   `git clone https://github.com/whitegreenstudios/neurahash-miner.git`, then `git checkout glm-node-trial-1`.
   glm-node is a separate folder in the public miner repository. It is not part of the miner's signed
   auto-update, and nothing here changes how the miner itself runs. Use this checkout only for glm-node: the
   miner's own tools self-update by checking out the signed release commit, which could move the files under a
   running node.
3. Use a dedicated virtual environment (the versions below are pinned to what the fleet runs; mixing them with other
   projects' packages causes silent differences):
   `python -m venv D:\glmnode\venv` then `D:\glmnode\venv\Scripts\Activate.ps1`
   (cmd.exe: `D:\glmnode\venv\Scripts\activate.bat`).
4. `python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128`
   then `python -m pip install -r glm-node\requirements-node.txt`.

Do not edit files in `glm-node/`: every file is pinned by `glm-node/MANIFEST.sha256`, and the training run refuses a
node whose code differs from the fleet's. `glm-node/.gitattributes` makes Git check the files out byte-for-byte;
keep it.

## 2. Build the model files

```powershell
python glm-node\build_packs.py
```

* Downloads `zai-org/GLM-4.7-Flash` (MIT licence) at the pinned revision `7dd20894a642a0aa287e9827cb1a1f7f91386b67`,
  one ~1.3 GB shard at a time, and re-arranges each MoE layer's 64 routed experts into one
  `packs\layer_<L>.safetensors` (46 files × 1,207,959,728 bytes). The packs are pure byte copies of the original
  BF16 tensors, so the result is deterministic: every pack is hashed while it is written and installed only if its
  sha256 equals the fleet's (`glm-node/manifests/packs_manifest.json`). Each source shard is also checked against
  Hugging Face's own sha256, then deleted (use `--keep-source` to keep them).
* Downloads the shared trunk (5,668,965,432 bytes: embeddings, attention, shared experts) from the public dataset
  `whitegreenstudios888/neurahash-data` (pinned revision), and writes `base\config\` (config.json + tokenizer) and
  `base\model_manifest.json`, all sha256-checked.
* Re-run it any time: it resumes and skips files that already verify. `--verify-only` re-hashes everything on disk.
  `--source DIR` builds from a local copy of the model instead of downloading.
* Time: dominated by the ~65 GB download (60 GB of model shards + the 5.7 GB trunk); the build itself is a few
  minutes on an NVMe SSD.

## 3. Get the corpus

```powershell
python glm-node\fetch_corpus.py --rebuild
```

Rebuilds `corpus\e5_scimc_train.jsonl` (19,792 science-QA rows; sha256 `f9b00e91e50d9eea...`) from the public
datasets allenai/ai2_arc, allenai/openbookqa and allenai/sciq at pinned revisions (~14 MB download, under a
minute), and checks the sha256. It also downloads the evaluation sets of HellaSwag, WinoGrande, BoolQ and PIQA, but
only to REMOVE overlapping questions; none of their text is in the corpus. It needs the tokenizer from step 2
(every row stores token ids). The datasets keep their own licences (sciq CC-BY-NC-3.0 = non-commercial, ai2_arc
CC-BY-SA-4.0; the full list is in `fetch_corpus.py`): you download them from their owners.
If the operator gives you the file instead: `python glm-node\fetch_corpus.py --from-file PATH` (or `--from-url URL`)
— the sha256 is checked either way.

## 4. Self-check

```powershell
python glm-node\node_selfcheck.py --disk-test --json D:\glmnode\selfcheck.json
```

One line per check — first the code's sha256s against `MANIFEST.sha256`, then GPU / VRAM (total and free) / driver,
a CUDA probe in a child process, RAM, disk space and speed, Python and package versions, the model files, the corpus,
**each network path separately** — relay HTTP (GET /health plus timed requests), STUN over UDP and over TCP, TURN over
TCP and over UDP (a real allocation when your TURN credentials are set, released at once; otherwise only the
server's challenge), the P2P sidecar (started locally) — your clock against the relay's, and the **golden check**
(the shared-average math on your CPU must give the fleet's digest; ~2 min, skip with `--quick`). With
`--relay-write-test` it also uploads 1 MiB once with your token (content-addressed: it cannot overwrite anything) to
prove the token and measure your uplink. It ends with the fallback ladder this node will use, e.g.

```
FALLBACK LADDER: P2P direct: UNTESTED (needs a peer) -> TURN TCP: OK -> TURN UDP: OK -> relay HTTP: OK
=> this node will fetch peers' parts via P2P over TURN TCP when that is faster, and fall back to the relay automatically
RESULT: READY (0 FAIL, 1 WARN)
```

Paste the whole output to the operator. It contains no secrets, no hostname, and your public IP only masked.
`--no-gpu` skips every CUDA call (and never sets `CUDA_VISIBLE_DEVICES=-1` around a CUDA probe, which crashes some
NVIDIA drivers).

## 5. Join a run

**Credentials** are per miner, issued to you alone by the operator (a self-service registration step may replace
this later), and live only in environment variables — never in a file in this folder, never in the run card (the
launcher writes your TURN user NAME into the node's generated P2P config; the password and the token never touch
the disk):

```powershell
$env:NEURAHASH_CONTENT_TOKEN = "<your relay token>"      # required
$env:NEURAHASH_TURN_USER     = "<your TURN user>"        # for P2P through TURN
$env:NEURAHASH_TURN_PASSWORD = "<your TURN password>"    # for P2P through TURN
```

These last for the PowerShell window (recommended). `setx NAME value` stores them for your user permanently; delete
them with `setx NAME ""` after the trial. On Linux: `export NAME=value`. Without TURN credentials the node still
trains — it uses the relay only.

**P2P key.** `python glm-node\glm_node.py keygen` creates `D:\glmnode\keys\p2p_sign.key` (keep it) and prints a
`PUBKEY <hex>` line: send that line to the operator. Peers sign their P2P signalling with these keys.

**Run card.** The operator sends a small JSON file for the run (`examples/run_card.example.json` shows the shape):
the run's name and attempt, **your node letter** (your slot), the training arguments every node must share, and
the P2P settings including every node's public key. It contains no secrets.

```powershell
python glm-node\glm_node.py preflight --card run_card.json   # every check, nothing started
python glm-node\glm_node.py run --card run_card.json         # start; keep the window open
```

`run` refuses to start until every check passes — a node that fails *at start* (wrong file, wrong version,
missing credential) either voids the attempt or leaves its peers waiting for it at step 0 until they give up
(45 minutes), which ends the attempt for everyone; so it never gets that far. Then it starts the training process
in its own process group, keeps Windows awake, shows the training log, and writes everything to
`D:\glmnode\runs\<run>\<letter>\`. It sets the VRAM cap from your card (6.5 GiB on an 8 GB card, as on the
fleet's own 8 GB node; never the runner's 23 GiB default) and refuses a cap that does not fit.

**Relay-only mode:** `--relay-only` makes this node fetch everything from the relay. It runs the node with
`--p2p force-relay`, which keeps the run's agreement — do **not** pass `--p2p off` yourself in a run whose card says
`--p2p auto`: a node whose P2P setting differs that way disagrees with its peers at step 0, which voids the attempt
for everyone. The launcher
also steps down to `force-relay` by itself, with a WARN line saying why, when P2P cannot work here: no sidecar
binary (or antivirus blocks it), its sha256 differs from the card, no TURN credentials, your key is not registered
in the card, or a P2P secret it needs is missing — and it finally builds (without starting) the P2P transport with
the runner's own `p2p_transport.py`, so any other static problem is caught too. The runner itself (Stage-0 copy)
also falls back: under `--p2p auto` a P2P setup problem at the node's start (missing or quarantined sidecar, wrong
exe sha256, unreadable key or password) is a WARNING and the node trains relay-only; the launcher checks first so
the reason is printed before anything starts. Once the node runs, a sidecar that fails to start or crashes is only
logged, and the relay carries on.

## 6. Stop, restart, rejoin

* **Stop:** press Ctrl+C once in the node's window: the node leaves at the **next round boundary**, after its
  round checkpoint is written (nothing of the round's work is lost). Press Ctrl+C again to leave at once. Either way
  the node's processes are ended and, for the run, this is a clean *leave*: the run goes on without you and nothing
  shared is damaged. Avoid stopping during the run's **first round** (the launcher warns): a node that drops out
  there can void the attempt. Tell the operator when you stop — a run tolerates only a few missed node-rounds.
* **Unattended:** `run ... --auto-rejoin 3` re-enters the run by itself after a crash (up to 3 times, 2 minutes
  apart, only while the relay answers; never after your own stop, a VOID, a setup refusal, a pause or a refused
  key).
* **Rejoin** (after a crash, a reboot, a network loss, or a stop):
  `python glm-node\glm_node.py run --card run_card.json --rejoin`. It resumes from this node's newest round
  checkpoint (`round_ckpt_<letter>_rNNNN.pt`, the last 2 are kept) and catches up from the relay. Away for more
  than ~30 rounds: add `--from-snapshot`. Keep the same `NEURAHASH_GLM_HOME` (the node's archive must survive).
* **Status:** `python glm-node\glm_node.py status --card run_card.json`.
* **Exit codes:** 0 run complete · 2 setup refused · 3 VOID · 4 resumable stop or pause (rejoin later) · 5 ABORT ·
  7 the relay refused **this node's key** (revoked, expired, outside its scope, or a quota; the run goes on without
  you): ask the operator for a new key, set it in `NEURAHASH_CONTENT_TOKEN`, then rejoin.

Problems: see [TROUBLESHOOTING.md](TROUBLESHOOTING.md). To send the operator your logs:
`python glm-node\glm_node.py logs --card run_card.json` (a zip without checkpoints, keys or tokens).

## What the node sends and to whom

* **Relay** (`http://47.84.93.96:8710`, the operator's content store; plain HTTP): your round updates (one
  lossless-coded part per round, ~37 MB, cut into ≤ 4 MiB resumable chunks), small markers and manifests, and your
  run record `TRAIN.json` (losses, timings, versions, your command line; in the `trial` profile the relay's copy has
  every local path cut to its last part, the file on your disk keeps them). Writes carry your token
  in the `X-Auth` header (plain HTTP: treat the token as a trial credential, not a password you reuse); every
  request carries `X-Miner-Id` = the first 16 hex of sha256("glm47h1x/<letter>") (no secret; the P2P sidecar's
  signalling requests carry their own id). Objects are addressed by their sha256, so an upload cannot change
  anyone else's bytes; which names an upload may bind is the relay's rule (the run's names are operator-scoped).
* **Peers** (P2P, WebRTC data channels through the TURN server `47.84.93.96:3478`): the same round parts, fetched
  by peers when that is faster than the relay; every byte is sha256-checked against the relay's manifest before use.
* **Hugging Face** (downloads only, once): the model, the trunk, the datasets.
* Nothing else: no telemetry, no hostname, no hardware IDs.

## How the transport falls back (automatic)

1. **Uploads always go to the relay first**; P2P only ever *fetches* (the relay is the one path every node needs).
2. **Fetches:** if a peer that holds the part is connected and the relay has not been clearly faster (1.25x), the
   part is fetched over P2P —
   ICE picks the path (TURN-only under the fleet's `ice_policy relay`; direct UDP hole-punching first under
   `nohost`). If no byte arrives in 10 s, bytes stop for 10 s, or P2P looks 1.5× slower than the relay, the relay
   GET starts in parallel and the first sha256-verified copy wins. After 3 failed P2P fetches in 6, the node uses
   the relay for everything, re-probing P2P every 6 fetches or 10 minutes.
3. A P2P sidecar that fails to start or crashes: logged, and the relay carries on. A peer that ever sends a bad byte
   is never used again by this node.
4. **Relay problems:** each chunk gets 3 tries (2 s, then 4 s back-off), then the whole part is retried until the
   round commits without it; a node that cannot reach anything waits, then raises an alarm (~3 h) and stops
   resumably (~4 h, exit 4) — rejoin later. Nothing unverified is ever applied, and the relay keeps an upload only
   when its bytes match its sha256, so a half-uploaded chunk leaves no object.
5. A node missing a round is dropped from that round only (its rows count as dropped); it re-enters the next round.
6. In the `trial` profile (the Stage-0 runner) one node's own failure stays that node's: a bad or out-of-bound part,
   its own speed bound (G1) or loss check, or its void marker EVICTS that node from the round (it stops resumably,
   exit 4, and may `--rejoin`) while the others continue on quorum.

## Linux notes

Same steps with `python3`, `source venv/bin/activate` and `export` instead of `$env:`. The Windows sidecar binary
does not run on Linux: build it with `python glm-node/sidecar/build_sidecar.py` (needs Go; it fetches go1.27.1
itself) — without it the node runs relay-only and still trains. Disable suspend while a node runs
(`systemd-inhibit --what=sleep python glm-node/glm_node.py run ...`).

## Files

| path | what | provenance |
|---|---|---|
| `glm_node.py` | the launcher (preflight, run, rejoin, status, logs, keygen) | new |
| `node_selfcheck.py` | the PASS/FAIL self-check | new |
| `build_packs.py`, `manifests/packs_manifest.json` | model-file builder and the fleet's sha256 of every file | new; byte-identity of the rebuild proven on layers 1, 23, 46 |
| `fetch_corpus.py` | corpus rebuild / fetch, sha256-checked | new |
| `lib/runner/glm47_h1x_chain.py` | the training node | byte-identical to the fleet's Stage-0 runner (sha256 `a2b2158a...`) |
| `lib/runner/p2p_transport.py`, `h1_wire_delta.py`, `diloco_golden_check.py` | P2P transport, wire codec, golden check | byte-identical to the fleet's |
| `lib/runner/bin/p2psidecar.exe` | the P2P sidecar (Go, pion WebRTC), Windows amd64 | rebuilt from `sidecar/src` (sha256 `be38da9a...`, reproducible) |
| `lib/*.py`, `tools/*.py` | the node's libraries and the pack reader (raw packs by default; the `nhz1` codec is optional) | byte-identical to the fleet's, except `lib/wan_gradient_chain_ckpt.py`: the fleet's file with only its two anchor-maintenance defaults (the ssh login and the remote store root; 3 source lines) moved to `NEURAHASH_ANCHOR_SSH` / `NEURAHASH_ANCHOR_STORE_ROOT` (unset = off); the operator's own nodes run this same public copy |
| `neurahash/`, `neurahash_torch/`, `tools/diloco_contributor.py`, `tools/ipfs_checkpoint.py` | modules the expert-LoRA module imports by name | byte copies of this repository's public files |
| `sidecar/` | the sidecar's Go source + `build_sidecar.py` | the source of the fleet's sidecar |
| `tests/` | import gate, manifest + secrets scan, launcher tests | new |

## Verify this folder yourself (CPU only, a few minutes)

```powershell
python -m pytest glm-node\tests -q                                        # import gate, manifest, secrets, launcher
python glm-node\lib\runner\glm47_h1x_chain.py --selftest                 # SELFTEST PASS ... checks
python glm-node\lib\runner\diloco_golden_check.py                        # "all": "cb5d16ca18642b5a..."
python glm-node\build_packs.py --verify-only                             # after step 2
```

The golden check's `all` digest must equal the fleet's, `cb5d16ca18642b5a3f9dd4873d9288236a4be26e7a08f837691843c4b0fbdb77`,
on any CPU — it is how the operator knows your machine computes the shared average exactly like everyone else's.
