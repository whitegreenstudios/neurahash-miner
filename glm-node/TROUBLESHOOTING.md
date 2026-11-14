# glm-node troubleshooting

Every row: **symptom** → what the node already does **by itself** → what **you** can do → what to **send** the
operator. The universal "send" is `python glm-node\glm_node.py logs --card run_card.json` (a zip of the node's logs,
`TRAIN.json`, transport logs and the self-check; no checkpoints, keys or tokens) plus the output of
`python glm-node\node_selfcheck.py`.

**Rejoin command** (used below): `python glm-node\glm_node.py run --card run_card.json --rejoin`
(add `--from-snapshot` if the node was away for more than ~30 rounds).
**Relay-only mode** (used below): add `--relay-only` to the `run` command; the node then never uses P2P. It runs
the node with `--p2p force-relay`. Never pass `--p2p off` yourself in a run whose card says `--p2p auto`: that node
would disagree with its peers at step 0, which voids the attempt for everyone.

## Before a run (self-check / preflight)

| symptom | automatic | you | send |
|---|---|---|---|
| `gpu ... FAIL` (no GPU / < 8 GB / driver < 570) | — | install / update the NVIDIA driver; an 8 GB card is the minimum | self-check output |
| `cuda-probe FAIL: torch ... cannot see the GPU` | — | torch is the CPU wheel: `python -m pip install --force-reinstall torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128` | self-check output |
| `CUDA_VISIBLE_DEVICES=-1 hides the GPU` | the self-check does not probe CUDA then (a probe under -1 crashes some NVIDIA drivers) | remove the variable (System Properties → Environment Variables, or `Remove-Item Env:CUDA_VISIBLE_DEVICES`) | — |
| `transformers ... has no glm4_moe_lite` or a `pkg:` version WARN/FAIL | — | use the venv and `pip install -r glm-node\requirements-node.txt`; the fleet's versions matter (they are not part of the run's own version check) | self-check output |
| `code FAIL: N files differ from MANIFEST.sha256` | the launcher refuses to start | re-clone; do not edit files; keep `glm-node/.gitattributes` (without it Git for Windows rewrites line endings) | the file names it lists |
| `runner sha256 ... != the run's` (preflight) | refused before start (otherwise the run would refuse this node as a version skew) | `git fetch --tags` and `git checkout <the tag the operator named>` | the two shas |
| `packs ... NOT verified` / `base-... FAIL` / `corpus ... FAIL` | refused before start | `python glm-node\build_packs.py` (resumes; `--verify-only` to re-hash), `python glm-node\fetch_corpus.py --rebuild` | the build log |
| `build_packs`: a download hangs at 0 bytes | Xet is disabled and IPv4 forced by default (both are measured fixes for home links) | Ctrl+C and re-run (downloads resume); try another time of day | the last 20 lines |
| `build_packs`: `sha256 ... != Hugging Face's` / `!= the fleet's` | the bad file is deleted; nothing is installed | re-run; if it repeats on the same file, your disk or RAM may be faulty | the full log |
| `disk FAIL` | — | free space (75 GB to build; once the packs are built, 10 GB free while running); use an NVMe SSD | — |
| `commit WARN: Windows commit limit ... < ... GB` (or `only ... GB ... free now`) | — | a node commits ~18.5 GB of RAM + pagefile; set the pagefile to **System managed size** (Settings > System > About > Advanced system settings > Performance > Advanced > Virtual memory > "Automatically manage paging file size"), reboot, and close big programs before a run; on a 16 GB PC this is required | self-check output |
| `disk-speed WARN/FAIL` (< 1.5 GB/s) | — | put `NEURAHASH_GLM_HOME` on your fastest NVMe drive; more free RAM caches packs | self-check output |
| `relay-http FAIL` | the launcher retries 3 times, then refuses to start | check your internet; corporate / school networks may block port 8710; try later | self-check output |
| `relay-admit WARN: ... operator_only` | — | the relay is not taking writes from miners right now: ask the operator before starting | — |
| `stun-udp FAIL` / `turn-udp FAIL` | P2P can still use TURN over TCP; the relay always works | allow outbound UDP 3478 in your firewall if you can (not required) | self-check output |
| `turn-tcp FAIL ... allocation refused: 401` | relay-only | your TURN credentials are wrong or expired: ask the operator for new ones | — (never send the password) |
| `sidecar FAIL` / "blocked by antivirus" | the launcher checks the sidecar before every start (and before every `--auto-rejoin` relaunch) and steps down to relay-only by itself, printing why (it dry-builds the transport with the runner's own code; the Stage-0 runner itself also falls back to relay-only under `--p2p auto`, so the node still joins and trains) | allow `glm-node\lib\runner\bin\p2psidecar.exe` in your antivirus, or rebuild it from source (`python glm-node\sidecar\build_sidecar.py`, needs Go), or use relay-only mode | self-check output |
| `clock WARN` (your clock is far from the relay's) | -- | turn on automatic time sync (Windows: Settings > Time & language > Sync now); P2P peers meet in 2-minute slots | -- |
| `golden FAIL` | -- | do not join: your CPU / numpy computes the shared average differently from the fleet's | the `golden` line of the self-check (the standalone `python glm-node\lib\runner\diloco_golden_check.py` also prints your PC's name: remove it before sending if you prefer) |
| `relay-write FAIL ... HTTP 401/403` (with `--relay-write-test`) | -- | your token is wrong, expired, or not allowed to write: ask the operator | the HTTP code (never the token) |
| `relay-write WARN` (slow uplink) | -- | a round part (~37 MB) must reach the relay within the round deadline; pause big uploads during a run | self-check output |
| `your public key ... is not the card's key for node X` | relay-only | send the operator the `PUBKEY` line from `glm_node.py keygen` and get an updated card | the PUBKEY line |
| `this node already ran in ... use --rejoin` | refused (starting fresh in a run you were part of would collide with your own earlier output) | use the rejoin command, or a new card if it is a new run | — |
| `profile ... requires --vram-cap X on node Y` | refused (the run would refuse it at start and void the attempt) | ask the operator: registered runs fix the cap per slot; trial runs use the unregistered `trial` (or `test`) profile, where the cap is free | — |
| `VRAM cap ... does not fit` (launcher) / the runner refuses `--vram-cap` above the card's total minus 1.25 GiB | refused before anything is published (your peers keep waiting for a relaunch) | use the card's cap (6.5 GiB on an 8 GB card); close GPU-heavy programs | self-check output |

## During a run

| symptom | automatic | you | send |
|---|---|---|---|
| P2P slow, failing or never connecting (`transport.jsonl`: `hedge`, `latch`; log: `!!!! P2P SIDECAR START FAILED`) | every fetch falls back to the relay (hedge race, per-peer latch, global latch with a re-probe every 6 fetches / 10 min); the node keeps training | nothing; or restart with `--relay-only` and the rejoin command if you prefer | logs zip |
| relay unreachable mid-run (log: `transport failure`, `xport PUT ... FAILED`) | chunk retries (3 per chunk), then the whole part again until the round commits without it; the node waits; `!!!! ALARM` after ~3 h; resumable stop (exit 4) after ~4 h | check your internet; when it is back, the node continues by itself (if it stopped: the rejoin command) | logs zip |
| your home IP changed / Wi-Fi dropped / router rebooted | every relay request opens a fresh connection; P2P peers drop and re-dial in the next 2-minute slot | nothing | — |
| `LATE:` lines (this node missed a round) | the round commits without this node; it re-enters the next round | keep the PC on and busy only with the node; tell the operator if it repeats | logs zip |
| very slow steps | — | close GPU-heavy programs (games, video, browsers with hardware acceleration); keep the packs on NVMe | `status` output |
| CUDA out of memory | the node stops (the VRAM cap turns a spill into a clean error instead of a frozen desktop) | close other GPU programs, then the rejoin command | logs zip |
| `The paging file is too small for this operation to complete` (os error 1455) | the node stops (host memory, not VRAM) | set the pagefile to System managed size (see `commit WARN` above), reboot, then the rejoin command | logs zip + self-check output |
| PC went to sleep | the launcher keeps Windows awake while the node runs (a node asleep at the start of training can stop the whole run) | laptops: plug in and set "never sleep"; if it slept anyway: the rejoin command | logs zip |
| disk full | the node crashes (it has no free-space guard of its own) | free >= 10 GB on the drive of `NEURAHASH_GLM_HOME`, then the rejoin command | logs zip |
| `==== VOID` (exit 3) at start: `init_meta ... differs from the peer's` | the node stops before training | wrong version or card: check the tag and the card with the operator | logs zip |
| `==== ABORT` (exit 5): `G3` or integrity failure | nothing unverified is ever applied; the node stops | the rejoin command; if it repeats, your disk or RAM may be corrupting data | logs zip + self-check with `--disk-test` |
| exit 4 (resumable stop or pause) | the node kept its last round checkpoint | wait for the operator's "go", then the rejoin command | — |
| exit 4 with an eviction reason (`trial` profile: this node's part was refused, or it tripped its own G1 speed bound / loss check) | the run goes on without this node; it kept its round checkpoint | tell the operator the reason line; when they say so, the rejoin command (a slow node: close other programs first) | logs zip |
| exit 7: `relay refused this node's key: <status> -- contact the operator` | after 3 refused uploads (401 / 403) of its own, or 15 min of quota refusals (429 / 507), the node publishes its leave marker if it can and stops; the run goes on without it; `--auto-rejoin` does NOT relaunch it (the same key would be refused again) | ask the operator for a new key, set it in `NEURAHASH_CONTENT_TOKEN`, then the rejoin command | the status code (never the token) |
| the window was closed / the PC rebooted | — | the rejoin command (same `NEURAHASH_GLM_HOME`) | — |

## Stopping

Ctrl+C once: the node leaves at the next round boundary, after its round checkpoint is written. Ctrl+C a second
time: it leaves at once. Either way the launcher ends the node's processes — for the run a clean leave (the run goes
on without you, nothing shared is touched) and you can rejoin later. Avoid it during the run's first round (the
launcher warns): a node that drops out there can void the attempt for everyone. A run also tolerates only a few
missed node-rounds in total, so tell the operator before planned stops. For unattended nodes,
`run ... --auto-rejoin 3` re-enters by itself after a crash (not after a stop, a VOID, a pause or a refused key).

## Network notes

* Everything is **outbound**: no port forwarding, no inbound firewall rule. Needed: TCP 8710 to the relay, TCP 3478
  to the TURN server (UDP 3478 optional), HTTPS to huggingface.co for the one-time downloads.
* Behind a VPN the node works, but P2P uses the VPN's exit address; some VPNs block UDP.
* A home IP change is harmless (see above). Two nodes behind the same home router never use a LAN path: training
  traffic is WAN-only by design.

## Key log lines (in `runs\<run>\<letter>\trainer_arm_<letter>.log`)

`GLM47-H1X TRAINER |` start · `DILOCO EXCHANGE` the agreed run · `G2` init check · `round NNN |` a committed
round · `LATE:` missed a round · `REJOIN from` / `REJOINED` · `P2P SIDECAR up:` / `!!!! P2P SIDECAR START FAILED` ·
`XPORT` chunked transport · `transport failure` / `G3 miss` · `!!!! ALARM` · `REQUESTED STOP` ·
`==== VOID` / `==== ABORT` / `==== TERMINAL:` the end. Transport rows: `xport.jsonl`, `transport.jsonl`,
`p2p\sidecar_events.jsonl`.
