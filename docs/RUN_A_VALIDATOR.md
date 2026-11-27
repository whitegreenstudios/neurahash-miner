# Run a NeuraHash validator (stage S1, light mode)

Status: DRAFT, 2026-11-27. Stage S1 of the plan to remove the GLM lane's trusted coordinator
(docs/research/STEP4_ROTATING_PROPOSER_DESIGN_2026-11-26.md, sec. 5.3).

## What you are signing up for

The GLM training lane is still run by one coordinator. It decides which miner contributions are
accepted and publishes a signed, hash-chained record of every decision. A validator independently
re-checks each record and publishes one signed verdict per event: **agree**, **disagree** or
**abstain**.

Nothing you publish can change the lane today. The coordinator never reads your verdicts. They are
evidence: stage S1 passes only when at least one validator that does **not** run on the project
owner's PCs has published verdicts for at least 7 days, and no machine has recorded a disagreement
over 500 consecutive events.

There are no rewards in stage S1. Later stages add a stake on neura_l1 and payouts, and they will be
announced separately.

## What it costs you

Measured on the project's own light validator (an RTX 4060 PC, which uses none of its GPU):

| Resource | Light mode |
|---|---|
| GPU | none |
| RAM | about 90 MB (peak 88 MB measured) |
| CPU | negligible: one check every 2 minutes |
| Disk | under 100 MB per month (local log of your verdicts) |
| Network | a few kB per event, outbound HTTP only; no open ports |
| Uptime | as continuous as you can; gaps are fine, the runner catches up |

Any always-on machine works: a small Linux VPS, a Raspberry Pi 4, an old laptop.

## What light mode checks

For every coordinator event it checks, with no model and no GPU:

1. **Signature and chain.** The record is signed by the pinned coordinator address
   `0x2165534Ae1C161B6A8523f37F39cE4cf45bF7B86`, and links to the previous record's hash.
2. **Signed gate numbers.** The coordinator's signed decision (z, threshold, gain, margin, losses)
   is internally consistent with what it decided.
3. **Model-root link.** A rejected contribution must leave the model unchanged; an accepted one
   must change it.
4. **Rewrites and omissions.** No published record was later rewritten or removed, and signed
   contributions are not silently skipped.

It does **not** re-score contributions (that needs the 30B model on a large machine), so its
verdict on the accept/reject decision itself is always "abstain (light)". A **disagree** on any of
the checks above is a real finding; please report it.

## Setup

### 1. Get the code

The light validator ships in the public miner repository (release 3.9.8 or later):

```
git clone https://github.com/whitegreenstudios/neurahash-miner
cd neurahash-miner
```

The public repository ships **light mode only**. `--mode full` re-scores contributions and needs the
project's private coordinator code; in a public clone it refuses with a message and exits.

### 2. Install Python and two packages

Python 3.10 or newer.

```
python3 -m venv venv
. venv/bin/activate            # Windows PowerShell: venv\Scripts\Activate.ps1
pip install numpy eth-account
```

No torch, no CUDA.

**Optional: use uv instead.** [uv](https://docs.astral.sh/uv/) installs packages much faster and can
download Python for you, so you do not need Python installed first. It does not make the validator
itself run faster. Install uv by following its official instructions
(https://docs.astral.sh/uv/getting-started/installation/), then:

```
uv venv --python 3.12          # creates .venv; downloads Python 3.12 if you do not have it
. .venv/bin/activate           # Windows PowerShell: .venv\Scripts\Activate.ps1
uv pip install numpy eth-account
```

With the environment activated, every later command in this guide works unchanged. In the systemd
unit below, use `.venv/bin/python` instead of `venv/bin/python`.

### 3. Create your validator key

Your key identifies you. It never leaves your machine and nobody from the project will ever ask
for it.

```
python3 -c "import secrets,json; print(json.dumps({'key': secrets.token_hex(32)}))" > validator_key.json
chmod 600 validator_key.json
```

Back it up somewhere safe. If you lose it, start a new one; your past verdicts stay valid under the
old address.

### 4. Run it

```
python3 tools/glm_s1_attester.py --mode light \
    --campaign <CAMPAIGN> \
    --key-file validator_key.json \
    --attest-dir ./validator_out \
    --follow --poll-s 120
```

`<CAMPAIGN>` is the current 16-character campaign id, published with each release (today:
`508ed8c1ec9a2f9a`). The first line of `validator_out/*.log` prints your validator address. Send that
address to the project so your verdicts can be counted.

To keep it running across reboots, use your system's service manager (systemd on Linux, Task
Scheduler on Windows). A minimal systemd unit:

```
[Unit]
Description=NeuraHash S1 light validator
After=network-online.target

[Service]
WorkingDirectory=/home/<you>/neurahash-miner
ExecStart=/home/<you>/neurahash-miner/venv/bin/python tools/glm_s1_attester.py --mode light --campaign <CAMPAIGN> --key-file validator_key.json --attest-dir validator_out --follow --poll-s 120
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```

## Checking it works

- `validator_out/attest_*.status.json` updates every poll and shows the last event checked.
- Each event appends one line to `validator_out/attest_*.jsonl`; the same bytes are published to
  the relay under `sharddiloco/glm/attest/<campaign>/<event>/<your address>`.
- Expect `abstain (light)` for every event. Any `disagree` line: please send that line.
- A failed publish is queued and retried; it never stops the runner.

## Privacy and safety

- Outbound HTTP only, to the project relay. No inbound ports.
- What is published: your validator address and your verdicts. Nothing about your machine.
- The runner writes only inside `--attest-dir`, and deletes only its own relay entries older than
  500 events.

## Stopping

Stop the process. Nothing else depends on it.
