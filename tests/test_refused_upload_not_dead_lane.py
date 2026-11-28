"""A miner whose OWN uploads are refused must not conclude the lane is dead and stop retrying.

THE INCIDENT (2026-11-27/28). The relay's pool of signed names filled up, so every miner upload got
HTTP 507 "name index full" from ~16:45 UTC. With nothing arriving, the coordinator's event counter
stopped; 3 h later every miner's stale-lane detector fired and PAUSED -- and a paused miner skips the
publish path, so it stopped retrying the very upload that would have moved the lane. The lane could
not restart by itself even after the relay was fixed: both miners had to be restarted by hand.

THE RULE. A store that answers with a refusal is alive, so "my latest upload stands refused" is
evidence AGAINST a dead lane. stale_action() returns "refused" then, and the loop keeps its normal
train / keep-best-round / retry-under-backoff path, printing a banner that names the refusal instead
of "THIS LANE LOOKS DEAD". A genuinely dead lane (no refusal pending) still pauses exactly as before.

Torch-free, no network, no GPU.
Run: C:/Python313/python.exe -m pytest tests/test_refused_upload_not_dead_lane.py -q
"""
import io
import os
import re
import sys
import urllib.error

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOLS = os.path.join(_REPO, "tools")
for _p in (_HERE, _REPO, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sharddiloco_glm_contributor as N                     # noqa: E402

FULL = b'{"ok": false, "error": "name index full: cg/508ed8c1ec9a2f9a/r724/glm-7872f61C.1"}'


def _507():
    return urllib.error.HTTPError("http://relay/n/x", 507, "full", {}, io.BytesIO(FULL))


def _stale_lane(threshold=100.0):
    lh = N.LaneLiveness(threshold_s=threshold, now=0.0)
    assert lh.observe(0.0, event=724, accepted_seen=10) == "waiting"     # first sight latches, no move
    assert lh.observe(150.0, event=724, accepted_seen=10) == "stale"
    return lh


def test_stale_action_truth_table():
    assert N.stale_action("live", False) == "run"
    assert N.stale_action("waiting", True) == "run"
    assert N.stale_action("recovered", True) == "run"
    assert N.stale_action("stale", False) == "pause"          # a dead lane still pauses, unchanged
    assert N.stale_action("stale", True) == "refused"         # our refused upload: keep retrying


def test_backoff_reports_refusing_until_a_publish_succeeds():
    b = N.PublishBackoff(base_s=1.0, cap_s=8.0)
    assert not b.refusing and b.last_reason is None
    b.failed(100.0, _507(), "m", reason=N.publish_failure_reason(_507()))
    assert b.refusing
    assert "507" in str(b.last_reason) or "full" in str(b.last_reason)
    # still refusing while it waits out the backoff (no attempt made): the lane must not pause meanwhile
    assert b.blocked(100.5) > 0 and b.refusing
    b.succeeded()
    assert not b.refusing and b.last_reason is None


def test_a_507_is_temporary_and_never_exits_the_miner():
    b = N.PublishBackoff(base_s=1.0, cap_s=8.0, permanent_max=3)
    for t in range(50):                                       # far past permanent_max
        b.failed(float(t), _507(), "m")
    assert b.refusing and b.failures == 50


def test_the_incident_sequence_now_recovers_without_a_restart():
    """Replay: refusals -> lane goes stale -> (old: pause forever) new: keep retrying -> relay fixed ->
    publish succeeds -> coordinator moves -> 'recovered'."""
    lh = _stale_lane()
    b = N.PublishBackoff(base_s=1.0, cap_s=8.0)
    b.failed(120.0, _507(), "m")
    st = lh.observe(160.0, event=724, accepted_seen=10)
    assert N.stale_action(st, b.refusing) == "refused"        # was "pause" before the fix
    b.succeeded()                                             # the relay accepts the retried round
    assert N.stale_action(lh.observe(170.0, event=724, accepted_seen=10), b.refusing) == "pause"
    assert lh.observe(900.0, event=725, accepted_seen=11) == "recovered"


def test_dead_lane_without_refusals_still_pauses_and_shouts_the_old_banner():
    lh = _stale_lane()
    b = N.PublishBackoff()
    assert N.stale_action(lh.observe(200.0, event=724), b.refusing) == "pause"
    assert any("THIS LANE LOOKS DEAD" in l for l in lh.stale_banner(200.0))


def test_refused_banner_is_ascii_names_the_refusal_and_never_says_dead():
    lh = _stale_lane()
    lines = lh.refused_banner(200.0, 7, "HTTP 507: name index full \u2013 x")
    text = "\n".join(lines)
    assert all(ord(c) < 128 for c in text)                    # cp1252 console safe
    assert "REFUSING YOUR UPLOADS" in text and "name index full" in text and "7 refusal" in text
    assert "LOOKS DEAD" not in text


def test_the_loop_uses_stale_action_and_does_not_skip_publishing_on_refusal():
    """Source pin: the async loop decides through stale_action, and only the 'pause' branch continues."""
    src = open(os.path.join(_TOOLS, "sharddiloco_glm_contributor.py"), encoding="utf-8").read()
    i = src.index("_stale_do = stale_action(_lane_state, _pub_backoff.refusing)")
    block = src[i:i + 1500]
    refused = block[block.index('if _stale_do == "refused":'):block.index('elif _stale_do == "pause":')]
    paused = block[block.index('elif _stale_do == "pause":'):]
    assert "continue" not in refused
    assert re.search(r"\n\s+continue\b", paused)
    assert 'if _lane_state == "stale":' not in src            # the old unconditional pause is gone
