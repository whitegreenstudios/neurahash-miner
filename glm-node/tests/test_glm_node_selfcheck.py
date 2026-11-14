"""node_selfcheck.py: the Windows commit-limit check (a node commits ~18.5 GB of RAM + pagefile, so a 16 GB PC fits
only with a system-managed pagefile). CPU only, no network.

Run: python -m pytest glm-node/tests/test_glm_node_selfcheck.py -q
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import node_selfcheck as SC  # noqa: E402


def test_a_16_gb_pc_without_a_pagefile_warns_and_names_the_fix():
    status, detail = SC.commit_verdict(16.0e9, 10.0e9)            # commit limit = RAM only
    assert status == "WARN" and "System managed" in detail and "18.5" in detail, detail


def test_a_16_gb_pc_with_a_system_managed_pagefile_passes():
    status, detail = SC.commit_verdict(40.0e9, 30.0e9)
    assert status == "PASS", detail


def test_too_little_free_commit_right_now_warns():
    status, detail = SC.commit_verdict(40.0e9, 10.0e9)
    assert status == "WARN" and "free now" in detail, detail


def test_the_boundary_is_the_node_commit_plus_headroom():
    need = (SC.NEED["node_commit_gb"] + SC.NEED["commit_headroom_gb"]) * 1e9
    assert SC.commit_verdict(need - 1e8, need)[0] == "WARN"
    assert SC.commit_verdict(need + 1e8, need)[0] == "PASS"


def test_an_unreadable_limit_is_info_not_a_failure():
    assert SC.commit_verdict(None, None)[0] == "INFO"


def test_the_probe_reads_this_machine():
    limit, avail = SC.commit_bytes()
    if os.name != "nt":
        assert (limit, avail) == (None, None)
    else:
        assert limit and avail is not None and 0 < avail <= limit, (limit, avail)
