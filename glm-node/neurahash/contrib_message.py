"""neurahash/contrib_message.py -- the canonical contribution-signature message (LEAF; no merge/ledger deps).

Extracted VERBATIM from neurahash/diloco_merge.py so client modules (the miner's
tools/diloco_contributor.publish_delta) can build the GAP1 signed-contribution bytes without importing
the coordinator-side merge/accept policy (poll_contrib_records/apply_delta_gated/fetch_delta). Pure,
side-effect-free byte-builder with no first-party imports. Mirrors the neura_l1.canon / neurahash.identity
extraction pattern -- identical recipe, identical bytes out; the signer here and the coordinator's verifier
must build the message through THIS one function so their signatures can never drift.
"""

# GAP1 (docs/GO_PUBLIC_DESIGN.md): wallet-signed contribution records. A public contribution today is
# unsigned free text and proves NOTHING about who produced it. We bind each delta to a wallet address
# the coordinator can verify by re-recovering the signer from a secp256k1 signature (never trusting a
# self-declared address). Bumping any signed field must bump this version tag -- an old signature can
# then never be reinterpreted against a new field layout.
CONTRIB_SIG_VERSION = "neurahash-diloco-contrib-v1"


def contrib_canonical_message(delta_cid, base_round, name, val_before, val_after):
    """The EXACT bytes a contribution record is signed over and verified against (GAP1). BOTH the signer
    (tools/diloco_contributor.publish_delta) and the verifier (poll_contrib_records) build the message
    through THIS one function, so they cannot drift out of sync -- a drift would silently break every
    signature. `name` is the contributor IDENTITY string (the record's own 'contributor' field, the
    value passed as `diloco --name`), NOT the registry-slot key. Fields are newline-joined after the
    version tag; the numeric/None fields go through str() so JSON round-tripping is lossless
    (json.dumps uses float.__repr__, which round-trips, and None <-> null <-> 'None')."""
    return (
        CONTRIB_SIG_VERSION + "\n"
        + str(delta_cid) + "\n"
        + str(base_round) + "\n"
        + str(name) + "\n"
        + str(val_before) + "\n"
        + str(val_after)
    ).encode("utf-8")
