"""
testnet_node — PUBLIC MINER TRIM.

The full node's testnet_node module drives a coordinator/deploy/economics harness (model, corpus,
ledger, mining pool, block reward, deploy). NONE of that ships in the public miner. This trimmed
copy exposes ONLY the shared pre-shared-key config the worker path reads:

    PSK, PSK_IS_DEFAULT

resolved from `neurahash.net_transport.resolve_psk()`:
  * NEURAHASH_PSK (a SECRET) for real deployments -> (that key, PSK_IS_DEFAULT=False)
  * otherwise the built-in DEMO key b"neurahash-demo-psk" -> (demo key, PSK_IS_DEFAULT=True)

The demo key is PUBLIC (committed here on purpose); it authenticates NOTHING against anyone who can
read the source and exists only so local/loopback dev needs no config. A real deployment MUST set a
secret --psk / NEURAHASH_PSK out of band on BOTH the miner and the coordinator.
"""
from neurahash.net_transport import resolve_psk

PSK, PSK_IS_DEFAULT = resolve_psk()   # NEURAHASH_PSK (secret) for real deployments; demo key is the
#                                       loopback/dev default (public in the repo - authenticates nothing
#                                       off-loopback). A real deployment sets a secret PSK out of band.
