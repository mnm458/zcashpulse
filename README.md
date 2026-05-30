# ZcashPulse

Network telemetry and block propagation analysis dashboard for [Zebra](https://github.com/ZcashFoundation/zebra).

## Overview

ZcashPulse queries Prometheus for Zebra node metrics and computes block propagation
time estimates using an Erdos-Renyi network model. It provides real-time visibility
into peer connectivity, block verification performance, chain fork activity, and
Orchard/Halo2 proof verification.

## Requirements

- Python 3.12+
- A running [Zebra](https://github.com/ZcashFoundation/zebra) node with metrics enabled (`metrics.endpoint_addr`)
- Prometheus scraping the Zebra metrics endpoint

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

The dashboard is served at `http://localhost:8080`.

By default, ZcashPulse expects Prometheus at `http://localhost:9090`. Set the
`PROMETHEUS_URL` environment variable or edit `app.py` to change this.

## Propagation Model

The `/api/propagation-metrics` endpoint estimates T50 and T90 block propagation
times across the Zcash network:

1. **Link latency** -- derived from peer RTT measurements (`zcash.net.peer.rtt.seconds`),
   halved for one-way latency and scaled by 1.5x for INV/GETDATA/BLOCK relay overhead.
2. **Processing delay** -- total download + verification time per block
   (`zcash.block.processing.delay.seconds`).
3. **Hop distribution** -- modeled as a geometric distribution over an Erdos-Renyi
   random graph with connection probability `p = deg / (N - 1)`, where `deg = 25`
   (Zebra's peerset target) and `N` is the observed peer count.
4. **Propagation CDF** -- a Gaussian mixture over hop counts, solved via bisection
   for the 50th and 90th percentile arrival times.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/metrics` | Block height, peer count, stale rate, active forks |
| `GET /api/block-height-history` | Block height over 1 hour |
| `GET /api/blocks-committed-history` | Block commit rate over 1 hour |
| `GET /api/proofs-verified-history` | Groth16 and Halo2 verification rates |
| `GET /api/reorgs-history` | Reorganization events over 7 days |
| `GET /api/propagation-metrics` | Full propagation model output |
| `GET /api/propagation-history` | RTT and processing delay percentiles over 1 hour |
| `GET /api/connected-peers` | Connected peer list with versions |
| `GET /api/peer-messages` | Per-command message counts |
| `GET /api/peer-versions` | Peer software version distribution |
| `GET /api/message-rates` | Inbound/outbound message rates over 1 hour |
| `GET /api/orchard-metrics` | Halo2 proof verification statistics |
| `GET /api/orchard-history` | Orchard verification performance over 1 hour |

## Zebra Metrics Required

ZcashPulse depends on metrics from a Zebra node running with the
[comprehensive metrics](https://github.com/AustinZcash/zebra/tree/feat/comprehensive_metrics)
instrumentation branch:

- `zcash.net.peer.rtt.seconds` -- peer round-trip time
- `zcash.block.processing.delay.seconds` -- block download + verification time
- `zcash.block.verify.delay.seconds` -- verification-only time
- `network.block.download.duration_ms` -- block download time
- `state.non_finalized.consensus.stale_blocks` -- stale block count
- `state.non_finalized.blocks.committed` -- committed block count
- `state.non_finalized.forks.active` -- active chain fork count
- `state.non_finalized.reorgs.count` -- reorganization count
- `consensus.proof.halo2.verification.duration_ms` -- Halo2 batch verification time

## License

MIT
