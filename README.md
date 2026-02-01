# zcashpulse

A real-time network telemetry dashboard for monitoring Zebra node operations.

## Tech Stack

- **Backend:** Flask (Python 3.12+)
- **Frontend:** HTML/CSS/JavaScript with Chart.js
- **Data Source:** Prometheus

## Features

### Real-Time Metrics

- Block height, peer count, stale rate, active forks
- Auto-refreshes every 10 seconds

### Block Propagation Model

- Network latency analysis (RTT percentiles)
- Propagation time estimates (T50, T90)
- Statistical modeling using Gaussian mixture distributions

### Orchard Bundle Verification

- Halo2 proof verification metrics
- Batch durations and throughput stats

### Peer Communications

- Connected peers with IP, protocol version, user agent
- Inbound/outbound message rates
- Peer software version distribution

## Charts

- Block height history (1 hour)
- Blocks committed rate
- Proof verification rates (Groth16 vs Halo2)
- Reorganizations (7 days)
- Message rates
- RTT & processing delay

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py

Dashboard runs on http://localhost:8080

Requirements

- Python 3.12+
- Prometheus running on localhost:9090
- Zebra node with metrics enabled

API Endpoints
┌───────────────────────────┬───────────────────────────┐
│         Endpoint          │        Description        │
├───────────────────────────┼───────────────────────────┤
│ /api/metrics              │ Current node metrics      │
├───────────────────────────┼───────────────────────────┤
│ /api/block-height-history │ 1-hour block height       │
├───────────────────────────┼───────────────────────────┤
│ /api/propagation-metrics  │ Network propagation model │
├───────────────────────────┼───────────────────────────┤
│ /api/orchard-metrics      │ Halo2 verification stats  │
├───────────────────────────┼───────────────────────────┤
│ /api/connected-peers      │ Peer node list            │
├───────────────────────────┼───────────────────────────┤
│ /api/peer-messages        │ Message statistics        │
└───────────────────────────┴───────────────────────────┘
```
