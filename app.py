#!/usr/bin/env python3

from flask import Flask, jsonify, render_template
import requests
from datetime import datetime

app = Flask(__name__)

PROMETHEUS_URL = "http://localhost:9090"


def query_prometheus(query):
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=5
        )
        data = r.json()
        if data["status"] == "success" and data["data"]["result"]:
            return data["data"]["result"][0]["value"][1]
        return None
    except Exception as e:
        print(f"Prometheus query error: {e}")
        return None


def query_prometheus_range(query, start, end, step="60s"):
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={
                "query": query,
                "start": start,
                "end": end,
                "step": step
            },
            timeout=10
        )
        data = r.json()
        if data["status"] == "success" and data["data"]["result"]:
            return data["data"]["result"][0]["values"]
        return []
    except Exception as e:
        print(f"Prometheus range query error: {e}")
        return []


def query_prometheus_multi(query):
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=10
        )
        data = r.json()
        if data["status"] == "success":
            return data["data"]["result"]
        return []
    except Exception as e:
        print(f"Prometheus multi query error: {e}")
        return []


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/metrics")
def get_metrics():
    """Get current node metrics"""
    block_height = query_prometheus("zcash_chain_verified_block_height")
    peer_count = query_prometheus("zcash_net_peers")

    # Calculate stale rate: stale_blocks / blocks_committed * 100
    # stale_blocks metric may not exist if no blocks have gone stale yet - default to 0
    stale_blocks = query_prometheus("state_non_finalized_consensus_stale_blocks")
    blocks_committed = query_prometheus("state_non_finalized_blocks_committed")

    stale_rate = None
    if blocks_committed is not None:
        stale_blocks_val = float(stale_blocks) if stale_blocks is not None else 0.0
        blocks_committed_val = float(blocks_committed)
        if blocks_committed_val > 0:
            stale_rate = round((stale_blocks_val / blocks_committed_val) * 100, 3)

    active_forks = query_prometheus("state_non_finalized_forks_active")

    return jsonify({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "block_height": int(block_height) if block_height else None,
        "peer_count": int(float(peer_count)) if peer_count else None,
        "stale_rate": stale_rate,
        "active_forks": int(float(active_forks)) if active_forks else None,
    })


@app.route("/api/block-height-history")
def block_height_history():
    import time
    end = time.time()
    start = end - 3600

    values = query_prometheus_range(
        "zcash_chain_verified_block_height",
        start, end, "60s"
    )

    data = [{"x": int(v[0]) * 1000, "y": int(v[1])} for v in values]
    return jsonify(data)


@app.route("/api/blocks-committed-history")
def blocks_committed_history():
    import time
    end = time.time()
    start = end - 3600

    values = query_prometheus_range(
        "rate(state_non_finalized_blocks_committed[5m]) * 300",
        start, end, "60s"
    )

    data = [{"x": int(v[0]) * 1000, "y": round(float(v[1]), 2)} for v in values]
    return jsonify(data)


@app.route("/api/proofs-verified-history")
def proofs_verified_history():
    import time
    end = time.time()
    start = end - 3600 

    groth16_values = query_prometheus_range(
        "rate(proofs_groth16_verified[5m]) * 300",
        start, end, "60s"
    )
    halo2_values = query_prometheus_range(
        "rate(proofs_halo2_verified[5m]) * 300",
        start, end, "60s"
    )

    groth16_data = [{"x": int(v[0]) * 1000, "y": round(float(v[1]), 2)} for v in groth16_values]
    halo2_data = [{"x": int(v[0]) * 1000, "y": round(float(v[1]), 2)} for v in halo2_values]

    return jsonify({
        "groth16": groth16_data,
        "halo2": halo2_data
    })


@app.route("/api/reorgs-history")
def reorgs_history():
    import time
    end = time.time()
    start = end - (7 * 24 * 3600)

    values = query_prometheus_range(
        "increase(state_non_finalized_reorgs_count[1h])",
        start, end, "3600"
    )

    data = [{"x": int(v[0]) * 1000, "y": round(float(v[1]))} for v in values]
    return jsonify(data)


@app.route("/api/connected-peers")
def connected_peers():
    results = query_prometheus_multi("zcash_net_peers_connected")

    peers = []
    for r in results:
        labels = r.get("metric", {})
        count = int(float(r.get("value", [0, 0])[1]))
        if count > 0:
            peers.append({
                "ip": labels.get("remote_ip", "unknown"),
                "version": labels.get("remote_version", "unknown"),
                "negotiated_version": labels.get("negotiated_version", "unknown"),
                "user_agent": labels.get("user_agent", "unknown"),
                "connections": count
            })

    peers.sort(key=lambda x: x["connections"], reverse=True)
    return jsonify(peers)


@app.route("/api/peer-messages")
def peer_messages():
    inbound_results = query_prometheus_multi("sum by (command) (zebra_net_in_requests)")
    inbound = {}
    for r in inbound_results:
        cmd = r.get("metric", {}).get("command", "unknown")
        val = int(float(r.get("value", [0, 0])[1]))
        inbound[cmd] = val

    outbound_results = query_prometheus_multi("sum by (command) (zebra_net_out_requests)")
    outbound = {}
    for r in outbound_results:
        cmd = r.get("metric", {}).get("command", "unknown")
        val = int(float(r.get("value", [0, 0])[1]))
        outbound[cmd] = val

    response_results = query_prometheus_multi("sum by (command) (zebra_net_in_responses)")
    responses = {}
    for r in response_results:
        cmd = r.get("metric", {}).get("command", "unknown")
        val = int(float(r.get("value", [0, 0])[1]))
        responses[cmd] = val

    return jsonify({
        "inbound_requests": inbound,
        "outbound_requests": outbound,
        "responses": responses
    })


@app.route("/api/peer-versions")
def peer_versions():
    results = query_prometheus_multi("zcash_net_peers_connected")

    versions = {}
    for r in results:
        labels = r.get("metric", {})
        user_agent = labels.get("user_agent", "unknown")
        count = int(float(r.get("value", [0, 0])[1]))
        if count > 0:
            if user_agent in versions:
                versions[user_agent] += count
            else:
                versions[user_agent] = count

    version_list = [{"user_agent": k, "count": v} for k, v in versions.items()]
    version_list.sort(key=lambda x: x["count"], reverse=True)
    return jsonify(version_list)


@app.route("/api/message-rates")
def message_rates():
    """Get message rates over the last hour"""
    import time
    end = time.time()
    start = end - 3600

    inbound_values = query_prometheus_range(
        "sum(rate(zebra_net_in_requests[5m])) * 60",
        start, end, "60s"
    )

    outbound_values = query_prometheus_range(
        "sum(rate(zebra_net_out_requests[5m])) * 60",
        start, end, "60s"
    )

    response_values = query_prometheus_range(
        "sum(rate(zebra_net_in_responses[5m])) * 60",
        start, end, "60s"
    )

    inbound_data = [{"x": int(v[0]) * 1000, "y": round(float(v[1]), 2)} for v in inbound_values]
    outbound_data = [{"x": int(v[0]) * 1000, "y": round(float(v[1]), 2)} for v in outbound_values]
    response_data = [{"x": int(v[0]) * 1000, "y": round(float(v[1]), 2)} for v in response_values]

    return jsonify({
        "inbound": inbound_data,
        "outbound": outbound_data,
        "responses": response_data
    })


@app.route("/api/propagation-metrics")
def propagation_metrics():
    """
    Get all metrics needed for the block propagation model.

    Returns raw metrics and computed model parameters:
    - N: number of peers (network size estimate)
    - deg: average degree (neighbors per node)
    - RTT statistics for link latency (μ_λ, σ_λ)
    - Block processing delay statistics (μ_d, σ_d)
    - Computed propagation estimates (T50, T90)
    """
    import math

    # Network structure parameters
    peer_count = query_prometheus("zcash_net_peers")
    N = int(float(peer_count)) if peer_count else None
    deg = 25  # peerset_initial_target_size from config

    # RTT metrics (for link latency λ)
    rtt_sum = query_prometheus("zcash_net_peer_rtt_seconds_sum")
    rtt_count = query_prometheus("zcash_net_peer_rtt_seconds_count")
    rtt_p50 = query_prometheus('zcash_net_peer_rtt_seconds{quantile="0.5"}')
    rtt_p90 = query_prometheus('zcash_net_peer_rtt_seconds{quantile="0.9"}')
    rtt_p0 = query_prometheus('zcash_net_peer_rtt_seconds{quantile="0"}')
    rtt_p100 = query_prometheus('zcash_net_peer_rtt_seconds{quantile="1"}')

    # Calculate RTT mean
    rtt_mean = None
    if rtt_sum and rtt_count:
        rtt_sum_f = float(rtt_sum)
        rtt_count_f = float(rtt_count)
        if rtt_count_f > 0:
            rtt_mean = rtt_sum_f / rtt_count_f

    # Estimate RTT stddev from quantiles (approximation using IQR method)
    rtt_stddev = None
    if rtt_p90 and rtt_p50:
        # Approximate stddev: (p90 - p50) / 1.28 for normal distribution
        rtt_stddev = (float(rtt_p90) - float(rtt_p50)) / 1.28

    # Block download duration (ms)
    download_sum = query_prometheus("network_block_download_duration_ms_sum")
    download_count = query_prometheus("network_block_download_duration_ms_count")
    download_p50 = query_prometheus('network_block_download_duration_ms{quantile="0.5"}')
    download_p90 = query_prometheus('network_block_download_duration_ms{quantile="0.9"}')

    download_mean = None
    if download_sum and download_count:
        download_sum_f = float(download_sum)
        download_count_f = float(download_count)
        if download_count_f > 0:
            download_mean = download_sum_f / download_count_f / 1000  # Convert to seconds

    # Block verification delay (seconds)
    verify_sum = query_prometheus("zcash_block_verify_delay_seconds_sum")
    verify_count = query_prometheus("zcash_block_verify_delay_seconds_count")
    verify_p50 = query_prometheus('zcash_block_verify_delay_seconds{quantile="0.5"}')
    verify_p90 = query_prometheus('zcash_block_verify_delay_seconds{quantile="0.9"}')

    verify_mean = None
    if verify_sum and verify_count:
        verify_sum_f = float(verify_sum)
        verify_count_f = float(verify_count)
        if verify_count_f > 0:
            verify_mean = verify_sum_f / verify_count_f

    # Total block processing delay (seconds) - this is μ_d
    processing_sum = query_prometheus("zcash_block_processing_delay_seconds_sum")
    processing_count = query_prometheus("zcash_block_processing_delay_seconds_count")
    processing_p50 = query_prometheus('zcash_block_processing_delay_seconds{quantile="0.5"}')
    processing_p90 = query_prometheus('zcash_block_processing_delay_seconds{quantile="0.9"}')
    processing_p0 = query_prometheus('zcash_block_processing_delay_seconds{quantile="0"}')
    processing_p100 = query_prometheus('zcash_block_processing_delay_seconds{quantile="1"}')

    processing_mean = None
    processing_stddev = None
    if processing_sum and processing_count:
        processing_sum_f = float(processing_sum)
        processing_count_f = float(processing_count)
        if processing_count_f > 0:
            processing_mean = processing_sum_f / processing_count_f

    if processing_p90 is not None and processing_p50 is not None:
        p90_val = float(processing_p90)
        p50_val = float(processing_p50)
        if p90_val > p50_val:
            processing_stddev = (p90_val - p50_val) / 1.28
        elif processing_mean:
            # Fallback: estimate stddev as ~50% of mean (common heuristic)
            processing_stddev = processing_mean * 0.5

    # === Compute Propagation Model Parameters ===

    # Link latency (one-way): λ = RTT / 2, adjusted for INV→GETDATA→BLOCK relay: λ_adj = 1.5 * λ
    mu_lambda = None  # Mean link latency
    sigma_lambda = None  # Stddev link latency
    if rtt_mean:
        mu_lambda = (rtt_mean / 2) * 1.5  # One-way latency with relay adjustment
    if rtt_stddev:
        sigma_lambda = (rtt_stddev / 2) * 1.5

    # Processing delay
    mu_d = processing_mean  # Mean processing delay
    sigma_d = processing_stddev  # Stddev processing delay

    # Per-hop delay: X = λ + d
    mu_X = None
    sigma_X = None
    if mu_lambda is not None and mu_d is not None:
        mu_X = mu_lambda + mu_d
    if sigma_lambda is not None:
        # Variance of sum = sum of variances (assuming independence)
        sigma_d_val = sigma_d if sigma_d is not None else 0
        sigma_X = math.sqrt(sigma_lambda**2 + sigma_d_val**2)

    # Hop distance distribution (Erdős–Rényi approximation)
    # p = deg / (N - 1), P(H=h) = (1-p)^(h-1) * p
    hop_weights = None
    expected_hops = None
    if N and N > 1 and deg:
        p = deg / (N - 1)
        if p > 0 and p < 1:
            w1 = p
            w2 = (1 - p) * p
            w3 = (1 - p)**2 * p
            Z = w1 + w2 + w3
            hop_weights = {
                "h1": w1 / Z,
                "h2": w2 / Z,
                "h3": w3 / Z
            }
            expected_hops = 1/p if p > 0 else None

    T50 = None  # Median propagation time
    T90 = None  # 90th percentile propagation time
    E_delta = None  # Expected propagation time

    if mu_X is not None and hop_weights:
        # E[Δ] = Σ w_h * h * μ_X
        E_delta = mu_X * (hop_weights["h1"] * 1 + hop_weights["h2"] * 2 + hop_weights["h3"] * 3)

        # For T50 and T90, compute proper Gaussian mixture CDF
        # F(t) = Σ P(H=h) * Φ((t - h*μ_X) / sqrt(h)*σ_X)
        if sigma_X is not None and sigma_X > 0:
            from scipy.stats import norm

            def mixture_cdf(t):
                """Compute CDF of the Gaussian mixture at time t"""
                cdf = 0
                for h, w in [(1, hop_weights["h1"]), (2, hop_weights["h2"]), (3, hop_weights["h3"])]:
                    mu_h = h * mu_X
                    sigma_h = math.sqrt(h) * sigma_X
                    cdf += w * norm.cdf(t, loc=mu_h, scale=sigma_h)
                return cdf

            # Binary search for T50 (where CDF = 0.5)
            def find_quantile(target_prob, lo=0, hi=30):
                """Find t where mixture_cdf(t) = target_prob using bisection"""
                for _ in range(50):  # 50 iterations gives ~15 decimal places
                    mid = (lo + hi) / 2
                    if mixture_cdf(mid) < target_prob:
                        lo = mid
                    else:
                        hi = mid
                return mid

            T50 = find_quantile(0.5)
            T90 = find_quantile(0.9)

    return jsonify({
        "timestamp": datetime.utcnow().isoformat() + "Z",

        # Network structure
        "network": {
            "N": N,
            "deg": deg,
            "expected_hops": round(expected_hops, 2) if expected_hops else None,
            "hop_weights": {k: round(v, 4) for k, v in hop_weights.items()} if hop_weights else None
        },

        # RTT / Link latency raw metrics
        "rtt": {
            "samples": int(float(rtt_count)) if rtt_count else 0,
            "mean_ms": round(rtt_mean * 1000, 2) if rtt_mean else None,
            "p50_ms": round(float(rtt_p50) * 1000, 2) if rtt_p50 else None,
            "p90_ms": round(float(rtt_p90) * 1000, 2) if rtt_p90 else None,
            "min_ms": round(float(rtt_p0) * 1000, 2) if rtt_p0 else None,
            "max_ms": round(float(rtt_p100) * 1000, 2) if rtt_p100 else None,
            "stddev_ms": round(rtt_stddev * 1000, 2) if rtt_stddev else None
        },

        # Block download metrics
        "download": {
            "samples": int(float(download_count)) if download_count else 0,
            "mean_ms": round(download_mean * 1000, 2) if download_mean else None,
            "p50_ms": round(float(download_p50), 2) if download_p50 else None,
            "p90_ms": round(float(download_p90), 2) if download_p90 else None
        },

        # Block verification metrics
        "verify": {
            "samples": int(float(verify_count)) if verify_count else 0,
            "mean_ms": round(verify_mean * 1000, 2) if verify_mean else None,
            "p50_ms": round(float(verify_p50) * 1000, 2) if verify_p50 else None,
            "p90_ms": round(float(verify_p90) * 1000, 2) if verify_p90 else None
        },

        # Total processing delay metrics (d = download + verify)
        "processing": {
            "samples": int(float(processing_count)) if processing_count else 0,
            "mean_ms": round(processing_mean * 1000, 2) if processing_mean else None,
            "p50_ms": round(float(processing_p50) * 1000, 2) if processing_p50 else None,
            "p90_ms": round(float(processing_p90) * 1000, 2) if processing_p90 else None,
            "min_ms": round(float(processing_p0) * 1000, 2) if processing_p0 else None,
            "max_ms": round(float(processing_p100) * 1000, 2) if processing_p100 else None,
            "stddev_ms": round(processing_stddev * 1000, 2) if processing_stddev else None
        },

        # Computed model parameters
        "model_params": {
            "mu_lambda_ms": round(mu_lambda * 1000, 2) if mu_lambda else None,
            "sigma_lambda_ms": round(sigma_lambda * 1000, 2) if sigma_lambda else None,
            "mu_d_ms": round(mu_d * 1000, 2) if mu_d else None,
            "sigma_d_ms": round(sigma_d * 1000, 2) if sigma_d else None,
            "mu_X_ms": round(mu_X * 1000, 2) if mu_X else None,
            "sigma_X_ms": round(sigma_X * 1000, 2) if sigma_X else None
        },

        # Propagation estimates
        "propagation": {
            "T50_ms": round(T50 * 1000, 2) if T50 else None,
            "T90_ms": round(T90 * 1000, 2) if T90 else None,
            "expected_ms": round(E_delta * 1000, 2) if E_delta else None
        }
    })


@app.route("/api/orchard-metrics")
def orchard_metrics():
    """Get Orchard bundle verification metrics"""
    orchard_nullifiers = query_prometheus("state_finalized_cumulative_orchard_nullifiers")

    halo2_last_duration = query_prometheus(
        "consensus_proof_halo2_verification_last_duration_ms"
    )

    halo2_duration_sum = query_prometheus(
        "consensus_proof_halo2_verification_duration_ms_sum"
    )
    halo2_duration_count = query_prometheus(
        "consensus_proof_halo2_verification_duration_ms_count"
    )

    halo2_mean_duration = None
    if halo2_duration_sum and halo2_duration_count:
        count = float(halo2_duration_count)
        if count > 0:
            halo2_mean_duration = round(float(halo2_duration_sum) / count, 2)

    halo2_verified = query_prometheus("proofs_halo2_verified")
    halo2_invalid = query_prometheus("proofs_halo2_invalid")

    # Derived metrics
    batch_count = int(float(halo2_duration_count)) if halo2_duration_count else 0
    bundle_count = int(float(halo2_verified)) if halo2_verified else 0
    nullifier_count = int(float(orchard_nullifiers)) if orchard_nullifiers else 0

    actions_per_bundle = round(nullifier_count / bundle_count, 2) if bundle_count > 0 else None
    bundles_per_batch = round(bundle_count / batch_count, 2) if batch_count > 0 else None
    actions_per_batch = round(nullifier_count / batch_count, 2) if batch_count > 0 else None
    per_bundle_ms = round(halo2_mean_duration / bundles_per_batch, 2) if halo2_mean_duration and bundles_per_batch and bundles_per_batch > 0 else None
    per_action_ms = round(halo2_mean_duration / actions_per_batch, 2) if halo2_mean_duration and actions_per_batch and actions_per_batch > 0 else None

    return jsonify({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "orchard_nullifiers": nullifier_count or None,
        "halo2_proof_verification": {
            "last_duration_ms": round(float(halo2_last_duration), 2) if halo2_last_duration else None,
            "mean_duration_ms": halo2_mean_duration,
            "batch_count": batch_count,
            "verified": bundle_count,
            "invalid": int(float(halo2_invalid)) if halo2_invalid else 0,
        },
        "derived": {
            "actions_per_bundle": actions_per_bundle,
            "bundles_per_batch": bundles_per_batch,
            "actions_per_batch": actions_per_batch,
            "per_bundle_ms": per_bundle_ms,
            "per_action_ms": per_action_ms,
        },
    })


@app.route("/api/orchard-history")
def orchard_history():
    """Get Orchard Halo2 verification duration history for charts"""
    import time
    end = time.time()
    start = end - 3600

    duration_values = query_prometheus_range(
        "rate(consensus_proof_halo2_verification_duration_ms_sum[5m])"
        " / rate(consensus_proof_halo2_verification_duration_ms_count[5m])",
        start, end, "60s"
    )

    rate_values = query_prometheus_range(
        "rate(proofs_halo2_verified[5m]) * 300",
        start, end, "60s"
    )

    nullifier_values = query_prometheus_range(
        "rate(state_finalized_cumulative_orchard_nullifiers[5m]) * 300",
        start, end, "60s"
    )

    duration_data = [{"x": int(v[0]) * 1000, "y": round(float(v[1]), 2)} for v in duration_values if v[1] != "NaN"]
    rate_data = [{"x": int(v[0]) * 1000, "y": round(float(v[1]), 2)} for v in rate_values]
    nullifier_data = [{"x": int(v[0]) * 1000, "y": round(float(v[1]), 2)} for v in nullifier_values]

    return jsonify({
        "halo2_duration": duration_data,
        "halo2_rate": rate_data,
        "orchard_nullifiers_rate": nullifier_data,
    })


@app.route("/api/propagation-history")
def propagation_history():
    """Get RTT and processing delay history for charts"""
    import time
    end = time.time()
    start = end - 3600

    # RTT p50 over time
    rtt_values = query_prometheus_range(
        'zcash_net_peer_rtt_seconds{quantile="0.5"} * 1000',
        start, end, "60s"
    )

    # Processing delay p50 over time
    processing_values = query_prometheus_range(
        'zcash_block_processing_delay_seconds{quantile="0.5"} * 1000',
        start, end, "60s"
    )

    rtt_data = [{"x": int(v[0]) * 1000, "y": round(float(v[1]), 2)} for v in rtt_values]
    processing_data = [{"x": int(v[0]) * 1000, "y": round(float(v[1]), 2)} for v in processing_values]

    return jsonify({
        "rtt": rtt_data,
        "processing": processing_data
    })


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
