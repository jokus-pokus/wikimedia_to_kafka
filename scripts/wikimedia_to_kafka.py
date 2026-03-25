### python scripts/wikimedia_to_kafka.py -b localhost:9092 -t wikimedia_features --log-interval 500

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from typing import Optional, Dict, Any

import requests
from kafka import KafkaProducer
from sseclient import SSEClient


WIKI_SSE_URL = "https://stream.wikimedia.org/v2/stream/recentchange"


def extract_features(ev: Dict[str, Any], last_ts_ms: Optional[int]) -> tuple[Dict[str, Any], int]:
    ts_ms: Optional[int] = None
    try:
        ts_sec = ev.get("timestamp", None)
        if ts_sec is not None:
            ts_ms = int(ts_sec) * 1000
    except Exception:
        ts_ms = None
    if ts_ms is None:
        ts_ms = int(time.time() * 1000)

    ln = ev.get("length", {}) or {}
    new = ln.get("new")
    old = ln.get("old")
    delta = 0
    try:
        if new is not None and old is not None:
            delta = int(new) - int(old)
    except Exception:
        delta = 0

    import math
    log_abs_delta = float(math.log1p(abs(delta)))
    signed_delta = float((1 if delta > 0 else (-1 if delta < 0 else 0)) * math.log1p(abs(delta)))

    if last_ts_ms is None:
        inter_arrival_ms = 0.0
    else:
        inter_arrival_ms = float(max(0, ts_ms - last_ts_ms))

    tags = ev.get("tags", []) or []
    comment = (ev.get("comment", "") or "").lower()
    revert_tags = {"mw-rollback", "mw-undo", "mw-manual-revert"}
    revert_ind = 1.0 if (any(t in revert_tags for t in tags) or "revert" in comment) else 0.0

    feats = {
        "ts_ms": ts_ms,
        "log_abs_delta": log_abs_delta,
        "signed_delta": signed_delta,
        "inter_arrival_ms": inter_arrival_ms,
        "revert_ind": revert_ind,
    }
    return feats, ts_ms


def build_producer(
    bootstrap_servers: str,
    security_protocol: str = "PLAINTEXT",
    sasl_mechanism: Optional[str] = None,
    sasl_username: Optional[str] = None,
    sasl_password: Optional[str] = None,
    acks: str | int = "1",
    compression_type: Optional[str] = None,
    linger_ms: int = 100,
    retries: int = 3,
) -> KafkaProducer:
    kwargs: Dict[str, Any] = dict(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks=acks,
        linger_ms=linger_ms,
        retries=retries,
        max_in_flight_requests_per_connection=1,
    )
    if compression_type:
        kwargs["compression_type"] = compression_type
    if security_protocol:
        kwargs["security_protocol"] = security_protocol
    if security_protocol and security_protocol.startswith("SASL"):
        if sasl_mechanism:
            kwargs["sasl_mechanism"] = sasl_mechanism
        if sasl_username is not None:
            kwargs["sasl_plain_username"] = sasl_username
        if sasl_password is not None:
            kwargs["sasl_plain_password"] = sasl_password
    return KafkaProducer(**kwargs)


def stream_to_kafka(
    topic: str,
    bootstrap_servers: str,
    security_protocol: str,
    sasl_mechanism: Optional[str],
    sasl_username: Optional[str],
    sasl_password: Optional[str],
    acks: str | int,
    compression_type: Optional[str],
    log_interval: int,
) -> int:
    stop = False

    def _sig_handler(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    prod = build_producer(
        bootstrap_servers,
        security_protocol,
        sasl_mechanism,
        sasl_username,
        sasl_password,
        acks=acks,
        compression_type=compression_type,
    )

    sent = 0
    failed = 0
    last_ts_local: Optional[int] = None
    backoff = 1.0

    while not stop:
        try:
            # Connect to SSE with generous read timeout to avoid idle disconnects
            with requests.get(WIKI_SSE_URL, stream=True, headers={
                "Accept": "text/event-stream",
                "User-Agent": "ClaSPy-Stream/1.0 (+classification-label-profile)"
            }, timeout=(5, 300)) as resp:
                client = SSEClient(resp)
                for msg in client.events():
                    if stop:
                        break
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        continue
                    feats, last_ts_local = extract_features(data, last_ts_local)
                    try:
                        prod.send(topic, feats)
                        sent += 1
                    except Exception:
                        failed += 1
                    if log_interval and (sent + failed) % log_interval == 0:
                        print(f"stats topic={topic} sent={sent} failed={failed}")
            backoff = 1.0
        except Exception as e:
            if stop:
                break
            # Backoff and retry
            sleep_s = backoff
            print(f"warn: SSE loop error: {e} -> retrying in {sleep_s:.1f}s", file=sys.stderr)
            time.sleep(sleep_s)
            backoff = min(30.0, backoff * 2.0)

    try:
        prod.flush(timeout=5)
    except Exception:
        pass
    try:
        prod.close(timeout=5)
    except Exception:
        pass

    print(f"exiting: sent={sent} failed={failed}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Stream Wikimedia RecentChanges features to Kafka.")
    ap.add_argument("--bootstrap", "-b", default="localhost:9092", help="Bootstrap servers, e.g., host:9092")
    ap.add_argument("--topic", "-t", default="wikimedia_features", help="Kafka topic to publish to")

    ap.add_argument("--security-protocol", choices=["PLAINTEXT", "SASL_PLAINTEXT", "SASL_SSL", "SSL"], default="PLAINTEXT")
    ap.add_argument("--sasl-mechanism", choices=["PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"], default=None)
    ap.add_argument("--sasl-username", default=None)
    ap.add_argument("--sasl-password", default=None)

    ap.add_argument("--acks", default="1", help="Kafka acks setting (e.g., 1, all)")
    ap.add_argument("--compression-type", choices=["gzip", "snappy", "lz4", "zstd"], default=None)
    ap.add_argument("--log-interval", type=int, default=500, help="Log stats every N messages (0=disable)")

    args = ap.parse_args()
    return stream_to_kafka(
        topic=args.topic,
        bootstrap_servers=args.bootstrap,
        security_protocol=args.security_protocol,
        sasl_mechanism=args.sasl_mechanism,
        sasl_username=args.sasl_username,
        sasl_password=args.sasl_password,
        acks=args.acks,
        compression_type=args.compression_type,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    sys.exit(main())
