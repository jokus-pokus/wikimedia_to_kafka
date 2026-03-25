from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from typing import Optional, Dict, Any, List

from kafka import KafkaConsumer, TopicPartition

### python scripts/kafka_tail.py -b localhost:9092 -t wikimedia_features -n 10 --tail-last


def _build_consumer(
    bootstrap_servers: str,
    security_protocol: str = "PLAINTEXT",
    sasl_mechanism: Optional[str] = None,
    sasl_username: Optional[str] = None,
    sasl_password: Optional[str] = None,
    group_id: Optional[str] = None,
    timeout_ms: int = 5000,
) -> KafkaConsumer:
    kwargs: Dict[str, Any] = dict(
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda v: v,  # bytes → decode later (raw or json)
        enable_auto_commit=False,
        api_version_auto_timeout_ms=timeout_ms,
        request_timeout_ms=timeout_ms,
        session_timeout_ms=min(10000, max(6000, timeout_ms)),
        connections_max_idle_ms=300000,
        consumer_timeout_ms=timeout_ms,  # stops iteration when no more data
    )
    if group_id is not None:
        kwargs["group_id"] = group_id

    if security_protocol:
        kwargs["security_protocol"] = security_protocol
    if security_protocol and security_protocol.startswith("SASL"):
        if sasl_mechanism:
            kwargs["sasl_mechanism"] = sasl_mechanism
        if sasl_username is not None:
            kwargs["sasl_plain_username"] = sasl_username
        if sasl_password is not None:
            kwargs["sasl_plain_password"] = sasl_password

    return KafkaConsumer(**kwargs)


def _decode_value(b: bytes, pretty: bool) -> str:
    if b is None:
        return "null"
    try:
        obj = json.loads(b.decode("utf-8"))
        return json.dumps(obj, indent=2 if pretty else None)
    except Exception:
        try:
            return b.decode("utf-8", errors="replace")
        except Exception:
            return str(b)


def consume_simple(
    topic: str,
    bootstrap_servers: str,
    max_messages: int,
    from_beginning: bool,
    security_protocol: str,
    sasl_mechanism: Optional[str],
    sasl_username: Optional[str],
    sasl_password: Optional[str],
    timeout_ms: int,
    pretty: bool,
) -> int:
    """Subscribe and read up to N messages, from beginning or latest."""
    group_id = None  # no commits
    consumer = _build_consumer(
        bootstrap_servers,
        security_protocol,
        sasl_mechanism,
        sasl_username,
        sasl_password,
        group_id,
        timeout_ms,
    )
    try:
        auto_offset = "earliest" if from_beginning else "latest"
        consumer.subscribe([topic])
        consumer.poll(timeout_ms=200)  # trigger assignment
        consumer.seek_to_beginning() if from_beginning else None
        count = 0
        for msg in consumer:
            line = _decode_value(msg.value, pretty)
            print(line)
            count += 1
            if count >= max_messages:
                break
        return 0 if count > 0 else 2
    finally:
        consumer.close()


def consume_tail(
    topic: str,
    bootstrap_servers: str,
    max_messages: int,
    security_protocol: str,
    sasl_mechanism: Optional[str],
    sasl_username: Optional[str],
    sasl_password: Optional[str],
    timeout_ms: int,
    pretty: bool,
) -> int:
    """Best-effort tail of last N messages across partitions.

    Strategy: assign all partitions, compute per-partition start offsets as
    max(end-N, begin), read forward until reaching end offsets; keep a deque
    of size N to emit the last N seen messages.
    """
    consumer = _build_consumer(
        bootstrap_servers,
        security_protocol,
        sasl_mechanism,
        sasl_username,
        sasl_password,
        group_id=None,
        timeout_ms=timeout_ms,
    )
    try:
        parts = consumer.partitions_for_topic(topic)
        if not parts:
            print(f"Topic '{topic}' has no partitions or does not exist.", file=sys.stderr)
            return 3
        tps = [TopicPartition(topic, p) for p in sorted(parts)]
        consumer.assign(tps)

        begin = consumer.beginning_offsets(tps)
        end = consumer.end_offsets(tps)

        # Compute starting offsets. Distribute roughly across partitions.
        approx_per_part = max(1, max_messages // max(1, len(tps)))
        for tp in tps:
            start = max(begin[tp], end[tp] - approx_per_part)
            consumer.seek(tp, start)

        # Collect until we reach end offsets or timeout
        buf: deque = deque(maxlen=max_messages)
        start_time = time.time()
        while True:
            polled = consumer.poll(timeout_ms=200)
            any_records = False
            for records in polled.values():
                if records:
                    any_records = True
                for msg in records:
                    buf.append(msg)
            # Stop when all partitions are at end
            at_end = all(consumer.position(tp) >= end[tp] for tp in tps)
            if at_end:
                break
            if not any_records and (time.time() - start_time) * 1000.0 > timeout_ms:
                break

        # Print last N by timestamp (approx chronological across partitions)
        msgs: List = list(buf)
        try:
            msgs.sort(key=lambda m: getattr(m, "timestamp", 0))
        except Exception:
            pass
        for m in msgs[-max_messages:]:
            print(_decode_value(m.value, pretty))
        return 0 if len(msgs) > 0 else 2
    finally:
        consumer.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Tail or consume Kafka topic messages for quick verification.")
    ap.add_argument("--bootstrap", "-b", default="localhost:9092", help="Bootstrap servers, e.g., host:9092")
    ap.add_argument("--topic", "-t", required=True, help="Topic name to consume")
    ap.add_argument("--max-messages", "-n", type=int, default=10, help="Maximum number of messages to print")
    ap.add_argument("--timeout-ms", type=int, default=5000, help="Consumer timeout when idle (ms)")

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--from-beginning", action="store_true", help="Start from earliest offset and read N messages (default)")
    mode.add_argument("--tail-last", action="store_true", help="Best-effort tail of the last N messages across partitions")

    ap.add_argument("--security-protocol", choices=["PLAINTEXT", "SASL_PLAINTEXT", "SASL_SSL", "SSL"], default="PLAINTEXT")
    ap.add_argument("--sasl-mechanism", choices=["PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"], default=None)
    ap.add_argument("--sasl-username", default=None)
    ap.add_argument("--sasl-password", default=None)

    fmt = ap.add_mutually_exclusive_group()
    fmt.add_argument("--pretty", dest="pretty", action="store_true", help="Pretty-print JSON (default)")
    fmt.add_argument("--raw", dest="pretty", action="store_false", help="Print raw values without JSON pretty-print")
    ap.set_defaults(pretty=True, from_beginning=True)

    args = ap.parse_args()

    if args.tail_last:
        return consume_tail(
            topic=args.topic,
            bootstrap_servers=args.bootstrap,
            max_messages=args.max_messages,
            security_protocol=args.security_protocol,
            sasl_mechanism=args.sasl_mechanism,
            sasl_username=args.sasl_username,
            sasl_password=args.sasl_password,
            timeout_ms=args.timeout_ms,
            pretty=args.pretty,
        )
    else:
        # default simple mode uses from-beginning flag
        return consume_simple(
            topic=args.topic,
            bootstrap_servers=args.bootstrap,
            max_messages=args.max_messages,
            from_beginning=args.from_beginning,
            security_protocol=args.security_protocol,
            sasl_mechanism=args.sasl_mechanism,
            sasl_username=args.sasl_username,
            sasl_password=args.sasl_password,
            timeout_ms=args.timeout_ms,
            pretty=args.pretty,
        )


if __name__ == "__main__":
    sys.exit(main())
