import argparse
import json
from collections import deque

import requests
from sseclient import SSEClient


def value_from_event(ev: dict, kind: str) -> float | None:
    try:
        if kind == "length_delta":
            ln = ev.get("length", {}) or {}
            new = ln.get("new")
            old = ln.get("old")
            if new is None or old is None:
                return None
            return float(new - old)
        if kind == "comment_len":
            c = ev.get("comment", "") or ""
            return float(len(c))
        if kind == "minor_edit":
            return 1.0 if ev.get("minor", False) else 0.0
        if kind == "is_bot":
            return 1.0 if ev.get("bot", False) else 0.0
    except Exception:
        return None
    return None


def stream_events(url: str, map_kind: str, limit: int, timeout: int = 30):
    headers = {
        "Accept": "text/event-stream",
        "User-Agent": "ClaSPy-Stream-Test/0.1 (+https://github.com/ermshaua/claspy)"
    }
    buf = deque()
    with requests.get(url, stream=True, headers=headers, timeout=timeout) as resp:
        resp.raise_for_status()
        client = SSEClient(resp)
        for msg in client.events():
            try:
                data = json.loads(msg.data)
            except Exception:
                continue
            v = value_from_event(data, map_kind)
            if v is None:
                continue
            buf.append(float(v))
            if len(buf) >= limit:
                break
    return list(buf)


def run_streaming_clasp(values, n_timepoints: int, n_warmup: int, jump: int, log_cps: bool, plot_path: str | None, stream_name: str):
    from claspy.streaming.segmentation import StreamingClaSPSegmentation
    import matplotlib.pyplot as plt

    clasp = StreamingClaSPSegmentation(
        n_timepoints=int(n_timepoints),
        n_warmup=int(n_warmup),
        window_size="suss",
        jump=int(jump),
        log_cps=bool(log_cps),
    )
    for v in values:
        clasp.update(float(v))

    try:
        last_cp = clasp.predict()
    except Exception:
        last_cp = 0

    print({
        "points": len(values),
        "last_cp": last_cp,
        "logged_cps": clasp.change_points,
    })

    if plot_path:
        plt.clf()
        clasp.plot(heading="Streaming ClaSP on Wikimedia signal", stream_name=stream_name, fig_size=(12, 6), font_size=14, file_path=plot_path)
        print(f"Saved plot to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Test Wikimedia SSE -> numeric signal -> (optional) Streaming ClaSP")
    parser.add_argument("--url", default="https://stream.wikimedia.org/v2/stream/recentchange", help="SSE endpoint URL")
    parser.add_argument("--map", dest="map_kind", choices=["length_delta", "comment_len", "minor_edit", "is_bot"], default="length_delta", help="Event->signal mapping")
    parser.add_argument("--limit", type=int, default=5000, help="Number of points to buffer from SSE")
    parser.add_argument("--window", dest="n_timepoints", type=int, default=2000, help="Streaming window length")
    parser.add_argument("--warmup", dest="n_warmup", type=int, default=1000, help="Warmup points before detection")
    parser.add_argument("--jump", type=int, default=5, help="Process step between detections")
    parser.add_argument("--log-cps", action="store_true", help="Log detected change points")
    parser.add_argument("--plot", dest="plot_path", default=None, help="Optional path to save plot (PNG)")

    args = parser.parse_args()

    print(f"Connecting to {args.url} … (map={args.map_kind}, limit={args.limit})")
    values = stream_events(args.url, args.map_kind, args.limit)
    if len(values) == 0:
        print("No numeric values received. Try a different mapping or higher limit.")
        return

    print(f"Buffered {len(values)} points. Running Streaming ClaSP …")
    run_streaming_clasp(values, args.n_timepoints, args.n_warmup, args.jump, args.log_cps, args.plot_path, args.map_kind)


if __name__ == "__main__":
    main()
