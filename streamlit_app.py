from __future__ import annotations
import io
import json
import threading
import time
import sys
from collections import deque
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

# Optional autorefresh utility for periodic reruns
try:
    from streamlit_autorefresh import st_autorefresh  # from package: streamlit-autorefresh
except Exception:
    st_autorefresh = None

st.set_page_config(page_title="ClaSPy App", layout="wide")
st.title("ClaSPy: Segmentation and State Detection")


@st.cache_data(show_spinner=False)
def _load_txt(content: bytes, delimiter: str, skiprows: int) -> np.ndarray:
    text = content.decode("utf-8", errors="ignore")
    buf = io.StringIO(text)
    delim = None if delimiter == "space" else ("\t" if delimiter == "tab" else ("," if delimiter == "comma" else ";"))
    arr = np.loadtxt(buf, delimiter=delim, skiprows=skiprows)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(float)

def _run_segmentation(ts: np.ndarray, params: dict):
    from claspy.segmentation import BinaryClaSPSegmentation
    seg = BinaryClaSPSegmentation(**params)
    cps = seg.fit_predict(ts)
    return seg, cps


def _run_state_detection(ts: np.ndarray, params: dict, change_points: np.ndarray | None):
    from claspy.state_detection import AgglomerativeCLaPDetection
    clap = AgglomerativeCLaPDetection(**params)
    state_seq = clap.fit_predict(ts, change_points=change_points)
    return clap, state_seq


with st.sidebar:
    st.header("Upload settings")
    uploaded = st.file_uploader("Upload a .txt time series", type=["txt"], accept_multiple_files=False)
    delimiter = st.selectbox("Delimiter", options=["space", "tab", "comma", "semicolon"], index=0, key="sb_delim")
    skiprows = st.number_input("Header rows to skip", key="sb_skiprows", min_value=0, max_value=100000, value=0, step=1)
    with st.expander("Environment"):
        st.caption("Active Python interpreter")
        st.code(sys.executable)
        st.caption("Python version")
        st.text(sys.version)

tabs = st.tabs(["Segmentation", "State Detection", "Streaming"])

with tabs[0]:
    st.subheader("Binary ClaSP Segmentation")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_segments_mode = st.selectbox("Number of segments", options=["learn", "fixed"], index=0, key="seg_nseg_mode")
        n_segments = st.number_input("If fixed: n_segments", key="seg_nsegments", min_value=1, max_value=10_000, value=5, step=1)
    with col2:
        window_size = st.selectbox("Window size", options=["suss", "fft", "acf", "fixed"], index=0, key="seg_window_size")
        ws_fixed = st.number_input("If fixed: window_size", key="seg_ws_fixed", min_value=2, max_value=100_000, value=50, step=1)
    with col3:
        validation = st.selectbox("Validation", options=["significance_test", "score_threshold"], index=0, key="seg_validation")
        threshold = st.text_input("Threshold (or 'default')", value="default", key="seg_threshold")
    run_seg = st.button("Run Segmentation", type="primary")

    if run_seg:
        if uploaded is None:
            st.error("Please upload a .txt file first.")
            st.stop()
        try:
            ts = _load_txt(uploaded.getvalue(), delimiter, skiprows)
        except Exception as e:
            st.exception(e)
            st.stop()

        params = {
            "n_segments": (n_segments if n_segments_mode == "fixed" else "learn"),
            "window_size": (ws_fixed if window_size == "fixed" else window_size),
            "validation": validation,
            "threshold": (threshold if threshold != "default" else "default"),
        }
        with st.spinner("Running ClaSP segmentation..."):
            try:
                seg, cps = _run_segmentation(ts, params)
            except Exception as e:
                st.exception(e)
                st.stop()

        # Use built-in plot
        plt.clf()
        seg.plot(heading="Segmentation", ts_name="TS", fig_size=(12, 6), font_size=14)
        st.pyplot(plt.gcf(), clear_figure=True)

        # Dataframe
        seg_starts = [0] + list(cps)
        seg_ends = list(cps) + [ts.shape[0]]
        df_segments = pd.DataFrame({
            "segment": np.arange(len(seg_starts)),
            "start": seg_starts,
            "end": seg_ends,
            "length": np.array(seg_ends) - np.array(seg_starts),
        })
        st.subheader("Segments")
        st.dataframe(df_segments, use_container_width=True)

with tabs[1]:
    st.subheader("Agglomerative CLaP State Detection")
    col1, col2 = st.columns(2)
    with col1:
        classifier = st.selectbox("Classifier", options=["rocket", "mrhydra", "weasel", "quant", "rdst", "proximityforest", "freshprince", "inception", "dummy"], index=0, key="sd_classifier")
        n_splits = st.number_input("CV folds (n_splits)", key="sd_n_splits", min_value=2, max_value=10, value=5, step=1)
        sample_size = st.number_input("Sample size per class", key="sd_sample_size", min_value=100, max_value=100000, value=1000, step=100)
    with col2:
        window_size_sd = st.selectbox("Window size", options=["suss", "fft", "acf", "fixed"], index=0, key="sd_window_size")
        ws_sd_fixed = st.number_input("If fixed: window_size", key="sd_ws_fixed", min_value=2, max_value=100_000, value=50, step=1)
        manual_cps_sd = st.checkbox("Provide CPs manually", value=False, key="sd_manual_cps")
        cps_sd_text = st.text_input("Manual CPs (comma-separated)", value="", key="sd_cps_text")
    run_sd = st.button("Run State Detection", type="primary")

    if run_sd:
        if uploaded is None:
            st.error("Please upload a .txt file first.")
            st.stop()
        try:
            ts = _load_txt(uploaded.getvalue(), delimiter, skiprows)
        except Exception as e:
            st.exception(e)
            st.stop()

        change_points = None
        if manual_cps_sd:
            try:
                cps_arr = np.array([int(x) for x in cps_sd_text.split(',') if x.strip()!=''], dtype=int)
                cps_arr = cps_arr[(cps_arr > 0) & (cps_arr < ts.shape[0])]
                change_points = np.unique(np.sort(cps_arr))
            except Exception:
                st.error("Invalid manual CP list. Use comma-separated integers.")
                st.stop()

        params = {
            "window_size": (ws_sd_fixed if window_size_sd == "fixed" else window_size_sd),
            "classifier": classifier,
            "n_splits": int(n_splits),
            "sample_size": int(sample_size),
        }
        with st.spinner("Running CLaP state detection..."):
            try:
                clap, state_seq = _run_state_detection(ts, params, change_points)
            except Exception as e:
                st.exception(e)
                st.stop()

        # Full plots: state detection and process
        plt.clf()
        clap.plot(heading="Detected States", ts_name="TS", fig_size=(12, 8), font_size=14, sparse=False)
        st.pyplot(plt.gcf(), clear_figure=True)

        plt.clf()
        clap.plot(heading="Discovered Process", fig_size=(8, 6), font_size=12, sparse=True)
        st.pyplot(plt.gcf(), clear_figure=True)

        # Data outputs
        cps_m = clap.get_change_points()
        labels_m = clap.get_segment_labels()
        seg_starts = [0] + list(cps_m)
        seg_ends = list(cps_m) + [ts.shape[0]]
        df_states = pd.DataFrame({
            "segment": np.arange(len(labels_m)),
            "start": seg_starts,
            "end": seg_ends,
            "label": labels_m,
            "length": np.array(seg_ends) - np.array(seg_starts),
        })
        st.subheader("State-labeled segments")
        st.dataframe(df_states, use_container_width=True)

        states, transitions = clap.predict(sparse=True)
        st.write({"states": list(states), "transitions": list(map(list, transitions))})

with tabs[2]:
    st.subheader("Wikimedia RecentChanges Stream (SSE)")
    st.caption("Source: https://stream.wikimedia.org/v2/stream/recentchange")

    # Session state init
    if "sse_thread" not in st.session_state:
        st.session_state.sse_thread = None
    if "sse_stop" not in st.session_state:
        st.session_state.sse_stop = None
    if "sse_buffer" not in st.session_state:
        st.session_state.sse_buffer = deque(maxlen=20000)
    if "sse_connected" not in st.session_state:
        st.session_state.sse_connected = False
    if "sse_last_event_ms" not in st.session_state:
        st.session_state.sse_last_event_ms = None
    if "sse_lock" not in st.session_state:
        st.session_state.sse_lock = threading.Lock()
    # Kafka counters and lock
    if "kafka_stats" not in st.session_state:
        st.session_state.kafka_stats = {"sent": 0, "failed": 0}
    if "kafka_lock" not in st.session_state:
        st.session_state.kafka_lock = threading.Lock()

    url = "https://stream.wikimedia.org/v2/stream/recentchange"
    st.markdown("Select a signal to process (or choose a fused combination).")
    sig_choice = st.selectbox(
        "Signal",
        options=["log_abs_delta", "signed_delta", "inter_arrival_ms", "revert_ind", "fused"],
        index=0,
        key="sse_sig_choice",
    )

    colA, colB, colC = st.columns(3)
    with colA:
        n_timepoints = st.number_input("Streaming window (n_timepoints)", min_value=200, max_value=100000, value=2000, step=100, key="sse_n_timepoints")
    with colB:
        n_warmup = st.number_input("Warmup points (n_warmup)", min_value=100, max_value=100000, value=1000, step=100, key="sse_n_warmup")
    with colC:
        jump = st.number_input("Jump (process step)", min_value=1, max_value=100, value=5, step=1, key="sse_jump")

    if sig_choice == "fused":
        st.markdown("""
        A fused signal is a single numeric stream created by combining several standardized features with user-defined weights. This helps capture multiple aspects of the data in one signal for streaming segmentation.
        """)
        colw1, colw2, colw3, colw4 = st.columns(4)
        with colw1:
            w_logabs = st.number_input("w_log_abs_delta", value=1.0, step=0.1, key="f_w_logabs")
        with colw2:
            w_signed = st.number_input("w_signed_delta", value=0.5, step=0.1, key="f_w_signed")
        with colw3:
            w_iam = st.number_input("w_inter_arrival", value=0.5, step=0.1, key="f_w_iam")
        with colw4:
            w_rev = st.number_input("w_revert_ind", value=1.0, step=0.1, key="f_w_rev")
        zscore_cont = st.checkbox("Z-score continuous features", value=True, key="f_zscore")

    # Kafka sink configuration
    with st.expander("Kafka sink (optional)"):
        kafka_enable = st.checkbox("Enable Kafka sink", value=False, key="kfk_enable")
        colk1, colk2 = st.columns([2, 1])
        with colk1:
            kafka_bootstrap = st.text_input("Bootstrap servers", value="localhost:9092", key="kfk_bootstrap")
        with colk2:
            kafka_topic = st.text_input("Topic", value="wikimedia_features", key="kfk_topic")
        colk3, colk4 = st.columns(2)
        with colk3:
            kafka_sec = st.selectbox("Security protocol", options=["PLAINTEXT", "SASL_PLAINTEXT", "SASL_SSL", "SSL"], index=0, key="kfk_sec")
        with colk4:
            kafka_mech = st.selectbox("SASL mechanism", options=["PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"], index=0, key="kfk_mech")
        colk5, colk6 = st.columns(2)
        with colk5:
            kafka_user = st.text_input("SASL username", value="", key="kfk_user")
        with colk6:
            kafka_pass = st.text_input("SASL password", value="", type="password", key="kfk_pass")

    # Prepare session state for Kafka producer
    if "kafka_producer" not in st.session_state:
        st.session_state.kafka_producer = None
    if "kafka_cfg" not in st.session_state:
        st.session_state.kafka_cfg = None

    def _extract_features(ev: dict, last_ts_ms: int | None) -> tuple[dict, int]:
        ts_ms = None
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
        log_abs_delta = float(np.log1p(abs(delta)))
        signed_delta = float(np.sign(delta) * np.log1p(abs(delta)))

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

    def _make_kafka_producer(cfg: dict | None):
        if not cfg:
            return None
        try:
            from kafka import KafkaProducer
        except Exception:
            return None
        try:
            kwargs = {
                "bootstrap_servers": cfg.get("bootstrap_servers", "localhost:9092"),
                "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
                "linger_ms": 100,
                "retries": 3,
                "max_in_flight_requests_per_connection": 1,
            }
            sec = cfg.get("security_protocol", "PLAINTEXT")
            if sec:
                kwargs["security_protocol"] = sec
            if sec.startswith("SASL"):
                if cfg.get("sasl_mechanism"):
                    kwargs["sasl_mechanism"] = cfg.get("sasl_mechanism")
                if cfg.get("sasl_username") is not None:
                    kwargs["sasl_plain_username"] = cfg.get("sasl_username")
                if cfg.get("sasl_password") is not None:
                    kwargs["sasl_plain_password"] = cfg.get("sasl_password")
            return KafkaProducer(**kwargs)
        except Exception:
            return None

    def _sse_worker(stop_evt: threading.Event, buf: deque, kafka_producer=None, kafka_topic: str | None = None, kafka_stats: dict | None = None, kafka_lock: threading.Lock | None = None):
        import requests
        from sseclient import SSEClient
        headers = {
            "Accept": "text/event-stream",
            "User-Agent": "ClaSPy-Stream-App/0.1 (+https://github.com/ermshaua/claspy)"
        }
        def _inc_counter(which: str):
            if kafka_stats is None or kafka_lock is None:
                return
            try:
                with kafka_lock:
                    kafka_stats[which] = int(kafka_stats.get(which, 0)) + 1
            except Exception:
                pass

        backoff = 1.0
        while not stop_evt.is_set():
            try:
                # Use generous read timeout; server should send keepalives
                with requests.get(url, stream=True, headers=headers, timeout=(5, 300)) as resp:
                    client = SSEClient(resp)
                    last_ts_local = None
                    for msg in client.events():
                        if stop_evt.is_set():
                            break
                        try:
                            data = json.loads(msg.data)
                        except Exception:
                            continue
                        feats, last_ts_local = _extract_features(data, last_ts_local)
                        buf.append(feats)
                        try:
                            with st.session_state.sse_lock:
                                st.session_state.sse_last_event_ms = int(time.time() * 1000)
                        except Exception:
                            pass
                        # Send to Kafka if configured
                        if kafka_producer is not None and kafka_topic:
                            try:
                                fut = kafka_producer.send(kafka_topic, feats)
                                try:
                                    fut.add_callback(lambda _r, _inc=_inc_counter: _inc("sent"))
                                    fut.add_errback(lambda _e, _inc=_inc_counter: _inc("failed"))
                                except Exception:
                                    _inc_counter("sent")
                            except Exception:
                                _inc_counter("failed")
                backoff = 1.0
            except Exception:
                time.sleep(backoff)
                backoff = min(30.0, backoff * 2.0)
                continue

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if not st.session_state.sse_connected:
            if st.button("Connect", type="primary"):
                st.session_state.sse_buffer.clear()
                st.session_state.sse_stop = threading.Event()
                # Build Kafka config if enabled
                kcfg = None
                if kafka_enable:
                    kcfg = {
                        "bootstrap_servers": kafka_bootstrap.strip(),
                        "security_protocol": kafka_sec,
                        "sasl_mechanism": kafka_mech if kafka_sec.startswith("SASL") else None,
                        "sasl_username": kafka_user if kafka_sec.startswith("SASL") else None,
                        "sasl_password": kafka_pass if kafka_sec.startswith("SASL") else None,
                        "topic": kafka_topic.strip(),
                    }
                    st.session_state.kafka_cfg = kcfg
                    # Create/refresh a producer
                    prod = _make_kafka_producer(kcfg)
                    st.session_state.kafka_producer = prod
                # Reset Kafka counters on each connect
                try:
                    with st.session_state.kafka_lock:
                        st.session_state.kafka_stats["sent"] = 0
                        st.session_state.kafka_stats["failed"] = 0
                except Exception:
                    pass

                t = threading.Thread(
                    target=_sse_worker,
                    args=(
                        st.session_state.sse_stop,
                        st.session_state.sse_buffer,
                        st.session_state.kafka_producer,
                        (st.session_state.kafka_cfg or {}).get("topic") if kafka_enable else None,
                        st.session_state.kafka_stats,
                        st.session_state.kafka_lock,
                    ),
                    daemon=True,
                )
                t.start()
                st.session_state.sse_thread = t
                st.session_state.sse_connected = True
                # Immediately rerun to start plotting without waiting
                try:
                    if hasattr(st, "rerun") and callable(getattr(st, "rerun")):
                        st.rerun()
                    elif hasattr(st, "experimental_rerun") and callable(getattr(st, "experimental_rerun")):
                        st.experimental_rerun()
                except Exception:
                    pass
        else:
            st.success("Connected")
    with c2:
        if st.session_state.sse_connected:
            if st.button("Disconnect"):
                if st.session_state.sse_stop is not None:
                    st.session_state.sse_stop.set()
                st.session_state.sse_connected = False
                st.session_state.sse_thread = None
                # Close Kafka producer if any
                if st.session_state.kafka_producer is not None:
                    try:
                        st.session_state.kafka_producer.flush(timeout=5)
                    except Exception:
                        pass
                    try:
                        st.session_state.kafka_producer.close(timeout=5)
                    except Exception:
                        pass
                    st.session_state.kafka_producer = None
                st.info("Disconnected")
    with c3:
        st.metric("Buffered points", f"{len(st.session_state.sse_buffer)}")
    with c4:
        stats = st.session_state.kafka_stats if "kafka_stats" in st.session_state else {"sent": 0, "failed": 0}
        sent = int(stats.get("sent", 0))
        failed = int(stats.get("failed", 0))
        label = "Kafka sent" if failed == 0 else f"Kafka sent (fail {failed})"
        st.metric(label, f"{sent}")
    with c5:
        last_ms = st.session_state.sse_last_event_ms
        if last_ms is None:
            st.metric("Last event (s)", "-")
        else:
            age_s = max(0, int((time.time() * 1000 - last_ms) / 1000))
            st.metric("Last event (s)", f"{age_s}")

    st.divider()

    # Auto-refresh controls for live preview
    ar1, ar2 = st.columns([1, 1])
    with ar1:
        auto_refresh = st.checkbox("Auto-refresh preview", value=True, key="sse_auto_refresh")
    with ar2:
        refresh_ms = st.number_input(
            "Refresh interval (ms)", min_value=200, max_value=10000, value=1000, step=100, key="sse_refresh_ms"
        )

    # Inline feature preview of last N points
    st.subheader("Feature preview")
    n_preview = st.number_input(
        "Preview last N points",
        min_value=50,
        max_value=20000,
        value=1000,
        step=50,
        key="sse_preview_n",
    )
    feats_list = list(st.session_state.sse_buffer)[-int(n_preview):]
    if len(feats_list) == 0:
        st.info("Waiting for data… connect to start buffering.")
    else:
        try:
            ts_ms = np.asarray([f["ts_ms"] for f in feats_list], dtype=float)
            t_rel_s = (ts_ms - ts_ms[0]) / 1000.0 if ts_ms.size > 0 else np.arange(len(feats_list))
            df_prev = pd.DataFrame(
                {
                    "t_s": t_rel_s,
                    "log_abs_delta": [float(f["log_abs_delta"]) for f in feats_list],
                    "signed_delta": [float(f["signed_delta"]) for f in feats_list],
                    "inter_arrival_ms": [float(f["inter_arrival_ms"]) for f in feats_list],
                    "revert_ind": [float(f["revert_ind"]) for f in feats_list],
                }
            ).set_index("t_s")

            pc1, pc2 = st.columns(2)
            with pc1:
                st.caption("log_abs_delta")
                st.line_chart(df_prev[["log_abs_delta"]])
            with pc2:
                st.caption("signed_delta")
                st.line_chart(df_prev[["signed_delta"]])

            pc3, pc4 = st.columns(2)
            with pc3:
                st.caption("inter_arrival_ms")
                st.line_chart(df_prev[["inter_arrival_ms"]])
            with pc4:
                st.caption("revert_ind")
                st.line_chart(df_prev[["revert_ind"]])

            # If fused is selected, preview fused as well
            if 'sig_choice' in locals() and sig_choice == "fused":
                lad = df_prev["log_abs_delta"].to_numpy()
                sgd = df_prev["signed_delta"].to_numpy()
                iam = df_prev["inter_arrival_ms"].to_numpy()
                rev = df_prev["revert_ind"].to_numpy()

                def z(x):
                    m = np.mean(x)
                    sd = np.std(x)
                    return (x - m) / sd if sd != 0 else x * 0.0

                if 'zscore_cont' in locals() and zscore_cont:
                    lad_z, sgd_z, iam_z = z(lad), z(sgd), z(iam)
                else:
                    lad_z, sgd_z, iam_z = lad, sgd, iam

                fused = w_logabs * lad_z + w_signed * sgd_z + w_iam * iam_z + w_rev * rev
                df_prev_fused = pd.DataFrame({"t_s": df_prev.index.values, "fused": fused}).set_index("t_s")
                st.caption("fused (weighted combination)")
                st.line_chart(df_prev_fused)
        except Exception as e:
            st.info("Feature preview unavailable yet. Connect the stream and buffer more points.")

    # Trigger periodic reruns while connected to keep preview live
    if st.session_state.sse_connected and auto_refresh:
        if st_autorefresh is not None:
            st_autorefresh(interval=int(refresh_ms), key="sse_autorefresh_tick")
        else:
            st.caption("Tip: install 'streamlit-autorefresh' for smooth live updates.")

    st.subheader("Process buffer with Streaming ClaSP (optional)")
    run_stream = st.button("Run streaming segmentation on buffered data")
    if run_stream:
        if len(st.session_state.sse_buffer) < int(n_warmup) + 10:
            st.warning("Not enough points buffered yet. Increase buffer or wait longer.")
        else:
            from claspy.streaming.segmentation import StreamingClaSPSegmentation
            clasp = StreamingClaSPSegmentation(
                n_timepoints=int(n_timepoints),
                n_warmup=int(n_warmup),
                window_size="suss",
                jump=int(jump),
                log_cps=True,
            )
            feats_list = list(st.session_state.sse_buffer)
            if sig_choice == "fused":
                lad = np.asarray([f["log_abs_delta"] for f in feats_list], dtype=float)
                sgd = np.asarray([f["signed_delta"] for f in feats_list], dtype=float)
                iam = np.asarray([f["inter_arrival_ms"] for f in feats_list], dtype=float)
                rev = np.asarray([f["revert_ind"] for f in feats_list], dtype=float)

                def z(x):
                    m = np.mean(x)
                    sd = np.std(x)
                    if sd == 0:
                        return x * 0.0
                    return (x - m) / sd

                if zscore_cont:
                    lad_z, sgd_z, iam_z = z(lad), z(sgd), z(iam)
                else:
                    lad_z, sgd_z, iam_z = lad, sgd, iam

                fused = w_logabs * lad_z + w_signed * sgd_z + w_iam * iam_z + w_rev * rev
                numeric = fused.tolist()
                stream_name = "fused"
            else:
                stream_name = sig_choice
                numeric = [float(f[sig_choice]) for f in feats_list]

            for v in numeric:
                clasp.update(float(v))
            try:
                last_cp = clasp.predict()
            except Exception as e:
                last_cp = 0
            st.write({"last_cp": last_cp, "logged_cps": clasp.change_points})

            # Plot current sliding window and profile
            plt.clf()
            try:
                clasp.plot(heading="Streaming ClaSP on Wikimedia signal", stream_name=stream_name, fig_size=(12, 6), font_size=14)
                st.pyplot(plt.gcf(), clear_figure=True)
            except Exception as e:
                st.exception(e)
