# CLaP / ClaSPy Visual App and Tools

This repo contains a Streamlit app and utility scripts to:
- Explore and segment uploaded time series (ClaSP)
- Detect recurring states and their transition process (CLaP)
- Stream and analyze the Wikimedia RecentChanges event stream (SSE) with a live feature preview and optional streaming segmentation


## Quick Start

Recommended environment: Python 3.10 (best compatibility with the pinned stack).

```powershell
# Create and activate a clean env (optional but recommended)
conda create -n clap310 python=3.10 -y
conda activate clap310

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
python -m streamlit run streamlit_app.py
```

Open the URL shown in the terminal (typically http://localhost:8501).


## App Overview (Streamlit)

The app has three main areas:

- Segmentation (ClaSP)
	- Upload a `.txt` file (univariate or multivariate with numeric columns)
	- Choose automatic or fixed number of segments, window-size strategy, and validation
	- Plot the segmentation and view a segments table

- State Detection (CLaP)
	- Upload a `.txt` file and optionally supply change points (CPs)
	- Configure classifier and CV folds, then run CLaP to get state labels and the process graph
	- Plots: detected states and discovered process graph; table with state-labeled segments

- Streaming (Wikimedia SSE)
	- Connect to the RecentChanges SSE endpoint and buffer incoming events
	- Extract rich features per event:
		- `log_abs_delta = log1p(abs(new_len − old_len))`
		- `signed_delta = sign(new_len − old_len) * log1p(abs(delta))`
		- `inter_arrival_ms = time since previous event (ms)`
		- `revert_ind = 1 if revert-like tags or comment contains 'revert', else 0`
	- Preview the last N points of each feature inline via charts
	- Choose a numeric signal to process: any single feature or a fused signal
	- Run Streaming ClaSP on the buffered numeric stream to estimate the last change point


### Fused Signal (what it is)

A fused signal is a single numeric stream created by combining several standardized features with user-defined weights. By z-scoring continuous features (so they share a comparable scale) and applying weights, you create one 1D time series that captures multiple aspects of the event stream simultaneously. This can make streaming change-point detection more sensitive to combined patterns (e.g., large edit bursts with frequent reverts).

Formula (when z-scoring is enabled for continuous features):

```
fused = w_log_abs_delta * z(log_abs_delta)
			+ w_signed_delta  * z(signed_delta)
			+ w_inter_arrival * z(inter_arrival_ms)
			+ w_revert_ind    * revert_ind
```

You can adjust the weights in the UI to emphasize the behaviors you care about.


### Interpreter Info

In the sidebar (Environment expander), the app shows:
- Active interpreter path (sys.executable)
- Python version (sys.version)

This helps verify which environment Streamlit is using.


## Scripts

### Wikimedia SSE test

Standalone script to consume Wikimedia RecentChanges via SSE, map events to a numeric signal, and optionally run Streaming ClaSP.

```powershell
# Basic run with length-delta mapping
python scripts/wikimedia_sse_test.py --map length_delta --limit 5000 --window 2000 --warmup 1000 --jump 5 --log-cps --plot figures/wm_sse_test.png

# Try other mappings
python scripts/wikimedia_sse_test.py --map comment_len   --limit 8000 --plot figures/wm_comment_len.png
python scripts/wikimedia_sse_test.py --map is_bot        --limit 8000 --plot figures/wm_is_bot.png
```

Arguments:
- `--map`: `length_delta | comment_len | minor_edit | is_bot`
- `--limit`: number of events to buffer from SSE
- `--window`, `--warmup`, `--jump`, `--log-cps`: streaming ClaSP parameters
- `--plot`: optional PNG output path


## Environment Notes

- Python: 3.10 recommended
- If you see dependency resolution issues, start from a clean env and ensure `numpy==1.23.5` (compatible with `numba==0.56.x`) as pinned in requirements.
- Some features (e.g., streaming SSE, state detection classifiers) are imported only when used to keep the app snappy and avoid hard failures when optional packages are missing.


## Troubleshooting

- Streamlit won’t start:
	- Ensure you installed via `pip install -r requirements.txt` in the active environment
	- Confirm the interpreter shown in the app’s sidebar matches the one you expect

- SSE 403 Forbidden:
	- We send a descriptive `User-Agent`; if you fork, keep a valid `User-Agent` header for Wikimedia EventStreams

- No change points detected in Streaming:
	- Increase `--limit` / buffer size and window length
	- Try a different signal (e.g., `comment_len`) or the fused signal with weights that emphasize your target behavior


## Roadmap / Optional

- Kafka source (opt-in): add a source selector to stream from Kafka brokers and clusters with pluggable auth; reuse the same feature extractor and fused signal pipeline.
- Multivariate streaming fusion: configurable pre-fusion and optional channel voting.

