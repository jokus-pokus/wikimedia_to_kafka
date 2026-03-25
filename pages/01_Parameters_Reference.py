import streamlit as st
import pandas as pd

st.set_page_config(page_title="ClaSPy Parameters Reference", layout="wide")
st.title("ClaSPy Parameters Reference")

st.markdown("""
This page summarizes key parameters for the main ClaSPy components used in the app. Defaults reflect the library's current behavior and README guidance. Use this as a quick reference when tuning.
""")


def table(title: str, rows: list[dict]):
    st.subheader(title)
    df = pd.DataFrame(rows, columns=["Parameter", "Type", "Default", "Valid values", "Description"])  # type: ignore
    st.dataframe(df, use_container_width=True, hide_index=True)


# Binary ClaSP Segmentation
segmentation_params = [
    {"Parameter": "n_segments", "Type": "str | int", "Default": "\"learn\"",
     "Valid values": "\"learn\" or positive int", "Description": "Segments to produce; 'learn' infers via validation."},
    {"Parameter": "n_estimators", "Type": "int", "Default": "10",
     "Valid values": ">=1", "Description": "Number of ClaSP profiles in the ensemble."},
    {"Parameter": "window_size", "Type": "str | int", "Default": "\"suss\"",
     "Valid values": "\"suss\", \"fft\", \"acf\", or positive int", "Description": "Window size method or fixed size."},
    {"Parameter": "k_neighbours", "Type": "int", "Default": "3",
     "Valid values": ">=1", "Description": "k for k-NN subsequence neighbors."},
    {"Parameter": "distance", "Type": "str", "Default": "\"znormed_euclidean_distance\"",
     "Valid values": "znormed_euclidean_distance | cinvariant_euclidean_distance | euclidean_distance",
     "Description": "Distance metric for subsequence similarity."},
    {"Parameter": "score", "Type": "str", "Default": "\"roc_auc\"",
     "Valid values": "roc_auc or other supported", "Description": "Classification score for candidate CP quality."},
    {"Parameter": "early_stopping", "Type": "bool", "Default": "True",
     "Valid values": "True/False", "Description": "Stop searching when additional CPs seem unlikely."},
    {"Parameter": "validation", "Type": "str", "Default": "\"significance_test\"",
     "Valid values": "significance_test | score_threshold", "Description": "How to accept candidate CPs."},
    {"Parameter": "threshold", "Type": "str | float", "Default": "\"default\"",
     "Valid values": "'default' or numeric", "Description": "p-value (significance_test) or score (score_threshold)."},
    {"Parameter": "excl_radius", "Type": "int", "Default": "5",
     "Valid values": ">=0", "Description": "Exclusion radius factor around points (multiplied by window)."},
    {"Parameter": "n_jobs", "Type": "int", "Default": "-1",
     "Valid values": "-1 or >=1", "Description": "Parallel jobs for ClaSP computation."},
    {"Parameter": "random_state", "Type": "int", "Default": "2357",
     "Valid values": "any int", "Description": "Random seed for reproducibility."},
]

table("BinaryClaSPSegmentation (claspy.segmentation)", segmentation_params)


# Agglomerative CLaP State Detection
state_params = [
    {"Parameter": "window_size", "Type": "str | int", "Default": "\"suss\"",
     "Valid values": "\"suss\", \"fft\", \"acf\", or positive int", "Description": "Window size for classification."},
    {"Parameter": "classifier", "Type": "str", "Default": "\"rocket\"",
     "Valid values": "rocket | mrhydra | weasel | quant | rdst | proximityforest | freshprince | inception | dummy",
     "Description": "Classifier used by CLaP during merging."},
    {"Parameter": "n_splits", "Type": "int", "Default": "5",
     "Valid values": ">=2", "Description": "Cross-validation folds for CLaP classification."},
    {"Parameter": "sample_size", "Type": "int", "Default": "1000",
     "Valid values": ">=1", "Description": "Max samples per class for classifier training."},
    {"Parameter": "n_jobs", "Type": "int", "Default": "-1",
     "Valid values": "-1 or >=1", "Description": "Parallel jobs for supported classifiers."},
    {"Parameter": "random_state", "Type": "int", "Default": "2357",
     "Valid values": "any int", "Description": "Random seed for reproducibility."},
]

st.markdown("""
In addition, you can optionally supply precomputed change points to CLaP via `change_points`. If omitted, ClaSP segmentation is run internally.
""")

table("AgglomerativeCLaPDetection (claspy.state_detection)", state_params)


# Streaming ClaSP Segmentation (planned)
streaming_params = [
    {"Parameter": "n_timepoints", "Type": "int", "Default": "10000",
     "Valid values": ">=1", "Description": "Length of the sliding window for streaming computation."},
    {"Parameter": "n_warmup", "Type": "int", "Default": "10000",
     "Valid values": ">=0 and <= n_timepoints", "Description": "Initial points to warm up before detection."},
    {"Parameter": "window_size", "Type": "str | int", "Default": "\"suss\"",
     "Valid values": "\"suss\", \"fft\", \"acf\", or positive int", "Description": "Window size or method for ClaSS."},
    {"Parameter": "k_neighbours", "Type": "int", "Default": "3",
     "Valid values": ">=1", "Description": "k for streaming k-NN subsequence neighbors."},
    {"Parameter": "distance", "Type": "str", "Default": "\"znormed_euclidean_distance\"",
     "Valid values": "znormed_euclidean_distance | cinvariant_euclidean_distance | euclidean_distance",
     "Description": "Distance metric for streaming k-NN."},
    {"Parameter": "score", "Type": "str", "Default": "\"f1\"",
     "Valid values": "f1 | accuracy", "Description": "Score used by classification-based ClaSP."},
    {"Parameter": "jump", "Type": "int", "Default": "5",
     "Valid values": ">=1", "Description": "Step between consecutive detection attempts."},
    {"Parameter": "validation", "Type": "str", "Default": "\"significance_test\"",
     "Valid values": "significance_test | score_threshold", "Description": "Acceptance of candidate CPs."},
    {"Parameter": "threshold", "Type": "str | float", "Default": "\"default\"",
     "Valid values": "'default' or numeric", "Description": "p-value (significance) or score threshold."},
    {"Parameter": "log_cps", "Type": "bool", "Default": "False",
     "Valid values": "True/False", "Description": "Log detected CPs as a list of timestamps."},
    {"Parameter": "excl_radius", "Type": "int", "Default": "5",
     "Valid values": ">=0", "Description": "Exclusion radius factor (multiplied by window size)."},
]

table("StreamingClaSPSegmentation (claspy.streaming)", streaming_params)


st.markdown("""
Tips:
- Prefer automatic window size methods ("suss", "fft", "acf") before setting a fixed size.
- If you know an approximate number of segments in advance, set `n_segments` to a fixed value.
- For state detection, start with `classifier='rocket'` and default `n_splits`, then tune `sample_size` for speed.
- In streaming scenarios, be mindful that `n_warmup` must complete before predictions are meaningful.
""")
