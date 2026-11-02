"""DL Model Intelligence Module — Integration bridge for the Streamlit Control Tower.

Loads committed DL model metadata, validation metrics, and attention diagnostics
from frozen artifacts without re-running inference. All numbers come from files
committed by the DL workstream and are NOT recomputed at dashboard load time.

Model: TemporalAttentionLSTMForecaster (Phase 3.4 Final)
Task:  Global port-level throughput & anchorage dwell regression
       (Complementary signal to per-container ML classification)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

# Paths — resolve relative to this file so the app works from any CWD
_DL_ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / "DL" / "ml engineer" / "artifacts"
_ATTENTION_LSTM_DIR = _DL_ARTIFACT_ROOT / "attention_lstm"
_DWELL_FORECAST_DIR = _DL_ARTIFACT_ROOT / "dwell_forecasting"
_EVAL_DIR = Path(__file__).resolve().parents[2] / "DL" / "evaluation engineer" / "artifacts" / "metrics"

# ── Architecture constants (from training_config.json) ─────────────────────
DL_MODEL_NAME = "TemporalAttentionLSTM · Phase 3.4 Final"
DL_MODEL_TYPE = "TemporalAttentionLSTMForecaster"
DL_ARCHITECTURE = {
    "Input window": "30 days × 20 features",
    "LSTM layers": "2-layer bidirectional encoder (h=64)",
    "Attention": "Additive Bahdanau temporal attention (dim=32)",
    "Output": "Residual regression head → predicted throughput",
    "Training data": "3,696,350 sequences · 2,065 global ports",
    "Val sequences": "743,400",
    "Parameters": "57,473 trainable",
}

# ── Validated metrics from checkpoint_metadata.json ────────────────────────
DL_VAL_METRICS = {
    "R²": 0.9815,
    "MAE": 2.893,
    "RMSE": 6.224,
    "Huber loss": 2.443,
    "Mean target (vessel calls/7d)": 17.43,
    "Mean prediction": 17.74,
}

# ── Dwell forecasting metrics (evaluation_metrics.json Phase 3.6) ──────────
DL_DWELL_METRICS = {
    "Task": "Anchorage waiting-time (hours)",
    "MAE (validation)": 5.74,
    "RMSE (validation)": 10.86,
    "R² (validation)": 0.719,
    "MAE reduction vs naive": "2.2%",
    "MAE (test 2022)": 4.08,
    "RMSE (test 2022)": 6.40,
    "R² (test 2022)": 0.610,
    "MAE reduction (test) vs naive": "7.0%",
}

# ── Attention profile — top-weighted timesteps ─────────────────────────────
DL_ATTENTION_PROFILE = {
    "Most attended": "t-2 (most recent, 14.4% weight)",
    "Second": "t-3 (11.5%)",
    "Third": "t-4 (8.0%)",
    "Recency bias": "Strong — final 7 days carry ~75% of total attention mass",
    "Earliest 14 days": "<10% combined attention (long-horizon memory suppressed)",
}


@st.cache_data(show_spinner=False)
def load_checkpoint_metadata() -> Optional[Dict[str, Any]]:
    """Load checkpoint metadata from committed DL artifact (cached)."""
    p = _ATTENTION_LSTM_DIR / "checkpoint_metadata.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_execution_metadata() -> Optional[Dict[str, Any]]:
    """Load execution metadata (training duration, throughput)."""
    p = _ATTENTION_LSTM_DIR / "execution_metadata.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_dwell_metrics() -> Optional[Dict[str, Any]]:
    """Load dwell forecasting evaluation metrics."""
    p = _DWELL_FORECAST_DIR / "evaluation_metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def get_val_metrics() -> Dict[str, Any]:
    """Return validated metrics dict, preferring live artifact data."""
    meta = load_checkpoint_metadata()
    if meta and "val_metrics" in meta:
        m = meta["val_metrics"]
        return {
            "R²": round(m.get("r2", DL_VAL_METRICS["R²"]), 4),
            "MAE": round(m.get("mae", DL_VAL_METRICS["MAE"]), 3),
            "RMSE": round(m.get("rmse", DL_VAL_METRICS["RMSE"]), 3),
            "Huber": round(m.get("huber_loss", DL_VAL_METRICS["Huber loss"]), 3),
            "Sequences": f"{int(m.get('count', 743_400)):,}",
        }
    return {
        "R²": DL_VAL_METRICS["R²"],
        "MAE": DL_VAL_METRICS["MAE"],
        "RMSE": DL_VAL_METRICS["RMSE"],
        "Huber": DL_VAL_METRICS["Huber loss"],
        "Sequences": "743,400",
    }


def get_training_stats() -> Dict[str, Any]:
    """Return training stats from execution metadata."""
    exec_meta = load_execution_metadata()
    if exec_meta:
        return {
            "Parameters": f"{exec_meta.get('trainable_parameters', 57473):,}",
            "Training time": f"{exec_meta.get('training_duration_sec', 840.7):.0f}s",
            "Inference": f"{exec_meta.get('inference_throughput_seq_per_sec', 32299):,.0f} seq/s",
        }
    return {
        "Parameters": "57,473",
        "Training time": "841s",
        "Inference": "32,299 seq/s",
    }
