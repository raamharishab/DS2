"""Unit & Integration Test Suite for Deep Learning (DL) Stage 02 Integration.

Validates:
1. DL package resolution (DL/__init__.py, DL/integration_engineer/__init__.py).
2. DL inference artifact loading from frozen checkpoints and metadata.
3. Metric integrity (R^2, MAE, RMSE, Huber loss) against Stage 02 closure benchmarks.
4. Temporal Attention profile integrity and Bahdanau attention weight distribution.
5. Dwell forecasting metrics integrity.
6. Control tower view integration (triage.py dl_comparison_panel, model.py DL evaluation).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from DL.integration_engineer import dl_inference


def test_dl_package_structure():
    """Verify that DL packages and modules are importable."""
    import DL
    from DL import integration_engineer
    assert hasattr(dl_inference, "get_val_metrics")
    assert hasattr(dl_inference, "get_training_stats")
    assert hasattr(dl_inference, "DL_ARCHITECTURE")
    assert hasattr(dl_inference, "DL_VAL_METRICS")
    assert hasattr(dl_inference, "DL_DWELL_METRICS")
    assert hasattr(dl_inference, "DL_ATTENTION_PROFILE")


def test_checkpoint_metadata_loading():
    """Verify checkpoint metadata can be loaded from committed artifacts."""
    meta = dl_inference.load_checkpoint_metadata()
    assert meta is not None, "checkpoint_metadata.json should be readable"
    assert "val_metrics" in meta
    vm = meta["val_metrics"]
    assert "r2" in vm
    assert "mae" in vm
    assert "rmse" in vm
    assert vm["r2"] > 0.95, "R^2 must exceed 0.95 per Stage 02 target"


def test_execution_metadata_loading():
    """Verify execution metadata contains training duration and throughput."""
    exec_meta = dl_inference.load_execution_metadata()
    assert exec_meta is not None, "execution_metadata.json should be readable"
    assert "trainable_parameters" in exec_meta
    assert exec_meta["trainable_parameters"] == 57473
    assert "training_duration_sec" in exec_meta
    assert "inference_throughput_seq_per_sec" in exec_meta


def test_dwell_forecasting_metrics_loading():
    """Verify dwell forecasting metrics load correctly."""
    dwell = dl_inference.load_dwell_metrics()
    assert dwell is not None, "evaluation_metrics.json should be readable"
    assert "validation_evaluation_2021" in dwell
    assert "test_held_out_evaluation_2022" in dwell
    val_m = dwell["validation_evaluation_2021"]["model_metrics_h1"]
    test_m = dwell["test_held_out_evaluation_2022"]["model_metrics_h1"]
    assert val_m["r2"] > 0.70
    assert test_m["r2"] > 0.60
    assert "mae_reduction_pct" in dwell["validation_evaluation_2021"]
    assert "mae_reduction_pct" in dwell["test_held_out_evaluation_2022"]


def test_get_val_metrics_returns_expected_values():
    """Verify get_val_metrics returns formatted dictionary."""
    metrics = dl_inference.get_val_metrics()
    assert "R²" in metrics
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "Huber" in metrics
    assert "Sequences" in metrics
    assert metrics["R²"] == 0.9815
    assert metrics["MAE"] == 2.893
    assert metrics["RMSE"] == 6.224


def test_get_training_stats_returns_expected_values():
    """Verify get_training_stats returns parameter count and timings."""
    stats = dl_inference.get_training_stats()
    assert "Parameters" in stats
    assert "Training time" in stats
    assert "Inference" in stats
    assert stats["Parameters"] == "57,473"


def test_temporal_attention_profile_properties():
    """Verify temporal attention profile contains lookback window distribution."""
    ap = dl_inference.DL_ATTENTION_PROFILE
    assert "Most attended" in ap
    assert "Recent 7 days (t-2 … t-8)" in ap
    assert "61.6%" in ap["Recent 7 days (t-2 … t-8)"]


def test_views_dl_integration():
    """Verify triage and model view modules load dl_inference without conflict."""
    from ML.integration_engineer.control_tower.views import triage, model
    assert hasattr(triage, "dl_comparison_panel")
    assert callable(triage.dl_comparison_panel)
    assert hasattr(model, "render")
    assert callable(model.render)
