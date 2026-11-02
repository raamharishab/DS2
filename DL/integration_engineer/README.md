# DL Integration Engineer — Stage 02 Streamlit Control Tower Integration

## Overview

This module integrates the completed Deep Learning (Stage 02) artifacts and models into the operational Streamlit Control Tower dashboard, presenting DL model intelligence alongside the deployed Machine Learning (Stage 01) model.

---

## Architecture & Comparability

As documented in the audit (`DL/evaluation engineer/documentation/ml_vs_dl_comparability_audit.md`), the Stage 01 ML model and Stage 02 DL model solve complementary supply-chain problems at different granularities:

| Dimension | Stage 01 (ML — Deployed) | Stage 02 (DL — Integrated) |
| :--- | :--- | :--- |
| **Model** | Multi-class LightGBM (`v1.4.1`) | `TemporalAttentionLSTMForecaster` (Phase 3.4 Final) |
| **Task** | 3-Class Triage (`ON_TIME`, `AT_RISK`, `CRITICAL_DELAY`) | Macro throughput regression (7-day forward vessel calls + cargo tonnage) + Anchorage dwell regression |
| **Granularity** | Single container stay at TPS Surabaya | 2,065 global ports, daily sequences |
| **Input** | 13 pre-discharge arrival features | 30-day lookback window × 20 features (T-2 safe) |
| **Primary Metric** | Macro-F1 = 0.85 (20,879 validation stays) | $R^2 = 0.9815$, $\text{MAE} = 2.893$, $\text{RMSE} = 6.224$ (743,400 validation sequences) |

The models operate as **complementary layers** of supply chain intelligence:
1. **Terminal-level triage** (ML): Instant micro-level dwell risk classification for incoming containers.
2. **Global network context** (DL): Macro-level forecast of congestion, throughput, and anchorage dwell pressure across the broader global maritime network.

---

## Deliverables

1. **[`dl_inference.py`](file:///Users/raamharishab/Desktop/DS2/DL/integration_engineer/dl_inference.py)**:
   - Zero-overhead metadata bridge loading frozen parameters and metrics from committed DL artifacts:
     * `DL/ml engineer/artifacts/attention_lstm/checkpoint_metadata.json`
     * `DL/ml engineer/artifacts/attention_lstm/execution_metadata.json`
     * `DL/ml engineer/artifacts/dwell_forecasting/evaluation_metrics.json`
     * `DL/ml engineer/artifacts/attention_lstm/attention_diagnostics.json`
   - Exposes formatted accessors: `get_val_metrics()`, `get_training_stats()`, and attention profiles.

2. **Streamlit Control Tower Integration**:
   - **[`views/triage.py`](file:///Users/raamharishab/Desktop/DS2/ML/integration_engineer/control_tower/views/triage.py)**: Added `dl_comparison_panel()` rendering bento grids, side-by-side ML vs DL operational snapshots, architecture specifications, throughput and dwell forecast metrics, and temporal attention decay profiles. The existing ML scoring UI/UX is completely preserved.
   - **[`views/model.py`](file:///Users/raamharishab/Desktop/DS2/ML/integration_engineer/control_tower/views/model.py)**: Added dedicated **DL Model Evaluation** section displaying model parameters, validation metrics, dwell forecast performance vs naive persistence baselines, and temporal attention distribution.

3. **Validation & Test Suite**:
   - **[`tests/test_dl_integration.py`](file:///Users/raamharishab/Desktop/DS2/tests/test_dl_integration.py)**: 8 integration tests covering package structure, artifact loading, metric value assertions, attention weights, and UI view module imports.

---

## Verification & Execution

### Run Integration Tests
```bash
python3 -m pytest tests/test_dl_integration.py tests/test_control_tower.py tests/test_integration_engineer.py
```

### Launch Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```
Navigate to **Container triage** (`http://localhost:8501/triage`) to view container risk scoring with the DL Intelligence panel, or **Model intelligence** (`http://localhost:8501/model`) for the full ML and DL evaluation metrics.
