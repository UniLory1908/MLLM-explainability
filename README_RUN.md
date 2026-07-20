# Run And Environment Notes

This repository contains the shareable code and lightweight documentation for the
MLLM explainability work. Heavy local artifacts are intentionally not committed.

## Environment

Install the main project dependencies with:

```bash
pip install -r requirements.txt
```

The local workstation used for the latest dashboard/statistical archive also
has CUDA-enabled PyTorch. If installing CUDA wheels manually, follow the PyTorch
instructions for the target CUDA version before running Qwen/TAM inference.

## Dashboard

The dashboard reads already generated TAM metadata, raw maps and precomputed
SQLite metrics. Starting the dashboard does not rerun model inference.

```bash
python -m scripts.dashboard.app --host 127.0.0.1 --port 5050
```

Default local URL:

```text
http://127.0.0.1:5050/
```

The dashboard expects the local derived database at:

```text
outputs/dashboard_index/tam_index.sqlite
```

and canonical TAM experiment data at:

```text
outputs/prompt_sensitivity/
```

These heavy local outputs are not committed to GitHub.

## Statistical Archive Export

To export the already computed dashboard/statistical tables as CSV/Parquet:

```bash
python -m scripts.dashboard.export_statistical_archive --archive-name final_csv_archive_20260629 --dashboard-base-url http://127.0.0.1:5050 --overwrite
```

This command exports tables from the existing dashboard DB and derived metrics.
It does not launch Qwen/TAM inference and does not recompute raw TAM maps.

The complete CSV archive prepared for Drive is documented in:

```text
docs/CSV_ARCHIVE_MANIFEST.md
```

## V6 Analysis

The v6 horizontal analysis uses the exported archive and dashboard DB:

```bash
python scripts/analysis/build_v6_horizontal_analysis.py
```

This builds dataset-level summaries, prompt fingerprints, diagnostic rankings
and dashboard-ready analysis views. It is a repeated-measures analysis of
TAM-derived attribution behavior, not a causal faithfulness benchmark.
