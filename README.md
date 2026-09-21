# SWIR-Project-yolo26

SWIR-domain adversarial robustness study on YOLO26. Companion code for the IVCL paper.

## What's here

- `robustness/` — attack and defense framework (SNUA-GO, ARIA, Combined, PGD, matched random controls; PGD-AT + SWIR adversarial training + robust checkpoint selector).
- `train.py`, `train_attack.py`, `train_swir_robust.py` — training entry points (standard, PGD-AT, D2/D3/D4/D5 SWIR-domain defenses).
- `pgd_robustness_grid_*.py` — PGD-20 sweep scripts for the leakage-safe and RASMD splits.
- `robustness/experiments/` — one script per milestone experiment (defense matrix, SNUA vs random control, ARIA vs random control, minimal §38 comparison).
- `aggregate_results_table3.py` — merges every result CSV into a single table.

## Results

- `aggregate_results_table2.csv` — original PGD-era table (single-run, test split).
- `aggregate_results_table3.csv` — merged table with the new SWIR-attack/defense matrix (mean ± std over ≥3 seeds), plus `n_seeds` / `selection_bias` / `source` columns.
- `defense_matrix_results.csv` — per-seed defense matrix, VAL split (development).
- `defense_matrix_checkpoint_rule.json` — frozen §13.3 checkpoint-selection rule per defense.
- `PROVENANCE_SNAPSHOT.json`, `ROBUSTNESS_AUDIT_M0.json` — provenance and framework audit.

## Not included in this repo

- **Datasets** (`NSLSR/`, `RASMD/`) — carry their own licenses; obtain from source.
- **Trained checkpoints** (`runs/train/*/weights/*.pt`) — regenerable via the training scripts.
- **Training/eval logs** — verbose, regenerable.

## Reproducing an evaluation

```
python run_patched.py                                   # applies the Ultralytics sandbox patch
python robustness/experiments/defense_matrix.py --split test
python aggregate_results_table3.py
```

`robustness/sandbox_patch.py` must be imported before Ultralytics — every entry-point script already does this.
