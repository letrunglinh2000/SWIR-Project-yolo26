"""
Build the merged aggregate results table (Table 3): the PGD-era results
(aggregate_results_table2.csv) plus the Milestone-5 SWIR attack/defense
matrix (defense_matrix_results*.csv), in ONE schema.

Why a new table rather than appending to table 2:
  * Table 2 is single-run (no seeds) and covers {clean, pgd20} only.
  * The defense matrix is >=3 seeds per stochastic attack, so every new row
    carries mean +- std over seeds and n_seeds; old rows get n_seeds=1 and
    blank std. Mixing them without those columns would silently present a
    1-run number and a 3-seed mean as the same kind of measurement.
  * `eval_split` is preserved verbatim: the defense matrix's development
    numbers are VAL, the final-report pass is TEST. D2/D3/D5's robust_best.pt
    was SELECTED on val, so its val rows are selection-biased -- only the
    test rows are comparable to table 2's test numbers. `selection_bias`
    marks exactly which rows that applies to.

Usage: python aggregate_results_table3.py
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
METRICS = ["precision", "recall", "mAP50", "mAP50-95"]

FIELDS = (
    ["dataset", "split_variant", "model", "training", "checkpoint", "eval_split",
     "attack", "n_seeds"]
    + [c for m in METRICS for c in (m, f"{m}_std")]
    + ["selection_bias", "source"]
)

# training-recipe label + the checkpoint rule frozen in
# defense_matrix_checkpoint_rule.json
DEFENSE_META = {
    "D0_standard":        ("standard",            "best"),
    "D1_pgd_at":          ("pgd_at",              "last"),
    "D2_random_swir_aug": ("random_swir_aug",     "robust_best"),
    "D3_snua_at":         ("snua_at",             "robust_best"),
    "D5_mixed_swir_at":   ("mixed_swir_at",       "robust_best"),
}
# robust_best.pt is chosen by a val-split rule (S = 0.5*mAP_clean +
# 0.5*mAP_robust under SNUA-GO weak/5-step), so its VAL rows are optimistic.
VAL_SELECTED = {"best", "robust_best"}


def _f(v):
    return float(v) if v not in ("", None) else None


def load_table2():
    rows = []
    src = REPO / "aggregate_results_table2.csv"
    if not src.exists():
        print(f"[warn] {src.name} not found -- table 3 will contain the new results only")
        return rows
    with open(src, newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("dataset"):
                continue
            rows.append({
                **{k: r[k] for k in ["dataset", "split_variant", "model", "training",
                                     "checkpoint", "eval_split", "attack"]},
                "n_seeds": 1,
                **{m: _f(r[m]) for m in METRICS},
                **{f"{m}_std": "" for m in METRICS},
                "selection_bias": "",
                "source": "aggregate_results_table2.csv",
            })
    return rows


def load_defense_matrix(path: Path):
    """Collapse the per-seed matrix into mean +- std rows."""
    if not path.exists():
        print(f"[skip] {path.name} not present")
        return []
    groups = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            split = r.get("split") or "val"   # pre---split runs had no column
            groups[(r["checkpoint"], r["condition"], split)].append(r)

    rows = []
    for (ckpt, cond, split), recs in groups.items():
        training, ckpt_file = DEFENSE_META[ckpt]
        agg = {}
        for m in METRICS:
            vals = [_f(x[m]) for x in recs if _f(x[m]) is not None]
            agg[m] = statistics.mean(vals) if vals else None
            agg[f"{m}_std"] = statistics.stdev(vals) if len(vals) > 1 else ""
        bias = ""
        if split == "val" and ckpt_file in VAL_SELECTED:
            bias = "checkpoint selected on this split"
        rows.append({
            "dataset": "NSLSR", "split_variant": "leakage-safe (segment)", "model": "nano",
            "training": training, "checkpoint": ckpt_file, "eval_split": split,
            "attack": cond, "n_seeds": len(recs), **agg,
            "selection_bias": bias, "source": path.name,
        })
    return rows


def load_all_evals(path: Path):
    """Read run_missing_evals.py's streaming CSV. Rows there are already
    per-seed; group and collapse to mean/std in the same shape as the D-matrix."""
    if not path.exists():
        print(f"[skip] {path.name} not present")
        return []
    groups = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["dataset"], r["split_variant"], r["model"], r["training"],
                   r["checkpoint"], r["eval_split"], r["attack"])
            groups[key].append(r)
    rows = []
    for key, recs in groups.items():
        agg = {}
        for m in METRICS:
            vals = [_f(x[m]) for x in recs if _f(x[m]) is not None]
            agg[m] = statistics.mean(vals) if vals else None
            agg[f"{m}_std"] = statistics.stdev(vals) if len(vals) > 1 else ""
        (ds, sv, model, training, ckpt, eval_split, attack) = key
        bias = ""
        if eval_split == "val" and ckpt in {"best", "robust_best"}:
            bias = "checkpoint selected on this split"
        rows.append({
            "dataset": ds, "split_variant": sv, "model": model, "training": training,
            "checkpoint": ckpt, "eval_split": eval_split, "attack": attack,
            "n_seeds": len(recs), **agg,
            "selection_bias": bias, "source": path.name,
        })
    return rows


def sort_key(r):
    ds = {"NSLSR": 0, "RASMD": 1}.get(r["dataset"], 2)
    sp = {"official (leaky)": 0, "leakage-safe (segment)": 1, "stratified (own)": 2}.get(r["split_variant"], 3)
    tr = {"standard": 0, "adversarial": 1, "pgd_at": 2, "random_swir_aug": 3,
          "snua_at": 4, "mixed_swir_at": 5}.get(r["training"], 9)
    at = {"clean": 0, "pgd20": 1, "snua_go_weak": 2, "snua_go_medium": 3,
          "aria_medium": 4, "combined_medium": 5}.get(r["attack"], 9)
    return (ds, sp, r["model"], tr, r["checkpoint"], r["eval_split"], at)


def main():
    rows = load_table2()
    for name in ["defense_matrix_results.csv", "defense_matrix_results_test.csv"]:
        rows += load_defense_matrix(REPO / name)
    rows += load_all_evals(REPO / "all_evals_results.csv")

    # dedupe: (dataset, split_variant, model, training, checkpoint, eval_split, attack)
    # -- if the all_evals CSV re-measured a cell the D-matrix also has, keep
    # whichever row has more seeds (typically the newer all_evals row).
    dedup = {}
    for r in rows:
        key = (r["dataset"], r["split_variant"], r["model"], r["training"],
               r["checkpoint"], r["eval_split"], r["attack"])
        existing = dedup.get(key)
        if existing is None or (r.get("n_seeds", 1) or 1) > (existing.get("n_seeds", 1) or 1):
            dedup[key] = r
    rows = list(dedup.values())
    rows.sort(key=sort_key)

    out = REPO / "aggregate_results_table3.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in FIELDS})
    print(f"wrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
