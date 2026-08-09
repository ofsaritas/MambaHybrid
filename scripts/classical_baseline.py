"""Random Forest and XGBoost baselines on preprocessed splits.

Usage:
    python scripts/classical_baseline.py
    python scripts/classical_baseline.py CICIDS2017
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "CICIDS2017": "dataset/CICIDS2017_smote3",
    "UNSW-NB15": "dataset/UNSW-NB15",
}
SEEDS = [42, 1, 2]
OUT_PATH = ROOT / "outputs" / "classical_baseline_results.json"


def load(ds_path):
    path = ROOT / ds_path
    Xtr = np.load(path / "X_train.npy")
    ytr = np.load(path / "y_train.npy")
    Xte = np.load(path / "X_test.npy")
    yte = np.load(path / "y_test.npy")
    return Xtr, ytr, Xte, yte


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(OUT_PATH) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    only_ds = sys.argv[1] if len(sys.argv) > 1 else None
    for ds_name, ds_path in DATASETS.items():
        if only_ds and ds_name != only_ds:
            continue
        print(f"=== {ds_name} ===", flush=True)
        Xtr, ytr, Xte, yte = load(ds_path)
        results[ds_name] = {"RF": [], "XGBoost": []}
        for seed in SEEDS:
            t0 = time.time()
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=None, n_jobs=-1,
                random_state=seed, class_weight="balanced_subsample",
            )
            rf.fit(Xtr, ytr)
            pred = rf.predict(Xte)
            m = metrics(yte, pred)
            m["seed"] = seed
            m["train_time_s"] = time.time() - t0
            results[ds_name]["RF"].append(m)
            print(f"  RF seed={seed}: {m}", flush=True)

            t0 = time.time()
            xgb = XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                tree_method="hist",
                objective="multi:softprob",
                num_class=int(np.max(ytr)) + 1,
                random_state=seed, n_jobs=-1, eval_metric="mlogloss",
            )
            xgb.fit(Xtr, ytr)
            pred = xgb.predict(Xte)
            m = metrics(yte, pred)
            m["seed"] = seed
            m["train_time_s"] = time.time() - t0
            results[ds_name]["XGBoost"].append(m)
            print(f"  XGB seed={seed}: {m}", flush=True)

        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print("DONE")


if __name__ == "__main__":
    main()
