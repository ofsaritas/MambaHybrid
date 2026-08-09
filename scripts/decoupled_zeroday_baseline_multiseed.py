"""Decoupled XGBoost + Isolation Forest zero-day baseline (multi-seed).

Usage:
    python scripts/decoupled_zeroday_baseline_multiseed.py
    python scripts/decoupled_zeroday_baseline_multiseed.py CICIDS2017_volumetric
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]

CONFIGS = {
    "CICIDS2017_stealthy": {
        "path": "dataset/CICIDS2017_smote3",
        "benign_label": "BENIGN",
        "holdout_labels": ["Bot", "PortScan"],
        "seeds": [42, 1, 2],
    },
    "CICIDS2017_volumetric": {
        "path": "dataset/CICIDS2017_smote3",
        "benign_label": "BENIGN",
        "holdout_labels": ["DDoS", "DoS Hulk"],
        "seeds": [42, 1, 2],
    },
    "UNSW-NB15": {
        "path": "dataset/UNSW-NB15",
        "benign_label": "Normal",
        "holdout_labels": ["Worms", "Shellcode"],
        "seeds": [42, 1, 2],
    },
}


def label_ids(le, names):
    classes = list(le.classes_)
    return [classes.index(n) for n in names]


def run_one_seed(cfg, seed, Xtr, ytr, Xte, yte, benign_id, holdout_ids, le):
    keep_mask = ~np.isin(ytr, list(holdout_ids))
    Xtr_clf, ytr_clf = Xtr[keep_mask], ytr[keep_mask]
    remaining = sorted(set(ytr_clf.tolist()))
    remap = {old: new for new, old in enumerate(remaining)}
    ytr_clf_remapped = np.array([remap[v] for v in ytr_clf])

    clf_test_mask = ~np.isin(yte, list(holdout_ids))
    Xte_clf, yte_clf = Xte[clf_test_mask], yte[clf_test_mask]
    yte_clf_remapped = np.array([remap[v] for v in yte_clf])

    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        tree_method="hist",
        objective="multi:softprob",
        num_class=len(remaining),
        random_state=seed, n_jobs=-1, eval_metric="mlogloss",
    )
    xgb.fit(Xtr_clf, ytr_clf_remapped)
    pred = xgb.predict(Xte_clf)
    clf_metrics = {
        "accuracy": float(accuracy_score(yte_clf_remapped, pred)),
        "weighted_f1": float(f1_score(yte_clf_remapped, pred, average="weighted")),
        "macro_f1": float(f1_score(yte_clf_remapped, pred, average="macro")),
        "mcc": float(matthews_corrcoef(yte_clf_remapped, pred)),
        "train_time_s": time.time() - t0,
    }

    X_benign_train = Xtr[ytr == benign_id]
    iso = IsolationForest(
        n_estimators=200, max_samples="auto", contamination="auto",
        random_state=seed, n_jobs=-1,
    )
    iso.fit(X_benign_train)

    X_benign_test = Xte[yte == benign_id]
    anomaly_score_benign = -iso.score_samples(X_benign_test)

    holdout_name_ids = label_ids(le, cfg["holdout_labels"])
    zeroday_results = {}
    for holdout_name, holdout_id in zip(cfg["holdout_labels"], holdout_name_ids):
        mask = yte == holdout_id
        X_holdout = Xte[mask]
        if len(X_holdout) == 0:
            zeroday_results[holdout_name] = {"error": "no test samples"}
            continue
        anomaly_score_holdout = -iso.score_samples(X_holdout)
        y_true = np.concatenate([
            np.zeros(len(anomaly_score_benign)),
            np.ones(len(anomaly_score_holdout)),
        ])
        y_score = np.concatenate([anomaly_score_benign, anomaly_score_holdout])
        zeroday_results[holdout_name] = {
            "auroc": float(roc_auc_score(y_true, y_score)),
            "pr_auc": float(average_precision_score(y_true, y_score)),
            "n_holdout_test_samples": int(len(X_holdout)),
        }
    return clf_metrics, zeroday_results


def run_dataset(name, cfg):
    print(f"=== {name} ===", flush=True)
    data_path = ROOT / cfg["path"]
    pp = joblib.load(data_path / "preprocess.joblib")
    le = pp["label_encoder"]
    benign_id = label_ids(le, [cfg["benign_label"]])[0]
    holdout_ids = set(label_ids(le, cfg["holdout_labels"]))

    Xtr = np.load(data_path / "X_train.npy")
    ytr = np.load(data_path / "y_train.npy")
    Xte = np.load(data_path / "X_test.npy")
    yte = np.load(data_path / "y_test.npy")

    per_seed_clf = []
    per_seed_zeroday = {h: [] for h in cfg["holdout_labels"]}
    for seed in cfg["seeds"]:
        clf_m, zd = run_one_seed(
            cfg, seed, Xtr, ytr, Xte, yte, benign_id, holdout_ids, le,
        )
        clf_m["seed"] = seed
        per_seed_clf.append(clf_m)
        for h in cfg["holdout_labels"]:
            per_seed_zeroday[h].append(zd[h])
        print(f"  seed={seed}: clf={clf_m}", flush=True)
        for h in cfg["holdout_labels"]:
            print(f"    zero-day {h}: {zd[h]}", flush=True)

    def mean_std(vals):
        arr = np.array(vals, dtype=float)
        return float(arr.mean()), float(arr.std())

    summary_clf = {}
    for key in ["accuracy", "weighted_f1", "macro_f1", "mcc"]:
        m, s = mean_std([r[key] for r in per_seed_clf])
        summary_clf[key] = {"mean": m, "std": s}

    summary_zeroday = {}
    for h in cfg["holdout_labels"]:
        aurocs = [r["auroc"] for r in per_seed_zeroday[h] if "auroc" in r]
        prs = [r["pr_auc"] for r in per_seed_zeroday[h] if "pr_auc" in r]
        if aurocs:
            am, astd = mean_std(aurocs)
            pm, pstd = mean_std(prs)
            summary_zeroday[h] = {
                "auroc_mean": am, "auroc_std": astd,
                "pr_auc_mean": pm, "pr_auc_std": pstd,
            }

    return {
        "n_seeds": len(cfg["seeds"]),
        "seeds": cfg["seeds"],
        "classification_summary": summary_clf,
        "zeroday_summary": summary_zeroday,
        "per_seed_classification": per_seed_clf,
        "per_seed_zeroday": per_seed_zeroday,
    }


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    results = {}
    out_path = ROOT / "outputs" / "decoupled_zeroday_baseline_multiseed_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for name, cfg in CONFIGS.items():
        if only and name != only:
            continue
        results[name] = run_dataset(name, cfg)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
