"""Evaluate MambaHybrid classification and zero-day detection.

Usage:
    python multi/evaluate_hybrid.py \
        --config multi/configs/hybrid_cicids2017_smote_v2.yaml \
        --run_dir weights/CICIDS2017/mamba_hybrid_v2/seed_42
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from multi.model import MambaHybrid
from src.data.preprocess import load_arrays, prepare_dataset
from src.evaluation.metrics import cm_report, compute_metrics
from src.utils.config import load_config


def load_model(run_dir, cfg, device='cpu'):
    ck = torch.load(
        run_dir / 'checkpoints/best.pt',
        map_location=device,
        weights_only=True,
    )
    mc = cfg.get('model', {})
    hcfg = cfg.get('hybrid', {})
    model = MambaHybrid(
        n_features=ck['n_features'],
        n_classes=ck['n_classes'],
        d_model=mc.get('d_model', 128),
        depth=mc.get('depth', 4),
        d_state=mc.get('d_state', 16),
        d_conv=mc.get('d_conv', 4),
        expansion=mc.get('expansion', 2),
        dropout=mc.get('dropout', 0.1),
        decoder_depth=hcfg.get('decoder_depth', 2),
    )
    model.load_state_dict(ck['model'])
    model.to(device)
    model.eval()
    return model, ck


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--run_dir', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run = Path(args.run_dir)
    hcfg = cfg.get('hybrid', {})
    prepare_dataset(cfg)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ck = load_model(run, cfg, device)
    prep = joblib.load(Path(cfg['dataset']['processed_dir']) / 'preprocess.joblib')

    classes_list_all = list(prep['label_encoder'].classes_)
    classes_list = ck['classes']
    benign_idx = ck['benign_idx']
    holdout_indices = ck.get('holdout_indices', [])

    n_classes_all = len(classes_list_all)
    kept_idx = sorted(set(range(n_classes_all)) - set(holdout_indices))
    label_remap = {old: new for new, old in enumerate(kept_idx)}

    bs = cfg['training']['batch_size']

    for d in ['metrics', 'predictions', 'figures']:
        (run / d).mkdir(parents=True, exist_ok=True)

    X, y = load_arrays(cfg['dataset']['processed_dir'], 'test')

    all_logits = []
    all_recon_err = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i + bs], dtype=torch.float32).to(device)
            logits, recon = model(xb, return_recon=True)
            all_logits.append(logits.cpu().numpy())
            all_recon_err.append(((xb - recon) ** 2).mean(dim=1).cpu().numpy())

    logits = np.concatenate(all_logits)
    probs = softmax(logits, axis=1)
    recon_err = np.concatenate(all_recon_err)

    if holdout_indices:
        known_mask = ~np.isin(y, holdout_indices)
        y_known = np.vectorize(label_remap.__getitem__)(y[known_mask])
        probs_known = probs[known_mask]
    else:
        y_known = y
        probs_known = probs

    m = compute_metrics(y_known, probs_known)
    cm, rep = cm_report(y_known, probs_known, classes_list)
    (run / 'metrics/test_metrics.json').write_text(
        json.dumps(m, indent=2), encoding='utf-8')
    (run / 'metrics/classification_report.json').write_text(
        json.dumps(rep, indent=2), encoding='utf-8')
    (run / 'metrics/confusion_matrix.json').write_text(
        json.dumps(cm, indent=2), encoding='utf-8')

    anomaly_labels = (y != benign_idx).astype(int)
    try:
        anomaly_auroc = float(roc_auc_score(anomaly_labels, recon_err))
        anomaly_prauc = float(average_precision_score(anomaly_labels, recon_err))
        fpr, tpr, _ = roc_curve(anomaly_labels, recon_err)
    except Exception:
        anomaly_auroc = anomaly_prauc = 0.0
        fpr = tpr = np.array([0.0, 1.0])

    thr_file = run / 'metrics/anomaly_threshold.json'
    threshold = None
    if thr_file.exists():
        threshold = json.loads(thr_file.read_text())['threshold']
    if threshold is None:
        threshold = float(np.percentile(
            recon_err[y == benign_idx],
            hcfg.get('anomaly_threshold_percentile', 95),
        ))

    zeroday_classes = hcfg.get('zeroday_holdout_classes', [])
    zd_results = {}
    for zd_cls in zeroday_classes:
        if zd_cls not in classes_list_all:
            continue
        zd_idx = classes_list_all.index(zd_cls)
        mask = (y == benign_idx) | (y == zd_idx)
        if mask.sum() < 2:
            continue
        zd_binary = (y[mask] == zd_idx).astype(int)
        zd_scores = recon_err[mask]
        try:
            zd_auroc = float(roc_auc_score(zd_binary, zd_scores))
            zd_prauc = float(average_precision_score(zd_binary, zd_scores))
        except Exception:
            zd_auroc = zd_prauc = 0.0
        attack_err = recon_err[y == zd_idx]
        n_detected = int((attack_err > threshold).sum())
        n_total = len(attack_err)
        zd_results[zd_cls] = {
            'auroc': zd_auroc,
            'pr_auc': zd_prauc,
            'detected_above_threshold': n_detected,
            'total_samples': n_total,
            'detection_rate': n_detected / n_total if n_total > 0 else 0.0,
            'mean_recon_err': float(attack_err.mean()),
        }

    per_class = {}
    for new_i, cls in enumerate(classes_list):
        orig_i = kept_idx[new_i]
        mask = (y == orig_i)
        if mask.sum() == 0:
            continue
        err = recon_err[mask]
        per_class[cls] = {
            'n': int(mask.sum()),
            'mean_recon_err': float(err.mean()),
            'median_recon_err': float(np.median(err)),
            'frac_above_thr': float((err > threshold).mean()),
        }

    for zd_cls in zeroday_classes:
        if zd_cls not in classes_list_all:
            continue
        zd_orig_i = classes_list_all.index(zd_cls)
        mask = (y == zd_orig_i)
        if mask.sum() == 0:
            continue
        err = recon_err[mask]
        per_class[f'{zd_cls} [ZERODAY]'] = {
            'n': int(mask.sum()),
            'mean_recon_err': float(err.mean()),
            'median_recon_err': float(np.median(err)),
            'frac_above_thr': float((err > threshold).mean()),
        }

    anomaly_results = {
        'anomaly_auroc_all_attacks': anomaly_auroc,
        'anomaly_pr_auc_all_attacks': anomaly_prauc,
        'threshold_used': threshold,
        'recon_err_benign_mean': float(recon_err[y == benign_idx].mean()),
        'recon_err_attack_mean': float(recon_err[y != benign_idx].mean()),
        'zeroday_holdout': zd_results,
        'per_class_stats': per_class,
    }
    (run / 'metrics/anomaly_results.json').write_text(
        json.dumps(anomaly_results, indent=2), encoding='utf-8')

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f'AUROC = {anomaly_auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Anomaly Detection ROC\n(All attacks vs BENIGN)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(run / 'figures/anomaly_roc.png', dpi=200)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.hist(recon_err[y == benign_idx], bins=150, alpha=0.6,
             label='BENIGN', density=True, color='steelblue')
    plt.hist(recon_err[y != benign_idx], bins=150, alpha=0.6,
             label='Attack', density=True, color='tomato')
    plt.axvline(threshold, color='black', linestyle='--', lw=1.5,
                label=f'Threshold = {threshold:.4f}')
    plt.xlabel('Reconstruction Error (MSE per feature)')
    plt.ylabel('Density')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(run / 'figures/recon_error_distribution.png', dpi=200)
    plt.close()

    cls_names = list(per_class.keys())
    cls_errors = [per_class[c]['mean_recon_err'] for c in cls_names]
    benign_cls_name = classes_list_all[benign_idx]
    colors = [
        'steelblue' if c == benign_cls_name else
        ('gold' if '[ZERODAY]' in c else 'tomato')
        for c in cls_names
    ]
    plt.figure(figsize=(max(8, len(cls_names) * 0.7), 5))
    plt.bar(range(len(cls_names)), cls_errors, color=colors)
    plt.axhline(threshold, color='black', linestyle='--', lw=1.5,
                label=f'Threshold = {threshold:.4f}')
    plt.xticks(range(len(cls_names)), cls_names, rotation=45, ha='right', fontsize=8)
    plt.ylabel('Mean Reconstruction Error (MSE)')
    plt.title('Per-Class Reconstruction Error')
    plt.legend()
    plt.tight_layout()
    plt.savefig(run / 'figures/per_class_recon_error.png', dpi=200)
    plt.close()

    print(f'\n{"=" * 55}')
    print(f'Classification  macro_F1 : {m["macro_f1"]:.4f}')
    print(f'Classification  MCC      : {m["mcc"]:.4f}')
    print(f'Anomaly AUROC (all attacks vs BENIGN): {anomaly_auroc:.4f}')
    print(f'Anomaly PR-AUC                       : {anomaly_prauc:.4f}')
    if zd_results:
        print('Zero-day simulation:')
        for cls, r in zd_results.items():
            print(
                f'  {cls}: AUROC={r["auroc"]:.4f}  '
                f'detected={r["detected_above_threshold"]}/{r["total_samples"]} '
                f'({r["detection_rate"] * 100:.1f}%)'
            )
    print(f'{"=" * 55}')
