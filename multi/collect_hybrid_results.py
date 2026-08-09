"""Collect evaluation metrics from run directories and print a summary.

Usage:
    python multi/collect_hybrid_results.py
"""

import json
from pathlib import Path

import numpy as np


def load_results(run_dir):
    run = Path(run_dir)
    results = {}
    tf = run / 'metrics/test_metrics.json'
    af = run / 'metrics/anomaly_results.json'
    if tf.exists():
        results['cls'] = json.loads(tf.read_text())
    if af.exists():
        results['ano'] = json.loads(af.read_text())
    return results


DATASETS = {
    'CICIDS2017_v1': {
        'base': 'weights/CICIDS2017/mamba_hybrid',
        'seeds': [42],
    },
    'CICIDS2017_v2': {
        'base': 'weights/CICIDS2017/mamba_hybrid_v2',
        'seeds': [42],
    },
}

root = Path('.')

for ds_name, ds_cfg in DATASETS.items():
    print(f"\n{'=' * 65}")
    print(f"  {ds_name}")
    print(f"{'=' * 65}")
    base = root / ds_cfg['base']

    cls_metrics = []
    ano_metrics = []
    zd_metrics = {}

    for seed in ds_cfg['seeds']:
        run_dir = base / f'seed_{seed}'
        r = load_results(run_dir)
        if not r:
            print(f"  seed_{seed}: no results yet")
            continue
        if 'cls' in r:
            c = r['cls']
            cls_metrics.append({
                'seed': seed,
                'acc': c.get('accuracy', float('nan')),
                'wf1': c.get('weighted_f1', float('nan')),
                'mf1': c.get('macro_f1', float('nan')),
                'mcc': c.get('mcc', float('nan')),
                'roc': c.get('roc_auc_ovr_macro', float('nan')),
            })
            roc = c.get('roc_auc_ovr_macro', float('nan'))
            print(
                f"  seed_{seed}  Acc={c['accuracy']:.4f}  wF1={c['weighted_f1']:.4f}  "
                f"mF1={c['macro_f1']:.4f}  MCC={c['mcc']:.4f}  ROC={roc:.4f}"
            )
        if 'ano' in r:
            a = r['ano']
            ano_metrics.append(a['anomaly_auroc_all_attacks'])
            print(
                f"         Anomaly AUROC={a['anomaly_auroc_all_attacks']:.4f}  "
                f"PR-AUC={a['anomaly_pr_auc_all_attacks']:.4f}  "
                f"thr={a['threshold_used']:.4f}"
            )
            if a.get('zeroday_holdout'):
                for zd_cls, zd_r in a['zeroday_holdout'].items():
                    zd_metrics.setdefault(zd_cls, []).append(zd_r)
                    print(
                        f"         [ZERODAY] {zd_cls}: AUROC={zd_r['auroc']:.4f}  "
                        f"det={zd_r['detected_above_threshold']}/{zd_r['total_samples']} "
                        f"({zd_r['detection_rate'] * 100:.1f}%)"
                    )

    if len(cls_metrics) > 1:
        accs = [m['acc'] for m in cls_metrics]
        wf1s = [m['wf1'] for m in cls_metrics]
        mf1s = [m['mf1'] for m in cls_metrics]
        print(f"\n  SUMMARY ({len(cls_metrics)} seeds):")
        print(f"    Accuracy:    {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"    Weighted F1: {np.mean(wf1s):.4f} +/- {np.std(wf1s):.4f}")
        print(f"    Macro F1:    {np.mean(mf1s):.4f} +/- {np.std(mf1s):.4f}")
        if ano_metrics:
            print(
                f"    Anomaly AUROC: {np.mean(ano_metrics):.4f} +/- {np.std(ano_metrics):.4f}"
            )
        for zd_cls, zd_list in zd_metrics.items():
            aurocs = [z['auroc'] for z in zd_list]
            drs = [z['detection_rate'] for z in zd_list]
            print(
                f"    [ZERODAY] {zd_cls}: AUROC {np.mean(aurocs):.4f}+/-{np.std(aurocs):.4f}  "
                f"DetRate {np.mean(drs) * 100:.1f}%+/-{np.std(drs) * 100:.1f}%"
            )

print("\nDone.")
