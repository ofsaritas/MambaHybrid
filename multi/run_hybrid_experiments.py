"""Run hybrid train/eval loops for one or more datasets.

Usage:
    python multi/run_hybrid_experiments.py --dataset CICIDS2017 --seeds 42
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

_env = os.environ.copy()
_env['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + _env.get('PYTHONPATH', '')

DATASET_CONFIGS = {
    'CICIDS2017': 'multi/configs/hybrid_cicids2017_smote_v2.yaml',
    'CICIDS2017_v1': 'multi/configs/hybrid_cicids2017_smote.yaml',
    'CICIDS2017_v2': 'multi/configs/hybrid_cicids2017_smote_v2.yaml',
    'UNSW-NB15': 'multi/configs/hybrid_unsw_nb15.yaml',
}


def _read_epoch(last_pt: Path) -> str:
    try:
        import torch
        s = torch.load(last_pt, map_location='cpu', weights_only=False)
        return str(s.get('epoch', '?'))
    except Exception:
        return '?'


def run(cmd, label):
    print(f'\n{"=" * 60}\n  {label}')
    print(f'  CMD: {" ".join(str(c) for c in cmd)}')
    print(f'{"=" * 60}')
    result = subprocess.run(cmd, capture_output=False, cwd=PROJECT_ROOT, env=_env)
    ok = result.returncode == 0
    print(f'  --> {"OK" if ok else "FAILED (exit " + str(result.returncode) + ")"}')
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42])
    ap.add_argument('--skip_existing', action='store_true')
    args = ap.parse_args()

    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == 'all' else [args.dataset]
    logs = {}

    for ds in datasets:
        cfg = DATASET_CONFIGS.get(ds)
        if cfg is None:
            print(f'Unknown dataset: {ds}')
            continue

        print(f'\n{"#" * 70}\n# HYBRID DATASET: {ds}\n{"#" * 70}')
        logs[ds] = {'trained': [], 'failed': [], 'skipped': []}

        for seed in args.seeds:
            run_dir = Path('outputs') / ds / 'mamba_hybrid' / f'seed_{seed}'
            label = f'{ds} | mamba_hybrid | seed={seed}'

            fully_done = (
                (run_dir / 'checkpoints' / 'best.pt').exists()
                and (run_dir / 'metrics' / 'test_metrics.json').exists()
            )
            if args.skip_existing and fully_done:
                print(f'  SKIP (already complete): {label}')
                logs[ds]['skipped'].append(label)
                continue

            train_cmd = [
                PYTHON, 'multi/train_hybrid.py',
                '--config', cfg, '--seed', str(seed),
            ]
            last_pt = run_dir / 'checkpoints' / 'last.pt'
            if last_pt.exists() and not fully_done:
                train_cmd += ['--resume', str(last_pt)]
                print(f'  AUTO-RESUME from last.pt (epoch {_read_epoch(last_pt)}): {label}')

            ok = run(train_cmd, f'TRAIN HYBRID | {label}')
            if not ok:
                logs[ds]['failed'].append(label)
                continue

            run(
                [
                    PYTHON, 'multi/evaluate_hybrid.py',
                    '--config', cfg, '--run_dir', str(run_dir),
                ],
                f'EVAL HYBRID  | {label}',
            )
            logs[ds]['trained'].append(label)

    print('\n\n' + '=' * 70)
    print('HYBRID EXPERIMENT SUMMARY')
    print('=' * 70)
    for ds, log in logs.items():
        print(
            f'\n{ds}: trained={len(log["trained"])}, '
            f'skipped={len(log["skipped"])}, failed={len(log["failed"])}'
        )
        if log['failed']:
            print('  FAILED:', log['failed'])

    Path('outputs').mkdir(exist_ok=True)
    Path('outputs/hybrid_experiment_log.json').write_text(
        json.dumps(logs, indent=2), encoding='utf-8')
    print('\nFull log -> outputs/hybrid_experiment_log.json')
