"""Joint training for MambaHybrid (classifier + benign reconstruction).

Usage:
    python multi/train_hybrid.py --config multi/configs/hybrid_cicids2017_smote_v2.yaml --seed 42
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from multi.model import MambaHybrid
from src.data.preprocess import load_arrays, prepare_dataset
from src.evaluation.metrics import compute_metrics
from src.utils.config import load_config, save_config
from src.utils.seed import set_seed


def focal_loss(logits, y, weight=None, gamma=2.0):
    ce = nn.functional.cross_entropy(logits, y, weight=weight, reduction='none')
    pt = torch.exp(-ce.clamp(max=88.0))
    fl = (1 - pt) ** gamma * ce
    return fl.nanmean()


def recon_loss_benign(x_orig, recon, y, benign_idx):
    mask = (y == benign_idx)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=x_orig.device)
    return nn.functional.mse_loss(recon[mask], x_orig[mask])


def find_benign_idx(classes_list, label_hint):
    if label_hint in classes_list:
        return classes_list.index(label_hint)
    for candidate in ('BENIGN', 'benign', 'Normal', 'normal', 'Background', 'background'):
        if candidate in classes_list:
            print(f'Label hint "{label_hint}" not found, using: {candidate}')
            return classes_list.index(candidate)
    print(f'No benign class found; using index 0 ({classes_list[0]})')
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resume', default=None, help='Path to last.pt to resume from')
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    prepare_dataset(cfg)

    Xtr, ytr = load_arrays(cfg['dataset']['processed_dir'], 'train')
    Xva, yva = load_arrays(cfg['dataset']['processed_dir'], 'val')

    prep = joblib.load(Path(cfg['dataset']['processed_dir']) / 'preprocess.joblib')
    n_features = Xtr.shape[1]
    classes_list = list(prep['label_encoder'].classes_)
    n_classes = len(classes_list)

    n_classes_dataset = n_classes
    hcfg_early = cfg.get('hybrid', {})
    zeroday_classes = hcfg_early.get('zeroday_holdout_classes', [])
    holdout_indices = [classes_list.index(c) for c in zeroday_classes if c in classes_list]
    if holdout_indices:
        print(f'  [HOLDOUT] Removing from train/val: {zeroday_classes} (indices {holdout_indices})')
        keep_tr = ~np.isin(ytr, holdout_indices)
        keep_va = ~np.isin(yva, holdout_indices)
        Xtr, ytr = Xtr[keep_tr], ytr[keep_tr]
        Xva, yva = Xva[keep_va], yva[keep_va]
        print(f'  [HOLDOUT] Train: {len(ytr)} samples, Val: {len(yva)} samples')
        kept_idx = sorted(set(range(n_classes_dataset)) - set(holdout_indices))
        label_remap = {old: new for new, old in enumerate(kept_idx)}
        ytr = np.vectorize(label_remap.__getitem__)(ytr)
        yva = np.vectorize(label_remap.__getitem__)(yva)
        classes_list = [classes_list[i] for i in kept_idx]
        n_classes = len(classes_list)
        print(f'  [HOLDOUT] Remapped labels. Training classes: {classes_list}')

    n_classes_train = len(np.unique(ytr))
    print(f'  Training classes: {n_classes_train} (total dataset classes: {n_classes_dataset})')

    hcfg = cfg.get('hybrid', {})
    benign_idx = find_benign_idx(classes_list, hcfg.get('benign_label', 'BENIGN'))
    print(f'Classes: {classes_list}')
    print(f'BENIGN index: {benign_idx} ({classes_list[benign_idx]})')
    print(f'n_features={n_features}, n_classes={n_classes}')

    warmup_epochs = hcfg.get('warmup_epochs', 5)
    ae_lambda = hcfg.get('ae_lambda', 0.1)
    decoder_depth = hcfg.get('decoder_depth', 2)
    pct = hcfg.get('anomaly_threshold_percentile', 95)
    total_epochs = cfg['training']['epochs']
    patience = cfg['training']['patience']
    lr = cfg['training']['lr']
    bs = cfg['training']['batch_size']
    noise_std = cfg['training'].get('noise_aug_std', 0.05)

    run_name = cfg.get('outputs', {}).get('run_name', 'mamba_hybrid')
    out = Path(cfg['outputs']['root']) / cfg['dataset']['name'] / run_name / f'seed_{args.seed}'
    ckpt_dir = out / 'checkpoints'
    metrics_dir = out / 'metrics'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / 'config_snapshot.yaml')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mc = cfg.get('model', {})
    n_classes_model = len(np.unique(ytr))
    model = MambaHybrid(
        n_features=n_features, n_classes=n_classes_model,
        d_model=mc.get('d_model', 128), depth=mc.get('depth', 4),
        d_state=mc.get('d_state', 16), d_conv=mc.get('d_conv', 4),
        expansion=mc.get('expansion', 2), dropout=mc.get('dropout', 0.1),
        decoder_depth=decoder_depth,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'MambaHybrid parameters: {total_params:,}')

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=cfg['training']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_epochs, eta_min=lr * 0.01)
    # AMP disabled: selective SSM is unstable in fp16 on some setups
    amp_enabled = False
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    cw = compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
    cw = np.clip(cw, a_min=None, a_max=50.0)
    class_weight = torch.tensor(cw, dtype=torch.float32, device=device)

    train_ds = TensorDataset(
        torch.tensor(Xtr, dtype=torch.float32),
        torch.tensor(ytr, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(Xva, dtype=torch.float32),
        torch.tensor(yva, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    best = -1
    bad = 0
    log = []
    start_epoch = 1
    nan_streak = 0

    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        if 'scheduler' in state:
            scheduler.load_state_dict(state['scheduler'])
        if 'scaler' in state:
            scaler.load_state_dict(state['scaler'])
        start_epoch = state['epoch'] + 1
        best = state.get('best_metric', -1)
        bad = state.get('bad_epochs', 0)
        log = state.get('train_log', [])
        print(f'  Resumed from epoch {state["epoch"]}, best_f1={best:.4f}, bad={bad}')

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        t0 = time.time()
        total_cls = 0.0
        total_ae = 0.0
        use_ae = (epoch > warmup_epochs)

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x_noisy = x + torch.randn_like(x) * noise_std
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=amp_enabled):
                if use_ae:
                    logits, recon = model(x_noisy, return_recon=True)
                    loss_cls = focal_loss(
                        logits, y, class_weight,
                        cfg['training'].get('focal_gamma', 2.0),
                    )
                    loss_ae = recon_loss_benign(x, recon, y, benign_idx)
                    loss = loss_cls + ae_lambda * loss_ae
                else:
                    logits = model(x_noisy)
                    loss_cls = focal_loss(
                        logits, y, class_weight,
                        cfg['training'].get('focal_gamma', 2.0),
                    )
                    loss_ae = torch.tensor(0.0)
                    loss = loss_cls

            if not torch.isfinite(loss):
                nan_streak += 1
                if nan_streak > 50:
                    raise RuntimeError(
                        f'Non-finite loss persists at epoch {epoch} (>{nan_streak} batches)'
                    )
                opt.zero_grad(set_to_none=True)
                continue
            nan_streak = 0

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg['training'].get('gradient_clip_norm', 1.0),
            )
            scaler.step(opt)
            scaler.update()
            total_cls += float(loss_cls.item()) * len(x)
            total_ae += float(loss_ae.item()) * len(x)

        scheduler.step()

        model.eval()
        probs_list = []
        ys_list = []
        ae_scores = []
        ae_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits, recon = model(x, return_recon=True)
                probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
                ys_list.append(y.numpy())
                scores = ((x - recon) ** 2).mean(dim=1).cpu().numpy()
                ae_scores.append(scores)
                ae_labels.append((y.numpy() != benign_idx).astype(int))

        yv = np.concatenate(ys_list)
        pv = np.concatenate(probs_list)
        m = compute_metrics(yv, pv)

        ae_scores = np.concatenate(ae_scores)
        ae_labels = np.concatenate(ae_labels)
        try:
            from sklearn.metrics import roc_auc_score
            ae_auroc = float(roc_auc_score(ae_labels, ae_scores))
        except Exception:
            ae_auroc = 0.0

        entry = {
            'epoch': epoch,
            'phase': 'joint' if use_ae else 'warmup',
            'train_loss_cls': total_cls / len(train_ds),
            'train_loss_ae': total_ae / len(train_ds),
            'val_macro_f1': m['macro_f1'],
            'val_anomaly_auroc': ae_auroc,
            'lr': scheduler.get_last_lr()[0],
            'seconds': time.time() - t0,
            **m,
        }
        log.append(entry)
        print(entry)

        state = {
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'best_metric': max(best, m['macro_f1']),
            'bad_epochs': bad + (0 if m['macro_f1'] > best else 1),
            'train_log': log,
            'n_features': n_features,
            'n_classes': n_classes_model,
            'n_classes_full': n_classes_dataset,
            'benign_idx': benign_idx,
            'classes': classes_list,
            'holdout_indices': holdout_indices,
            'model_name': 'mamba_hybrid',
        }
        torch.save(state, ckpt_dir / 'last.pt')
        if m['macro_f1'] > best:
            best = m['macro_f1']
            bad = 0
            torch.save(state, ckpt_dir / 'best.pt')
        else:
            bad += 1

        (metrics_dir / 'train_log.json').write_text(
            json.dumps(log, indent=2), encoding='utf-8')

        joint_epoch = epoch - warmup_epochs
        if bad >= patience and joint_epoch >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    best_state = torch.load(ckpt_dir / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(best_state['model'])
    model.eval()

    benign_scores = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            scores = model.anomaly_score(x).cpu().numpy()
            mask = (y.numpy() == benign_idx)
            if mask.any():
                benign_scores.append(scores[mask])
    benign_scores = np.concatenate(benign_scores)
    threshold = float(np.percentile(benign_scores, pct))
    print(f'Anomaly threshold ({pct}th percentile of BENIGN val MSE): {threshold:.6f}')

    (metrics_dir / 'anomaly_threshold.json').write_text(
        json.dumps({
            'threshold': threshold,
            'percentile': pct,
            'benign_val_mse_mean': float(benign_scores.mean()),
            'benign_val_mse_std': float(benign_scores.std()),
        }, indent=2),
        encoding='utf-8',
    )

    print(f'\nDone. Best val macro_f1={best:.4f}  |  Final AE threshold={threshold:.6f}')
