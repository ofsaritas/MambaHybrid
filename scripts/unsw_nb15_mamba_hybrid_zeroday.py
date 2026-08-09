"""MambaHybrid zero-day training on UNSW-NB15 (Worms + Shellcode holdout).

Usage:
    python scripts/unsw_nb15_mamba_hybrid_zeroday.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.networks import MambaResidualBlock  # noqa: E402

DATA_PATH = ROOT / "dataset" / "UNSW-NB15"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 512
WARMUP_EPOCHS = 5
TOTAL_EPOCHS = 50
PATIENCE = 10
D_MODEL, DEPTH, D_STATE, D_CONV, EXPANSION, DROPOUT = 128, 4, 16, 4, 2, 0.1
DECODER_DEPTH = 2
NOISE_STD = 0.05
GRAD_CLIP = 1.0
LR = 1e-3
AE_LAMBDA = 0.1
ANOMALY_PCT = 95
HOLDOUT_CLASSES = ["Worms", "Shellcode"]
BENIGN_LABEL = "Normal"
OUT_DIR = ROOT / "outputs"
OUT_JSON = OUT_DIR / "unsw_nb15_mamba_hybrid_zeroday_results.json"


class MambaEncoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.value_proj = nn.Linear(1, D_MODEL, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_features, D_MODEL))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            MambaResidualBlock(D_MODEL, d_state=D_STATE, d_conv=D_CONV,
                               expand=EXPANSION, dropout=DROPOUT)
            for _ in range(DEPTH)
        ])
        self.norm = nn.LayerNorm(D_MODEL)

    def forward(self, x):
        z = self.value_proj(x.unsqueeze(-1)) + self.pos_embed[:, :x.shape[1], :]
        for block in self.blocks:
            z = block(z)
        return self.norm(z).mean(dim=1)


class MambaDecoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.pos_embed_dec = nn.Parameter(torch.zeros(1, n_features, D_MODEL))
        nn.init.trunc_normal_(self.pos_embed_dec, std=0.02)
        self.blocks = nn.ModuleList([
            MambaResidualBlock(D_MODEL, d_state=D_STATE, d_conv=D_CONV,
                               expand=EXPANSION, dropout=DROPOUT)
            for _ in range(DECODER_DEPTH)
        ])
        self.norm = nn.LayerNorm(D_MODEL)
        self.proj = nn.Linear(D_MODEL, 1, bias=True)

    def forward(self, emb):
        z = emb.unsqueeze(1).expand(-1, self.n_features, -1) + self.pos_embed_dec
        for block in self.blocks:
            z = block(z)
        return self.proj(self.norm(z)).squeeze(-1)


class MambaHybrid(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.encoder = MambaEncoder(n_features)
        self.classifier = nn.Linear(D_MODEL, n_classes)
        self.decoder = MambaDecoder(n_features)

    def forward(self, x, return_recon=False):
        emb = self.encoder(x)
        logits = self.classifier(emb)
        if return_recon:
            return logits, self.decoder(emb)
        return logits

    def anomaly_score(self, x):
        emb = self.encoder(x)
        recon = self.decoder(emb)
        return ((x - recon) ** 2).mean(dim=1)


def focal_loss(logits, y, gamma=2.0):
    ce = nn.functional.cross_entropy(logits, y, reduction="none")
    pt = torch.exp(-ce.clamp(max=88.0))
    return ((1 - pt) ** gamma * ce).nanmean()


def recon_loss_benign(x_orig, recon, y, benign_idx):
    mask = y == benign_idx
    if mask.sum() == 0:
        return torch.tensor(0.0, device=x_orig.device)
    return nn.functional.mse_loss(recon[mask], x_orig[mask])


def load_data():
    import joblib
    prep = joblib.load(DATA_PATH / "preprocess.joblib")
    classes = list(prep["label_encoder"].classes_)
    Xtr = np.load(DATA_PATH / "X_train.npy")
    ytr = np.load(DATA_PATH / "y_train.npy")
    Xva = np.load(DATA_PATH / "X_val.npy")
    yva = np.load(DATA_PATH / "y_val.npy")
    Xte = np.load(DATA_PATH / "X_test.npy")
    yte = np.load(DATA_PATH / "y_test.npy")

    holdout_ids = [classes.index(c) for c in HOLDOUT_CLASSES]
    benign_id_orig = classes.index(BENIGN_LABEL)

    keep_tr = ~np.isin(ytr, holdout_ids)
    keep_va = ~np.isin(yva, holdout_ids)
    Xtr, ytr = Xtr[keep_tr], ytr[keep_tr]
    Xva, yva = Xva[keep_va], yva[keep_va]

    kept = sorted(set(range(len(classes))) - set(holdout_ids))
    remap = {old: new for new, old in enumerate(kept)}
    ytr = np.vectorize(remap.__getitem__)(ytr)
    yva = np.vectorize(remap.__getitem__)(yva)
    benign_id = remap[benign_id_orig]

    clf_test_mask = ~np.isin(yte, holdout_ids)
    Xte_clf = Xte[clf_test_mask]
    yte_clf = np.vectorize(remap.__getitem__)(yte[clf_test_mask])

    return (Xtr, ytr, Xva, yva, Xte, yte, Xte_clf, yte_clf,
            benign_id, holdout_ids, classes, len(kept))


def main():
    print("Loading UNSW-NB15 data...", flush=True)
    (Xtr, ytr, Xva, yva, Xte, yte, Xte_clf, yte_clf,
     benign_id, holdout_ids, classes, n_classes) = load_data()
    n_features = Xtr.shape[1]
    print(f"Xtr={Xtr.shape} n_classes={n_classes} holdout={HOLDOUT_CLASSES} "
          f"benign_id={benign_id} device={DEVICE}", flush=True)

    benign_id_orig = classes.index(BENIGN_LABEL)
    benign_test_mask = yte == benign_id_orig

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = MambaHybrid(n_features, n_classes).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MambaHybrid parameters: {total_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=TOTAL_EPOCHS, eta_min=LR * 0.01)

    train_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                              torch.tensor(ytr, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(Xva, dtype=torch.float32),
                            torch.tensor(yva, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    log = []
    best_f1 = -1
    best_state = None
    bad_epochs = 0

    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        use_ae = epoch > WARMUP_EPOCHS
        t0 = time.time()
        total_cls, total_ae = 0.0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_noisy = x + torch.randn_like(x) * NOISE_STD
            opt.zero_grad(set_to_none=True)
            if use_ae:
                logits, recon = model(x_noisy, return_recon=True)
                loss_cls = focal_loss(logits, y)
                loss_ae = recon_loss_benign(x, recon, y, benign_id)
                loss = loss_cls + AE_LAMBDA * loss_ae
            else:
                logits = model(x_noisy)
                loss_cls = focal_loss(logits, y)
                loss_ae = torch.tensor(0.0)
                loss = loss_cls
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total_cls += float(loss_cls.item()) * len(x)
            total_ae += float(loss_ae.item()) * len(x)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            Xva_t = torch.tensor(Xva, dtype=torch.float32).to(DEVICE)
            preds = []
            for i in range(0, len(Xva_t), 4096):
                logits = model(Xva_t[i:i + 4096])
                preds.append(logits.argmax(dim=1).cpu().numpy())
            preds = np.concatenate(preds)
        val_f1 = float(f1_score(yva, preds, average="macro"))

        entry = {
            "epoch": epoch, "phase": "joint" if use_ae else "warmup",
            "train_loss_cls": total_cls / len(train_ds),
            "train_loss_ae": total_ae / len(train_ds),
            "val_macro_f1": val_f1, "seconds": time.time() - t0,
        }
        log.append(entry)
        print(f"  epoch {epoch} ({entry['phase']}): val_macro_f1={val_f1:.4f} "
              f"{entry['seconds']:.1f}s", flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        joint_epoch = epoch - WARMUP_EPOCHS
        if bad_epochs >= PATIENCE and joint_epoch >= PATIENCE:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "unsw_nb15_mamba_hybrid_zeroday_trainlog.json", "w") as f:
            json.dump(log, f, indent=2)

    model.load_state_dict(best_state)
    model.eval()

    # final classification metrics on 8 known-class test set
    with torch.no_grad():
        Xte_clf_t = torch.tensor(Xte_clf, dtype=torch.float32).to(DEVICE)
        preds = []
        for i in range(0, len(Xte_clf_t), 4096):
            logits = model(Xte_clf_t[i:i + 4096])
            preds.append(logits.argmax(dim=1).cpu().numpy())
        preds = np.concatenate(preds)
    macro_f1 = float(f1_score(yte_clf, preds, average="macro"))
    weighted_f1 = float(f1_score(yte_clf, preds, average="weighted"))
    accuracy = float(accuracy_score(yte_clf, preds))
    mcc = float(matthews_corrcoef(yte_clf, preds))

    # anomaly threshold from BENIGN val samples
    with torch.no_grad():
        Xva_t = torch.tensor(Xva, dtype=torch.float32).to(DEVICE)
        scores_va = []
        for i in range(0, len(Xva_t), 4096):
            scores_va.append(model.anomaly_score(Xva_t[i:i + 4096]).cpu().numpy())
        scores_va = np.concatenate(scores_va)
    benign_va_mask = yva == benign_id
    threshold = float(np.percentile(scores_va[benign_va_mask], ANOMALY_PCT))

    # zero-day AUROC on full test set (original labels, includes held-out classes)
    with torch.no_grad():
        Xte_t = torch.tensor(Xte, dtype=torch.float32).to(DEVICE)
        scores = []
        for i in range(0, len(Xte_t), 4096):
            scores.append(model.anomaly_score(Xte_t[i:i + 4096]).cpu().numpy())
        scores = np.concatenate(scores)

    benign_scores = scores[benign_test_mask]
    zeroday_auroc = {}
    for cname in HOLDOUT_CLASSES:
        cid = classes.index(cname)
        mask = yte == cid
        if mask.sum() == 0:
            continue
        y_true = np.concatenate([np.zeros(len(benign_scores)), np.ones(mask.sum())])
        y_score = np.concatenate([benign_scores, scores[mask]])
        zeroday_auroc[cname] = float(roc_auc_score(y_true, y_score))

    result = {
        "seed": SEED,
        "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
        "n_epochs_run": len(log), "best_val_macro_f1": best_f1,
        "anomaly_threshold": threshold, "anomaly_threshold_percentile": ANOMALY_PCT,
        "classification_8class": {
            "accuracy": accuracy, "weighted_f1": weighted_f1,
            "macro_f1": macro_f1, "mcc": mcc,
        },
        "zeroday_auroc": zeroday_auroc,
        "holdout_classes": HOLDOUT_CLASSES,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    torch.save(best_state, OUT_DIR / "unsw_nb15_mamba_hybrid_zeroday_best.pt")
    print("RESULT:", json.dumps(result, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
