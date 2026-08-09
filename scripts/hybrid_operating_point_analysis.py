"""Operating-point sweep for MambaHybrid reconstruction scores.

Usage:
    python scripts/hybrid_operating_point_analysis.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.networks import MambaResidualBlock  # noqa: E402


class MambaEncoder(nn.Module):
    def __init__(self, n_features, d_model=128, depth=4, d_state=16, d_conv=4,
                 expansion=2, dropout=0.1):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_features, d_model))
        self.blocks = nn.ModuleList([
            MambaResidualBlock(d_model, d_state=d_state, d_conv=d_conv,
                               expand=expansion, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        z = self.value_proj(x.unsqueeze(-1)) + self.pos_embed[:, :x.shape[1], :]
        for block in self.blocks:
            z = block(z)
        return self.norm(z).mean(dim=1)


class MambaDecoder(nn.Module):
    def __init__(self, n_features, d_model=128, depth=2, d_state=16, d_conv=4,
                 expansion=2, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.pos_embed_dec = nn.Parameter(torch.zeros(1, n_features, d_model))
        self.blocks = nn.ModuleList([
            MambaResidualBlock(d_model, d_state=d_state, d_conv=d_conv,
                               expand=expansion, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 1, bias=True)

    def forward(self, emb):
        z = emb.unsqueeze(1).expand(-1, self.n_features, -1) + self.pos_embed_dec
        for block in self.blocks:
            z = block(z)
        return self.proj(self.norm(z)).squeeze(-1)


class MambaHybrid(nn.Module):
    def __init__(self, n_features, n_classes, d_model=128, depth=4, d_state=16,
                 d_conv=4, expansion=2, dropout=0.1, decoder_depth=2, **kw):
        super().__init__()
        self.encoder = MambaEncoder(
            n_features, d_model, depth, d_state, d_conv, expansion, dropout,
        )
        self.classifier = nn.Linear(d_model, n_classes)
        self.decoder = MambaDecoder(
            n_features, d_model, decoder_depth, d_state, d_conv, expansion, dropout,
        )

    def forward(self, x, return_recon=False):
        emb = self.encoder(x)
        logits = self.classifier(emb)
        if return_recon:
            recon = self.decoder(emb)
            return logits, recon
        return logits


RUNS = [
    {
        "name": "CICIDS2017_v1",
        "data_path": "dataset/CICIDS2017_smote3",
        "seeds": [42],
        "run_dir_tpl": str(ROOT / "weights/CICIDS2017/mamba_hybrid/seed_{seed}"),
        "benign_label": "BENIGN",
        "zeroday_classes": ["Bot", "PortScan"],
    },
    {
        "name": "CICIDS2017_v2",
        "data_path": "dataset/CICIDS2017_smote3",
        "seeds": [42],
        "run_dir_tpl": str(ROOT / "weights/CICIDS2017/mamba_hybrid_v2/seed_{seed}"),
        "benign_label": "BENIGN",
        "zeroday_classes": ["DDoS", "DoS Hulk"],
    },
]

UNSW_RUN = {
    "name": "UNSW-NB15",
    "data_path": "dataset/UNSW-NB15",
    "checkpoint": str(ROOT / "weights/UNSW-NB15/mamba_hybrid_zeroday_best.pt"),
    "benign_label": "Normal",
    "zeroday_classes": ["Worms", "Shellcode"],
    "n_features": 42,
    "n_classes": 8,
    "d_model": 128,
    "depth": 4,
    "d_state": 16,
    "d_conv": 4,
    "expansion": 2,
    "dropout": 0.1,
    "decoder_depth": 2,
}

FPR_TARGETS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
ASSUMED_ATTACK_RATE = 1e-4


def precision_at(recall, fpr, attack_rate):
    tp = recall * attack_rate
    fp = fpr * (1 - attack_rate)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def _sweep(model, data_path, benign_label, zeroday_classes):
    pp = joblib.load(ROOT / data_path / "preprocess.joblib")
    classes = list(pp["label_encoder"].classes_)
    benign_id = classes.index(benign_label)
    X = np.load(ROOT / data_path / "X_test.npy")
    y = np.load(ROOT / data_path / "y_test.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    bs = 1024
    recon_errs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i + bs], dtype=torch.float32, device=device)
            _, recon = model(xb, return_recon=True)
            recon_errs.append(((xb - recon) ** 2).mean(dim=1).cpu().numpy())
    recon_err = np.concatenate(recon_errs)
    benign_err = recon_err[y == benign_id]

    result = {"n_benign_test": int(len(benign_err)), "fpr_sweep": {}}
    for fpr in FPR_TARGETS:
        thr = float(np.percentile(benign_err, 100 * (1 - fpr)))
        actual_fpr = float((benign_err > thr).mean())
        per_class = {}
        for zd_cls in zeroday_classes:
            if zd_cls not in classes:
                continue
            zd_id = classes.index(zd_cls)
            zd_err = recon_err[y == zd_id]
            if len(zd_err) == 0:
                continue
            recall = float((zd_err > thr).mean())
            per_class[zd_cls] = {
                "detection_rate": recall,
                "precision_at_1e-4_attack_rate": precision_at(
                    recall, actual_fpr, ASSUMED_ATTACK_RATE,
                ),
                "n_test_samples": int(len(zd_err)),
            }
        result["fpr_sweep"][f"{fpr:.3f}"] = {
            "threshold": thr,
            "actual_benign_fpr": actual_fpr,
            "per_class": per_class,
        }
    return result


def run_one_raw_state_dict(cfg):
    model = MambaHybrid(
        n_features=cfg["n_features"], n_classes=cfg["n_classes"],
        d_model=cfg["d_model"], depth=cfg["depth"], d_state=cfg["d_state"],
        d_conv=cfg["d_conv"], expansion=cfg["expansion"], dropout=cfg["dropout"],
        decoder_depth=cfg["decoder_depth"],
    )
    state = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return _sweep(model, cfg["data_path"], cfg["benign_label"], cfg["zeroday_classes"])


def run_one(cfg, seed):
    run_dir = Path(cfg["run_dir_tpl"].format(seed=seed))
    ck = torch.load(run_dir / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    with open(run_dir / "config_snapshot.yaml") as f:
        ycfg = yaml.safe_load(f)
    mc = ycfg["model"]
    hcfg = ycfg["hybrid"]

    model = MambaHybrid(
        n_features=ck["n_features"], n_classes=ck["n_classes"],
        d_model=mc.get("d_model", 128), depth=mc.get("depth", 4),
        d_state=mc.get("d_state", 16), d_conv=mc.get("d_conv", 4),
        expansion=mc.get("expansion", 2), dropout=mc.get("dropout", 0.1),
        decoder_depth=hcfg.get("decoder_depth", 2),
    )
    model.load_state_dict(ck["model"])
    model.eval()
    return _sweep(model, cfg["data_path"], cfg["benign_label"], cfg["zeroday_classes"])


def main():
    out = {}
    out_path = ROOT / "outputs" / "hybrid_operating_point_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for cfg in RUNS:
        out[cfg["name"]] = {}
        for seed in cfg["seeds"]:
            print(f"=== {cfg['name']} seed={seed} ===", flush=True)
            r = run_one(cfg, seed)
            out[cfg["name"]][seed] = r
            for fpr_key, sweep in r["fpr_sweep"].items():
                for cls, stats in sweep["per_class"].items():
                    print(
                        f"  FPR={fpr_key} {cls}: "
                        f"detection_rate={stats['detection_rate']:.4f} "
                        f"precision@1e-4={stats['precision_at_1e-4_attack_rate']:.4f}",
                        flush=True,
                    )
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)

    print(f"=== {UNSW_RUN['name']} ===", flush=True)
    r = run_one_raw_state_dict(UNSW_RUN)
    out[UNSW_RUN["name"]] = {"single": r}
    for fpr_key, sweep in r["fpr_sweep"].items():
        for cls, stats in sweep["per_class"].items():
            print(
                f"  FPR={fpr_key} {cls}: "
                f"detection_rate={stats['detection_rate']:.4f} "
                f"precision@1e-4={stats['precision_at_1e-4_attack_rate']:.4f}",
                flush=True,
            )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
