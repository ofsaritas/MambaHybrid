"""Load released checkpoints and run a dummy forward pass.

Usage (from repo root):
    set PYTHONPATH=.
    python scripts/load_weights_demo.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multi.model import MambaHybrid  # noqa: E402

CHECKPOINTS = [
    ROOT / "weights/CICIDS2017/mamba_hybrid/seed_42/checkpoints/best.pt",
    ROOT / "weights/CICIDS2017/mamba_hybrid_v2/seed_42/checkpoints/best.pt",
    ROOT / "weights/UNSW-NB15/mamba_hybrid_zeroday_best.pt",
]


def load_one(path: Path) -> None:
    print(f"\n=== {path.relative_to(ROOT)} ===")
    if not path.exists():
        print("  MISSING")
        return

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        n_features = ckpt.get("n_features")
        n_classes = ckpt.get("n_classes")
        classes = ckpt.get("classes")
        holdouts = ckpt.get("holdout_indices")
        print(f"  n_features={n_features}  n_classes={n_classes}")
        if classes is not None:
            print(f"  classes ({len(classes)}): {classes}")
        if holdouts is not None:
            print(f"  holdout_indices: {holdouts}")
    else:
        state = ckpt
        n_features = None
        n_classes = None

    if n_features is None or n_classes is None:
        if "classifier.weight" in state:
            n_classes = state["classifier.weight"].shape[0]
        if "encoder.pos_embed" in state:
            n_features = state["encoder.pos_embed"].shape[1]
        print(f"  inferred n_features={n_features}  n_classes={n_classes}")

    model = MambaHybrid(
        n_features=int(n_features),
        n_classes=int(n_classes),
        d_model=128,
        depth=4,
        d_state=16,
        d_conv=4,
        expansion=2,
        dropout=0.1,
        decoder_depth=2,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  missing={len(missing)} unexpected={len(unexpected)} params={n_params:,}")
    model.eval()
    x = torch.randn(2, int(n_features))
    with torch.no_grad():
        logits, recon = model(x, return_recon=True)
        score = model.anomaly_score(x)
    print(
        f"  forward OK  logits={tuple(logits.shape)} "
        f"recon={tuple(recon.shape)} score={tuple(score.shape)}"
    )


if __name__ == "__main__":
    for p in CHECKPOINTS:
        load_one(p)
    print("\nDone.")
