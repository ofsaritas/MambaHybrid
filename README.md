# MambaHybrid

Code and trained weights for:

**MambaHybrid: Selective state-space models for zero-day network intrusion detection**  
Ömer Faruk Sarıtaş, Abdullah Avan  
https://github.com/ofsaritas/MambaHybrid

## Contents

- `multi/` — model, train, evaluate, configs
- `src/` — Mamba blocks, preprocessing, metrics
- `scripts/` — baselines and analysis utilities
- `weights/` — trained checkpoints (CICIDS2017, UNSW-NB15)
- `results/` — reported metrics (JSON)

Raw datasets are not included. Download CICIDS2017 and UNSW-NB15 from their official sources.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

From the repository root:

```bash
# Windows PowerShell
$env:PYTHONPATH = (Get-Location).Path

# Linux/macOS
export PYTHONPATH=$PWD
```

Optional (Linux + CUDA): install `mamba-ssm` for the official CUDA kernel. Without it, a pure-PyTorch selective-scan fallback is used.

## Datasets

| Dataset | Link |
|---------|------|
| CICIDS2017 | https://www.unb.ca/cic/datasets/ids-2017.html |
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset |

Place raw CSVs under the paths in the YAML configs (`data/raw/...`), or edit those paths.

## Train

```bash
python multi/train_hybrid.py --config multi/configs/hybrid_cicids2017_smote.yaml --seed 42
python multi/train_hybrid.py --config multi/configs/hybrid_cicids2017_smote_v2.yaml --seed 42
python multi/train_hybrid.py --config multi/configs/hybrid_unsw_nb15.yaml --seed 42
```

UNSW standalone trainer (same protocol, pure-PyTorch path):

```bash
python scripts/unsw_nb15_mamba_hybrid_zeroday.py
```

## Evaluate / load weights

```bash
python multi/evaluate_hybrid.py \
  --config multi/configs/hybrid_cicids2017_smote_v2.yaml \
  --run_dir weights/CICIDS2017/mamba_hybrid_v2/seed_42

python scripts/load_weights_demo.py
```

Released checkpoints:

| Path | Holdout |
|------|---------|
| `weights/CICIDS2017/mamba_hybrid/seed_42/checkpoints/best.pt` | Bot, PortScan |
| `weights/CICIDS2017/mamba_hybrid_v2/seed_42/checkpoints/best.pt` | DDoS, DoS Hulk |
| `weights/UNSW-NB15/mamba_hybrid_zeroday_best.pt` | Worms, Shellcode |

## Baselines

```bash
python scripts/classical_baseline.py
python scripts/decoupled_zeroday_baseline_multiseed.py
python scripts/hybrid_operating_point_analysis.py
```

Reported metrics are also under `results/`.

## Citation

```bibtex
@article{saritas2026mambahybrid,
  title   = {MambaHybrid: Selective state-space models for zero-day network intrusion detection},
  author  = {Sar{\i}ta{\c{s}}, {\"O}mer Faruk and Avan, Abdullah},
  journal = {International Journal of Information Security Science},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT — see [LICENSE](LICENSE).

Contact: omer@kayseri.edu.tr
