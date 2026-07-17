# DA-TWML

Domain-Aware Top-k Weighted Meta-Learning (DA-TWML) for few-shot petrographic
thin-section classification.

This repository contains the source code and lightweight result records for the
manuscript:

**DA-TWML: Domain-Aware Top-k Weighted Meta-Learning for Few-Shot Petrographic
Thin-Section Classification**

## Repository Contents

- `src/rock_fewshot_datwml_experiments.py`  
  Main implementation for ProtoNet, Random Top-k, loss-only task scoring,
  score-factor ablations, and full DA-TWML.
- `scripts/`  
  PowerShell entry points for 5-way 1-shot, 5-way 5-shot, and combined runs.
- `examples/quick_test.py`  
  Lightweight quick test that verifies the DA-TWML episode scoring and Top-k
  weighted aggregation logic without requiring the image dataset.
- `analysis/`  
  Figure-generation and result-analysis scripts used for manuscript diagnostics.
- `results/runs_acml/`  
  Lightweight CSV/JSON result summaries, class splits, histories, and per-run
  records. Model checkpoints and raw images are not included.
- `DATA.md`  
  Dataset placement instructions.

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The experiments were developed with Python 3.12 and PyTorch. A GPU is
recommended for full training runs, but the quick test runs on CPU.

## Quick Test

Run the quick test before launching full experiments:

```bash
python examples/quick_test.py
```

Expected output:

```text
DA-TWML quick test passed.
```

The quick test creates synthetic episode-level scores, performs Top-k episode
selection, computes softmax weights, and checks that the weighted meta-loss is
finite and reproducible. It is intended for reviewers to confirm that the
repository is executable without downloading the image dataset.

## Dataset

The experiments use the Nanjing University petrographic thin-section teaching
dataset. The dataset used in the manuscript is archived separately on Zenodo:

https://doi.org/10.5281/zenodo.21304480

After downloading the data, set `DATA_ROOT` and `META_DIR` to the local dataset
folder. See `DATA.md` for the expected folder layout.

## Reproducing the Main Experiments

Windows PowerShell example:

```powershell
$env:DATA_ROOT="D:\path\to\DA-TWML_dataset"
$env:META_DIR="D:\path\to\DA-TWML_dataset"
.\scripts\run_acml_1shot.ps1
```

Run the 5-way 5-shot ablation:

```powershell
.\scripts\run_acml_5shot.ps1
```

Run both shot settings:

```powershell
.\scripts\run_acml_all.ps1
```

The main script can also be called directly:

```bash
python src/rock_fewshot_datwml_experiments.py \
  --data_root "/path/to/DA-TWML_dataset" \
  --meta_dir "/path/to/DA-TWML_dataset" \
  --out_dir runs_acml \
  --experiment ablation \
  --methods protonet random_topk loss_only loss_u loss_d loss_c loss_u_d loss_u_c full \
  --ways 5 --shots 1 5 --query 5 \
  --seeds 0 1 2 \
  --epochs 30 --episodes_per_epoch 100 \
  --val_episodes 200 --test_episodes 1000 \
  --meta_batch 4 --topk 2 --image_size 84
```

## Outputs

Each run writes:

- `result_*.json`: best validation/test accuracy and run metadata.
- `history_*.csv`: epoch-level validation/test records.
- `class_splits_seed*.json`: class-disjoint train/validation/test splits.
- `all_results.csv` and `all_results_summary.csv`: collected result tables.

## Code Availability and License

The source code is openly available from:

https://github.com/zhangmy2002/DA-TWML

Archived release:

https://doi.org/10.5281/zenodo.21304321

The code is released under the MIT License. See `LICENSE`.

## Citation

If you use this code, please cite the associated manuscript after publication.

