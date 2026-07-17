# Data Instructions

This repository does not store raw petrographic thin-section images. The
dataset used in the manuscript is archived separately on Zenodo:

https://doi.org/10.5281/zenodo.21304480

## Expected Data Root

After downloading and extracting the dataset, set the command-line arguments
`--data_root` and `--meta_dir` to the local dataset folder.

A typical local structure is:

```text
DA-TWML_dataset/
  images/
    igneous/
    metamorphic/
    sedimentary/
  metadata.csv
  class_summary.csv
  splits/
    README_splits.md
  README_DATASET.md
  LICENSE
```

The release keeps the original image filenames and provides `metadata.csv` with
stable image identifiers, broad rock groups, parsed class labels, relative image
paths, file extensions, and file sizes.

## Using a Custom Local Layout

If your local copy uses a different folder layout, pass the appropriate path to:

```bash
python src/rock_fewshot_datwml_experiments.py \
  --data_root "/path/to/dataset" \
  --meta_dir "/path/to/dataset"
```

## Leakage Control

The scripts create class-disjoint train/validation/test splits. If additional
crops, paired-polarization metadata, or specimen-level metadata are added, all
derived images from the same source specimen should remain in the same split.

