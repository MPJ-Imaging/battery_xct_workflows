# Overhangs Mini U-Net (cylindrical cell segmentation)

This directory contains a compact U-Net–style model (with a ResNet encoder) for segmenting electrode overhang regions in Li-ion cylindrical cell XCT slices. It is designed as a reproducible, minimal example to accompany the overhang analysis notebook in `battery_xct_workflows`.

The goal is not to provide a production-ready model, but a transparent reference implementation that can be inspected, retrained, or adapted.

---

## Contents

- `data_generator.py`  
  Utilities for preparing and augmenting 2D overhang slices, including:
  - Loading multi-page TIFF stacks.
  - Rescaling and normalising images and masks.
  - Keras `ImageDataGenerator`-based augmentation and simple noise/brightness transforms.
  - A script entry-point to write augmented patches to disk.

- `unet_resnet_overhangs.py`  
  Definition and training script for the mini U-Net model:
  - `TiffDataGenerator`: Keras `Sequence` for on-disk `.tif` image/mask batches.
  - `build_overhangs_model()`: constructs and compiles a U-Net with a ResNet backbone and Dice/IoU metrics.
  - A training entry-point that fits the model on patches in `data/train/` and `data/test/`, with checkpointing and live plotting.

- `__init__.py`  
  Re-exports key utilities so they can be imported as:
  ```python
  from battery_xct_workflows.models.overhangs_mini_unet import (
      build_overhangs_model,
      TiffDataGenerator,
      combined_generator,
      add_noise,
      adjust_brightness,
  )

## Data and model artefacts

Full-resolution training data and the trained model are hosted on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17543023.svg)](https://doi.org/10.5281/zenodo.17543023)

## Expected Directory Structure 

```text
battery_xct_workflows/
  notebooks/
    01_cylindrical_cell_overhangs.ipynb
    02_cylindrical_cell_can.ipynb
  models/
    overhangs_mini_unet/
      data_generator.py
      unet_resnet_overhangs.py
      README.md
      data/
        train/
          images/
          masks/
        test/
          images/
          masks/
```

## Usage and Contributions 

Feel free to use in your work, extend and retrain the model by building larger and more diverse datasets, change model size, etc.
