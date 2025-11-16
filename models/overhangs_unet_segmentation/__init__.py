"""
Overhangs Mini U-Net model and data utilities.

This subpackage contains:

- Data generation and augmentation helpers for overhang segmentation.
- A U-Net + ResNet backbone model builder.
- A TIFF-based Keras Sequence for on-disk training data.
"""

from .unet_resnet_overhangs import (
    build_overhangs_model,
    TiffDataGenerator,
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_CHANNELS,
    BACKBONE,
)

from .data_generator import (
    combined_generator,
    custom_preprocessing_masks,
    add_noise,
    adjust_brightness,
)

__all__ = [
    # Model + config
    "build_overhangs_model",
    "TiffDataGenerator",
    "IMG_HEIGHT",
    "IMG_WIDTH",
    "IMG_CHANNELS",
    "BACKBONE",
    # Data generator helpers
    "combined_generator",
    "custom_preprocessing_masks",
    "add_noise",
    "adjust_brightness",
]