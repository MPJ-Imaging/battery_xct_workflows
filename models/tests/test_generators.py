# tests/test_generators.py

import numpy as np
import tensorflow as tf
import tifffile as tifff

from pathlib import Path

from battery_xct_workflows.models.overhangs_unet_segmentation.data_generator import (
    combined_generator,
    custom_preprocessing_masks,
    add_noise,
    adjust_brightness,
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_CHANNELS,
)

from battery_xct_workflows.models.overhangs_unet_segmentation.unet_resnet_overhangs import (
    TiffDataGenerator,
)


# ----------------------------------------------------------------------
# Tests for in-memory combined_generator and augmentation helpers
# ----------------------------------------------------------------------

def test_custom_preprocessing_masks_binarises():
    """custom_preprocessing_masks should threshold to {0, 1}."""
    mask_batch = np.array(
        [
            [[0.2, 0.7], [0.49, 0.51]],
        ],
        dtype=np.float32,
    )  # shape (1, 2, 2)

    processed = custom_preprocessing_masks(mask_batch)
    unique_vals = np.unique(processed)

    assert processed.dtype == np.float32
    assert np.array_equal(unique_vals, np.array([0.0, 1.0], dtype=np.float32))


def test_add_noise_preserves_shape_and_range():
    """add_noise should not change shape and should keep values in [0, 1]."""
    img = np.random.rand(4, 5, 3).astype(np.float32)
    noisy = add_noise(img)

    assert noisy.shape == img.shape
    assert noisy.dtype == np.float32
    assert noisy.min() >= 0.0
    assert noisy.max() <= 1.0


def test_adjust_brightness_preserves_shape_and_range():
    """adjust_brightness should not change shape and should keep values in [0, 1]."""
    img = np.random.rand(4, 5, 3).astype(np.float32)
    adjusted = adjust_brightness(img)

    assert adjusted.shape == img.shape
    assert adjusted.dtype == np.float32
    assert adjusted.min() >= 0.0
    assert adjusted.max() <= 1.0


def test_combined_generator_shapes_and_binary_masks():
    """
    combined_generator should yield batches with correct shapes
    and masks remaining binary after augmentation.
    """
    n_samples = 8
    batch_size = 4

    # Synthetic images and masks
    images = np.random.rand(
        n_samples,
        IMG_HEIGHT,
        IMG_WIDTH,
        IMG_CHANNELS,
    ).astype(np.float32)

    # Integer masks {0, 1}, broadcast to 3 channels to match your pipeline
    masks = np.random.randint(
        0,
        2,
        size=(n_samples, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        dtype=np.uint8,
    ).astype(np.float32)

    gen = combined_generator(images, masks, batch_size=batch_size, seed=123)

    img_batch, mask_batch = next(gen)

    # Shapes
    assert img_batch.shape == (batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    assert mask_batch.shape == (batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    # Types
    assert img_batch.dtype == np.float32
    assert mask_batch.dtype == np.float32

    # Masks should remain binary after custom_preprocessing_masks
    unique_mask_vals = np.unique(mask_batch)
    assert np.all(np.isin(unique_mask_vals, [0.0, 1.0]))


# ----------------------------------------------------------------------
# Tests for on-disk TiffDataGenerator
# ----------------------------------------------------------------------

def test_tiff_data_generator_batch_shapes(tmp_path):
    """
    TiffDataGenerator should load .tif files and yield batches with the
    requested spatial size and a channel dimension.
    """
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    # Create a few tiny synthetic .tif image/mask pairs
    n_files = 4
    h, w, c = 32, 48, IMG_CHANNELS

    for i in range(n_files):
        img = np.random.rand(h, w, c).astype(np.float32)
        # Simple binary mask with same shape
        mask = (np.random.rand(h, w, c) > 0.5).astype(np.float32)

        tifff.imwrite(str(img_dir / f"img_{i}.tif"), img)
        tifff.imwrite(str(mask_dir / f"mask_{i}.tif"), mask)

    batch_size = 2
    target_size = (IMG_HEIGHT, IMG_WIDTH)

    gen = TiffDataGenerator(
        image_dir=str(img_dir),
        mask_dir=str(mask_dir),
        batch_size=batch_size,
        image_size=target_size,
        shuffle=False,
    )

    # __len__ should be n_files // batch_size
    assert len(gen) == n_files // batch_size

    batch_images, batch_masks = gen[0]

    # Expected shapes after resize
    assert batch_images.shape[0] == batch_size
    assert batch_masks.shape[0] == batch_size

    assert batch_images.shape[1] == IMG_HEIGHT
    assert batch_images.shape[2] == IMG_WIDTH

    assert batch_masks.shape[1] == IMG_HEIGHT
    assert batch_masks.shape[2] == IMG_WIDTH

    # Keep a channel dimension (either 1 or >=1)
    assert batch_images.ndim == 4
    assert batch_masks.ndim == 4

    # Types should be float32 per generator implementation
    assert batch_images.dtype == np.float32
    assert batch_masks.dtype == np.float32


def test_tiff_data_generator_shuffle_on_epoch_end(tmp_path):
    """
    on_epoch_end should shuffle indices when shuffle=True.
    We check that the order of indices changes at least once.
    """
    img_dir = tmp_path / "images2"
    mask_dir = tmp_path / "masks2"
    img_dir.mkdir()
    mask_dir.mkdir()

    n_files = 6
    h, w, c = 16, 16, IMG_CHANNELS

    for i in range(n_files):
        img = np.random.rand(h, w, c).astype(np.float32)
        mask = (np.random.rand(h, w, c) > 0.5).astype(np.float32)
        tifff.imwrite(str(img_dir / f"img_{i}.tif"), img)
        tifff.imwrite(str(mask_dir / f"mask_{i}.tif"), mask)

    gen = TiffDataGenerator(
        image_dir=str(img_dir),
        mask_dir=str(mask_dir),
        batch_size=2,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=True,
    )

    original_indices = gen.indices.copy()
    gen.on_epoch_end()
    new_indices = gen.indices

    # With high probability, the order should change at least once.
    # (If it doesn't, test will occasionally be flaky, but it's extremely unlikely.)
    assert not np.array_equal(original_indices, new_indices)

