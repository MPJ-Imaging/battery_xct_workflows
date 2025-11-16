import os
import logging

# Use tf.keras backend for segmentation_models
os.environ['SM_FRAMEWORK'] = 'tf.keras'

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import tifffile as tifff
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------
# Basic logging / version info
# --------------------------------------------------------------------
print(f'Using TensorFlow v.{tf.__version__}')
print(f'Using Keras v.{tf.keras.__version__}')

# Suppress verbose TensorFlow logging (optional)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# --------------------------------------------------------------------
# Configuration Parameters
# --------------------------------------------------------------------

# Image dimensions used by the model
IMG_CHANNELS = 3
IMG_HEIGHT = 256
IMG_WIDTH = 800

# Batch size for the augmented data generator
BATCH_SIZE = 10
SEED = 42

# Raw image / mask directories (kept for reference; not used in this script)
IMAGE_DIR = 'training/axial_images'
MASK_DIR = 'training/axial_masks'
IMAGE_VAL_DIR = 'validation/axial_images'
MASK_VAL_DIR = 'validation/axial_masks'

# Output directories for augmented data
TRAIN_IMG_DIR = 'data/train/images'
TRAIN_MASK_DIR = 'data/train/masks'
VAL_IMG_DIR = 'data/test/images'
VAL_MASK_DIR = 'data/test/masks'

# Ensure output directories exist
for d in (TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR):
    os.makedirs(d, exist_ok=True)

# Model backbone
BACKBONE = 'resnet101'

# --------------------------------------------------------------------
# Data Loading and Preprocessing
# --------------------------------------------------------------------

def load_data(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load images and corresponding masks from multi-page TIFF files,
    resize, normalise, and binarise.

    Parameters
    ----------
    height : int
        Target height for resizing.
    width : int
        Target width for resizing.

    Returns
    -------
    images : np.ndarray
        Float32 array of shape (N, height, width, 3) in [0, 1].
    masks : np.ndarray
        Float32 array of shape (N, height, width, 3) with 0/1 values.
    """
    image_stack = tifff.imread('data/images.tif').astype(np.uint8)
    mask_stack = tifff.imread('data/masks.tif').astype(np.uint8)

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for img, mask in zip(image_stack, mask_stack):
        # Image: grayscale → RGB, resize, normalise to [0, 1]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_rgb = cv2.resize(img_rgb, (width, height))
        img_rgb = (img_rgb / np.amax(img_rgb)).astype(np.float32)
        images.append(img_rgb)

        # Mask: grayscale → RGB, resize, binarise
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask_rgb = cv2.resize(mask_rgb, (width, height))
        mask_bin = (mask_rgb > 0).astype(np.float32)
        masks.append(mask_bin)

    images_array = np.array(images, dtype=np.float32)
    masks_array = np.array(masks, dtype=np.float32)
    return images_array, masks_array


print("Loading and preprocessing data...")
images, masks = load_data(IMG_HEIGHT, IMG_WIDTH)

images, val_images, masks, val_masks = train_test_split(
    images,
    masks,
    test_size=0.1,
    random_state=SEED,
)

print(f"Loaded {len(images)} training images and {len(masks)} training masks.")
print(f"Loaded {len(val_images)} validation images and {len(val_masks)} validation masks.")
print(f"Training array shape: {images.shape}")

# --------------------------------------------------------------------
# Data Augmentation
# --------------------------------------------------------------------

def custom_preprocessing_masks(mask_batch: np.ndarray) -> np.ndarray:
    """
    Ensure masks remain binary after augmentation.

    Parameters
    ----------
    mask_batch : np.ndarray
        Batch of masks with arbitrary values after augmentation.

    Returns
    -------
    np.ndarray
        Batch of masks thresholded to 0 or 1.
    """
    return np.where(mask_batch > 0.5, 1, 0).astype(np.float32)


def combined_generator(
    image_array: np.ndarray,
    mask_array: np.ndarray,
    batch_size: int,
    seed: int,
):
    """
    Create a generator that yields augmented (image, mask) batches.

    Two ImageDataGenerators are constructed with identical augmentation
    parameters and the same seed, so that images and masks receive
    consistent geometric transforms.

    Parameters
    ----------
    image_array : np.ndarray
        Array of input images of shape (N, H, W, C).
    mask_array : np.ndarray
        Array of corresponding masks of shape (N, H, W, C).
    batch_size : int
        Batch size for training.
    seed : int
        Seed for reproducibility.

    Yields
    ------
    (img_batch, mask_batch) : Tuple[np.ndarray, np.ndarray]
        Augmented image and mask batches.
    """
    image_datagen = ImageDataGenerator(
        width_shift_range=0.03,
        height_shift_range=0.015,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant',
    )

    mask_datagen = ImageDataGenerator(
        width_shift_range=0.03,
        height_shift_range=0.015,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant',
        preprocessing_function=custom_preprocessing_masks,
    )

    image_generator = image_datagen.flow(
        image_array,
        batch_size=batch_size,
        seed=seed,
    )

    mask_generator = mask_datagen.flow(
        mask_array,
        batch_size=batch_size,
        seed=seed,
    )

    while True:
        img_batch = next(image_generator)
        mask_batch = next(mask_generator)
        yield img_batch, mask_batch


def add_noise(image: np.ndarray) -> np.ndarray:
    """
    Add low-level Gaussian noise to an image, keeping values in [0, 1].
    """
    noise_std = np.random.uniform(0, 0.02)  # updated from 0.01 on 090225
    noise = np.random.normal(loc=0.0, scale=noise_std, size=image.shape)
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image.astype(np.float32)


def adjust_brightness(image: np.ndarray) -> np.ndarray:
    """
    Apply a simple affine brightness/contrast adjustment, clipped to [0, 1].
    """
    brightness_factor_1 = np.random.uniform(0.8, 1.2)   # updated from 0.9–1.1 on 090225
    brightness_factor_2 = np.random.uniform(-0.15, 0.15)  # updated on 090225
    adjusted_image = np.clip((image * brightness_factor_1) + brightness_factor_2, 0, 1)
    return adjusted_image.astype(np.float32)


train_generator = combined_generator(images, masks, batch_size=BATCH_SIZE, seed=SEED)
val_generator = combined_generator(val_images, val_masks, batch_size=BATCH_SIZE, seed=SEED)

# --------------------------------------------------------------------
# Write Augmented Dataset to Disk
# --------------------------------------------------------------------

# Training data
print('Saving training data...')
aug_count = 0
for img_batch, mask_batch in train_generator:
    for k in range(len(img_batch)):
        # Apply noise and brightness only to the image patches.
        # Doing this outside ImageDataGenerator avoids misalignment with masks.
        img = add_noise(img_batch[k])
        img = adjust_brightness(img)

        tifff.imwrite(os.path.join(TRAIN_IMG_DIR, f'aug_{aug_count}.tif'), img)
        tifff.imwrite(os.path.join(TRAIN_MASK_DIR, f'aug_{aug_count}.tif'), mask_batch[k])
        aug_count += 1

    if aug_count >= 1600:
        break

    print(f"Saved {aug_count} training patches")

# Validation data
print('Saving validation data...')
aug_count = 0
for img_batch, mask_batch in val_generator:
    for k in range(len(img_batch)):
        tifff.imwrite(os.path.join(VAL_IMG_DIR, f'aug_{aug_count}.tif'), img_batch[k])
        tifff.imwrite(os.path.join(VAL_MASK_DIR, f'aug_{aug_count}.tif'), mask_batch[k])
        aug_count += 1

    if aug_count >= 400:
        break

    print(f"Saved {aug_count} validation patches")

# --------------------------------------------------------------------
# END
# --------------------------------------------------------------------
