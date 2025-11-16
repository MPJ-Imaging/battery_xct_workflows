import os
import logging

# Use tf.keras backend for segmentation_models
os.environ['SM_FRAMEWORK'] = 'tf.keras'

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt
from segmentation_models import Unet
from segmentation_models.losses import DiceLoss
from segmentation_models.metrics import iou_score
import tifffile as tifff

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
IMG_CHANNELS = 3  # model requires RGB
IMG_HEIGHT = 256
IMG_WIDTH = 800

# Paths to training/validation patches
TRAIN_IMAGE_DIR = "data/train/images"
TRAIN_MASK_DIR = "data/train/masks"
VAL_IMAGE_DIR = "data/test/images"
VAL_MASK_DIR = "data/test/masks"

# Training parameters
BATCH_SIZE = 1
EPOCHS = 400
STEPS_PER_EPOCH = 700
SEED = 64

# Model backbone
BACKBONE = 'resnet50'

# --------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------

class TiffDataGenerator(Sequence):
    """
    Keras Sequence for loading .tif image and mask patches in batches.

    This generator reads image and mask files from disk on the fly,
    resizes them to the desired spatial resolution, and returns them
    as NumPy arrays suitable for model.fit().
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        batch_size: int,
        image_size: tuple[int, int],
        shuffle: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        image_dir : str
            Directory containing .tif image patches.
        mask_dir : str
            Directory containing corresponding .tif mask patches.
        batch_size : int
            Number of samples per batch.
        image_size : tuple[int, int]
            Target (height, width) of input images.
        shuffle : bool, optional
            Whether to shuffle file order after each epoch.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle

        # Load file names
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

        assert len(self.image_filenames) == len(self.mask_filenames), (
            "Mismatch between number of images and masks: "
            f"{len(self.image_filenames)} vs {len(self.mask_filenames)}"
        )

        self.indices = np.arange(len(self.image_filenames))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return int(np.floor(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate one batch of data.

        Parameters
        ----------
        index : int
            Index of the batch.

        Returns
        -------
        batch_images : np.ndarray
            Batch of images with shape (B, H, W, C).
        batch_masks : np.ndarray
            Batch of masks with shape (B, H, W, C).
        """
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images: list[np.ndarray] = []
        batch_masks: list[np.ndarray] = []

        for i in batch_indices:
            img_path = os.path.join(self.image_dir, self.image_filenames[i])
            mask_path = os.path.join(self.mask_dir, self.mask_filenames[i])

            # Load .tif images (assumed to already be in [0, 1] if normalised upstream)
            img = tifff.imread(img_path).astype(np.float32)
            mask = tifff.imread(mask_path).astype(np.float32)  # binary / multi-channel masks

            # Resize to target spatial size
            img_resized = tf.image.resize(img, self.image_size)
            mask_resized = tf.image.resize(
                mask,
                self.image_size,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            )

            batch_images.append(img_resized)
            batch_masks.append(mask_resized)

        return np.array(batch_images), np.array(batch_masks)

    def on_epoch_end(self) -> None:
        """Shuffle dataset indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)

# --------------------------------------------------------------------
# Model Definition and Compile
# --------------------------------------------------------------------

def build_overhangs_model(
    backbone: str = BACKBONE,
) -> tf.keras.Model:
    """
    Build and compile the U-Net model for overhang segmentation.

    Parameters
    ----------
    backbone : str, optional
        Encoder backbone name for segmentation_models.Unet.

    Returns
    -------
    model : tf.keras.Model
        Compiled Keras model ready for training.
    """
    # Learning-rate schedule (smooth exponential decay over epochs)
    initial_lr = 1e-6
    target_lr = 1e-7
    target_epochs = EPOCHS

    # Decay per *epoch* so that after ~target_epochs: lr ≈ target_lr
    decay_per_epoch = (target_lr / initial_lr) ** (1.0 / target_epochs)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=STEPS_PER_EPOCH,
        decay_rate=decay_per_epoch,
        staircase=False,  # smooth decay
    )

    # Define model: encoder_weights=None trains from scratch; encoder_freeze=False keeps encoder trainable
    model = Unet(
        backbone,
        encoder_weights=None,
        encoder_freeze=False,
        classes=1,
        activation='sigmoid',
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer,
        loss=DiceLoss,
        metrics=[iou_score],
    )
    return model

# --------------------------------------------------------------------
# Custom Callback for Live Plotting
# --------------------------------------------------------------------

class LiveMetricsPlot(Callback):
    """
    Keras callback for live plotting of validation loss and IOU during training.

    Uses Matplotlib in interactive mode to update two curves (loss and IOU)
    at the end of each epoch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.epochs: list[int] = []
        self.val_losses: list[float] = []
        self.val_ious: list[float] = []

        # Initialise interactive plot
        plt.ion()
        self.fig, self.ax1 = plt.subplots()

        # Validation Loss axis
        (self.line1,) = self.ax1.plot([], [], 'b-', label='Validation Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Validation Loss', color='b')
        self.ax1.tick_params(axis='y', labelcolor='b')

        # Second y-axis for IOU
        self.ax2 = self.ax1.twinx()
        (self.line2,) = self.ax2.plot([], [], 'r-', label='Validation IOU')
        self.ax2.set_ylabel('Validation IOU', color='r')
        self.ax2.tick_params(axis='y', labelcolor='r')

        # Combined legend
        lines = [self.line1, self.line2]
        labels = [line.get_label() for line in lines]
        self.ax1.legend(lines, labels, loc='upper left')

        self.ax1.set_title('Live Validation Loss and IOU')
        plt.show()

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        val_loss = logs.get('val_loss')
        val_iou = logs.get('val_iou_score')

        if val_loss is None or val_iou is None:
            return

        self.epochs.append(epoch + 1)
        self.val_losses.append(float(val_loss))
        self.val_ious.append(float(val_iou))

        # Update Validation Loss plot
        self.line1.set_xdata(self.epochs)
        self.line1.set_ydata(self.val_losses)
        self.ax1.relim()
        self.ax1.autoscale_view()

        # Update Validation IOU plot
        self.line2.set_xdata(self.epochs)
        self.line2.set_ydata(self.val_ious)
        self.ax2.relim()
        self.ax2.autoscale_view()

        # Redraw figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# --------------------------------------------------------------------
# Training Script
# --------------------------------------------------------------------

if __name__ == "__main__":
    # Create generators
    train_generator = TiffDataGenerator(
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR,
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=True,
    )

    val_generator = TiffDataGenerator(
        VAL_IMAGE_DIR,
        VAL_MASK_DIR,
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=False,
    )

    # Inspect first validation batch
    X_val, y_val = val_generator[0]
    print(f"Validation batch shape: {X_val.shape}, {y_val.shape}")

    # Build and summarise model
    model = build_overhangs_model(backbone=BACKBONE)
    model.summary()

    # Checkpoint: save best model based on validation IOU
    checkpoint = ModelCheckpoint(
        'unet_resnet_overhangs_model_mini2.keras',
        monitor='val_iou_score',
        mode='max',
        verbose=1,
        save_best_only=True,
    )

    live_metrics_plot = LiveMetricsPlot()

    print("Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=STEPS_PER_EPOCH / 10,
        callbacks=[checkpoint, live_metrics_plot],
    )

# --------------------------------------------------------------------
# END
# --------------------------------------------------------------------
