# tests/test_model_build.py

import numpy as np
import tensorflow as tf

from battery_xct_workflows.models.overhangs_unet_segmentations.unet_resnet_overhangs import (
    build_overhangs_model,
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_CHANNELS,
)


def test_model_builds_and_is_keras_model():
    """Model builder should return a compiled Keras Model."""
    model = build_overhangs_model()
    assert isinstance(model, tf.keras.Model)
    # Check that the model has an input and output tensor
    assert len(model.inputs) == 1
    assert len(model.outputs) == 1


def test_model_forward_pass_output_shape():
    """
    A dummy batch should run through the model and produce an output
    with the expected spatial dimensions and number of channels.
    """
    model = build_overhangs_model()

    # Build a dummy input batch
    batch_size = 2
    input_shape = (batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    x = np.random.rand(*input_shape).astype("float32")

    # Forward pass
    y = model(x, training=False)

    # y may be a Tensor or NumPy array depending on execution mode
    if isinstance(y, tf.Tensor):
        y = y.numpy()

    # For classes=1, activation='sigmoid' we expect a single-channel output
    assert y.shape[0] == batch_size
    assert y.shape[1] == IMG_HEIGHT
    assert y.shape[2] == IMG_WIDTH
    assert y.shape[3] == 1  # single output channel for binary segmentation

    # Check outputs are in [0, 1] range due to sigmoid
    assert y.min() >= 0.0
    assert y.max() <= 1.0

