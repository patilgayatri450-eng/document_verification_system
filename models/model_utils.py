"""
model_utils.py
--------------
Loads the trained MobileNetV2 model and predicts document authenticity.

Class mapping:
    0 = forged
    1 = genuine
"""

import numpy as np
import tensorflow as tf
from pathlib import Path


IMG_SIZE = (224, 224)

MODELS_DIR = Path(__file__).resolve().parent

_model_cache = {}


def load_model(model_name="mobilenetv2_model.keras"):

    if model_name in _model_cache:
        return _model_cache[model_name]

    model_path = MODELS_DIR / model_name

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    print(f"Loading model: {model_path}")

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    print(f"Model loaded successfully: {model_name}")
    print(f"Model output shape: {model.output_shape}")

    _model_cache[model_name] = model

    return model


def predict_authenticity(
    image_path,
    model_name="mobilenetv2_model.keras"
):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    model = load_model(model_name)

    # Load image
    img = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE,
        color_mode="rgb"
    )

    # Convert image
    arr = tf.keras.utils.img_to_array(img)

    # IMPORTANT:
    # Do NOT preprocess here because the trained model
    # already contains MobileNetV2 preprocessing.

    arr = np.expand_dims(arr, axis=0)

    predictions = model.predict(
        arr,
        verbose=0
    )[0]

    print("Raw model prediction:", predictions)

    # Make sure model returns exactly two values
    if len(predictions) != 2:
        raise ValueError(
            f"Expected 2 model outputs, got {len(predictions)}"
        )

    forged_probability = float(predictions[0])
    genuine_probability = float(predictions[1])

    # Class mapping
    if genuine_probability >= forged_probability:

        label = "genuine"
        confidence = genuine_probability

    else:

        label = "forged"
        confidence = forged_probability

    result = {
        "label": label,
        "confidence": confidence,
        "forged_probability": forged_probability,
        "genuine_probability": genuine_probability
    }

    print("VISION RESULT:", result)

    return result