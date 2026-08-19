"""
train_mobilenetv2.py
--------------------

MobileNetV2 transfer-learning model for:

    0 = forged
    1 = genuine

Dataset:

data/
    train/
        forged/
        genuine/

    val/
        forged/
        genuine/

Output:

models/mobilenetv2_model.keras
"""

import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 16

INITIAL_EPOCHS = 15

FINE_TUNE_EPOCHS = 15

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "data" / "train"

VAL_DIR = BASE_DIR / "data" / "val"

MODEL_OUT = (
    Path(__file__).resolve().parent
    / "mobilenetv2_model.keras"
)


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    print("\nLoading MobileNetV2...")

    base = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )

    # Freeze base initially
    base.trainable = False

    inputs = tf.keras.Input(
        shape=IMG_SIZE + (3,),
        name="document_image"
    )

    # --------------------------------------------------------
    # DOCUMENT-SAFE AUGMENTATION
    # --------------------------------------------------------

    x = layers.RandomRotation(
        0.02
    )(inputs)

    x = layers.RandomZoom(
        height_factor=0.05,
        width_factor=0.05
    )(x)

    x = layers.RandomContrast(
        0.10
    )(x)

    # NO horizontal flip
    #
    # Documents should not be flipped because
    # text/barcodes/signatures become backwards.

    # --------------------------------------------------------
    # MobileNetV2 preprocessing
    # --------------------------------------------------------

    x = preprocess_input(x)

    # --------------------------------------------------------
    # MobileNetV2
    # --------------------------------------------------------

    x = base(
        x,
        training=False
    )

    # --------------------------------------------------------
    # Global pooling
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D()(x)

    # --------------------------------------------------------
    # Dropout
    # --------------------------------------------------------

    x = layers.Dropout(
        0.4
    )(x)

    # --------------------------------------------------------
    # Classification
    #
    # 0 = forged
    # 1 = genuine
    # --------------------------------------------------------

    outputs = layers.Dense(
        2,
        activation="softmax",
        name="classification"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="Document_MobileNetV2"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base


# ============================================================
# LOAD DATASETS
# ============================================================

def get_datasets():

    print("\nLoading training dataset...")

    train_ds = tf.keras.utils.image_dataset_from_directory(

        TRAIN_DIR,

        image_size=IMG_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="int",

        shuffle=True,

        seed=42
    )

    print("\nLoading validation dataset...")

    val_ds = tf.keras.utils.image_dataset_from_directory(

        VAL_DIR,

        image_size=IMG_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="int",

        shuffle=False
    )

    print("\n==========================================")
    print("CLASS MAPPING")
    print("==========================================")

    print(
        "Classes:",
        train_ds.class_names
    )

    print("==========================================")

    expected_classes = [
        "forged",
        "genuine"
    ]

    if train_ds.class_names != expected_classes:

        raise ValueError(

            "\nIncorrect class mapping!\n\n"

            f"Expected:\n"
            f"{expected_classes}\n\n"

            f"Found:\n"
            f"{train_ds.class_names}\n\n"

            "Your folders must be:\n"
            "data/train/forged\n"
            "data/train/genuine\n"
            "data/val/forged\n"
            "data/val/genuine"
        )

    if val_ds.class_names != expected_classes:

        raise ValueError(

            "\nValidation dataset class mapping is incorrect!\n\n"

            f"Expected:\n"
            f"{expected_classes}\n\n"

            f"Found:\n"
            f"{val_ds.class_names}"
        )

    print("\nCorrect mapping confirmed:")

    print(
        "0 = forged"
    )

    print(
        "1 = genuine"
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(
        AUTOTUNE
    )

    val_ds = val_ds.prefetch(
        AUTOTUNE
    )

    return train_ds, val_ds


# ============================================================
# TRAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" STUDENT DOCUMENT VERIFICATION SYSTEM")
    print(" MobileNetV2 Training")
    print("==========================================")

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not TRAIN_DIR.exists():

        raise FileNotFoundError(
            f"\nTraining directory not found:\n{TRAIN_DIR}"
        )

    if not VAL_DIR.exists():

        raise FileNotFoundError(
            f"\nValidation directory not found:\n{VAL_DIR}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_ds, val_ds = get_datasets()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, base = build_model()

    print("\n==========================================")
    print("MODEL")
    print("==========================================")

    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=5,

            restore_best_weights=True,

            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(

            filepath=MODEL_OUT,

            monitor="val_accuracy",

            mode="max",

            save_best_only=True,

            save_weights_only=False,

            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=2,

            min_lr=1e-7,

            verbose=1
        )
    ]

    # ========================================================
    # PHASE 1
    # ========================================================

    print("\n==========================================")
    print("PHASE 1")
    print("Training classification head")
    print("==========================================")

    model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=INITIAL_EPOCHS,

        callbacks=callbacks
    )

    # ========================================================
    # PHASE 2
    # ========================================================

    print("\n==========================================")
    print("PHASE 2")
    print("Fine-tuning MobileNetV2")
    print("==========================================")

    base.trainable = True

    # Freeze all except last 30 layers

    for layer in base.layers[:-30]:

        layer.trainable = False

    # Keep BatchNorm frozen

    for layer in base.layers:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=FINE_TUNE_EPOCHS,

        callbacks=callbacks
    )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print("\nLoading best saved model...")

    best_model = tf.keras.models.load_model(
        MODEL_OUT,
        compile=False
    )

    # ========================================================
    # SAVE
    # ========================================================

    best_model.save(
        MODEL_OUT
    )

    print("\n==========================================")
    print("MODEL SAVED")
    print("==========================================")

    print(
        MODEL_OUT
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    print("\n==========================================")
    print("FINAL EVALUATION")
    print("==========================================")

    best_model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    loss, accuracy = best_model.evaluate(
        val_ds,
        verbose=1
    )

    print(
        f"\nValidation Loss     : {loss:.4f}"
    )

    print(
        f"Validation Accuracy : {accuracy:.4f}"
    )

    print(
        f"Validation Accuracy : {accuracy * 100:.2f}%"
    )

    print("\n==========================================")
    print("TRAINING COMPLETED")
    print("==========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()