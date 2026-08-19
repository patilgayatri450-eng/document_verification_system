"""
train_cnn.py
------------
Trains a custom CNN from scratch to classify document images as
GENUINE vs FORGED (or "tampered").

Expected folder layout (create this yourself from MIDV-500/FMIDV/
your own scanned samples — see README.md):

    data/train/genuine/*.jpg
    data/train/forged/*.jpg
    data/val/genuine/*.jpg
    data/val/forged/*.jpg

Run:
    python models/train_cnn.py
Output:
    models/cnn_model.h5
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "data" / "train"
VAL_DIR = BASE_DIR / "data" / "val"
MODEL_OUT = Path(__file__).resolve().parent / "cnn_model.h5"


def build_cnn(input_shape=(224, 224, 3), num_classes=2):
    model = models.Sequential([
        layers.Rescaling(1. / 255, input_shape=input_shape),

        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int"
    )
    # class_names[0] should be "forged", class_names[1] "genuine" (alphabetical)
    print("Class mapping:", train_ds.class_names)

    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomBrightness(0.1),
        layers.RandomContrast(0.1),
    ])
    train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y))

    AUTOTUNE = tf.data.AUTOTUNE
    return train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE)


def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"{TRAIN_DIR} not found. Populate data/train/genuine and "
            f"data/train/forged before training (see README.md)."
        )

    train_ds, val_ds = get_datasets()
    model = build_cnn()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_OUT), save_best_only=True),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    model.save(MODEL_OUT)
    print(f"Saved CNN model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
