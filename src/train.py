import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from model import create_model

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10

# ⚠️ Update this path if dataset location changes
DATASET_PATH = "/tmp/tmpxs769cm6/Training"

def train():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.3,
        brightness_range=[0.7, 1.3],
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )

    val_gen = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    model = create_model()

    model.compile(
        optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[
            EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2)
        ]
    )

    model.save("brain_tumor_detector.keras")

    return history


if __name__ == "__main__":
    train()
