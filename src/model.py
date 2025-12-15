import tensorflow as tf
from tensorflow.keras.regularizers import l2

IMAGE_SIZE = 224
NUM_CLASSES = 4

def create_model():
    input_layer = tf.keras.layers.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_tensor=input_layer,
        include_top=False,
        weights="imagenet"
    )

    for layer in base_model.layers[-10:]:
        layer.trainable = True

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(
        256, activation="relu",
        kernel_regularizer=l2(0.001)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    output = tf.keras.layers.Dense(
        NUM_CLASSES, activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs=input_layer,
        outputs=output
    )

    return model
