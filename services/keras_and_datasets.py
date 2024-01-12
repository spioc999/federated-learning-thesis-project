import tensorflow as tf
import numpy as np

INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10

def get_model() -> tf.keras.Model:
    """Constructs a simple model architecture suitable for MNIST."""
    """
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    """
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=INPUT_SHAPE),
            tf.keras.layers.Conv2D(10, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def load_datasets(num_clients):
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    train_partition_size = len(x_train) // num_clients
    test_partition_size = len(x_test) // num_clients

    x_train_datasets, y_train_datasets, x_test_datasets, y_test_datasets = [], [], [], []

    for i in range(num_clients):
      start_index_train, start_index_test = i * train_partition_size, i * test_partition_size
      end_index_train, end_index_test = (i+1) * train_partition_size, (i+1) * test_partition_size

      x_train_datasets.append(x_train[start_index_train:end_index_train])
      y_train_datasets.append(y_train[start_index_train:end_index_train])

      x_test_datasets.append(x_test[start_index_test:end_index_test])
      y_test_datasets.append(y_test[start_index_test:end_index_test])

    return x_train_datasets, y_train_datasets, x_test_datasets, y_test_datasets