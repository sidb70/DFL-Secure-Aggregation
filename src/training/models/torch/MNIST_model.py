

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import logging
import numpy as np


class DigitClassifier:
    def __init__(self, epochs: int, batch_size: int, num_samples: int, node_hash: int, logger: logging.Logger):
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.node_hash = node_hash
        self.logger = logger
        self.losses = {'train': [], 'validation': []}
        self.load_data()
        self.build_model()

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        np.random.seed(self.node_hash)
        permutation = np.random.permutation(x_train.shape[0])
        x_train, y_train = x_train[permutation], y_train[permutation]

        if self.num_samples > 0:
            x_train, y_train = x_train[:self.num_samples], y_train[:self.num_samples]

        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]

        self.train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000,
                                                                                       seed=self.node_hash).batch(
            self.batch_size)
        self.val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.batch_size)

        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def build_model(self):
        self.model = Sequential([
            layers.Rescaling(1. / 255, input_shape=(28, 28, 1)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10)
        ])

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self):
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs
        )

        self.losses['train'] = history.history['loss']
        self.losses['validation'] = history.history['val_loss']

        self.logger.info('Training complete')

    def plot_losses(self):
        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, self.losses['train'], label='Training Loss')
        plt.plot(epochs_range, self.losses['validation'], label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()



classifier = DigitClassifier(epochs=5, batch_size=128, num_samples=1000, node_hash=42, logger=logger)
classifier.train()
classifier.plot_losses()
