
import numpy as np
import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Conv2D, BatchNormalization, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import regularizers


class ClassifierModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(ClassifierModel, self).__init__()
        num_classes = 10
        N_LAYERS = 3
        FILTER_LENGTH = 5
        CONV_FILTER_COUNT = 56
        BATCH_SIZE = 32
        LSTM_COUNT = 96
        EPOCH_COUNT = 70
        NUM_HIDDEN = 64
        L2_regularization = 0.001
        # Optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.learning_rate, momentum=hp.momentum
        )

        self.architecture = [
            # Conv2D(32, 3, 2, padding="same", activation="relu", name="block1_conv1"),
            # BatchNormalization(momentum=0.9),
           
            # MaxPool2D(2, name="block1_pool"),
            # Dropout(0.4),

            # Conv2D(32, 3, 2, padding="same", activation="relu", name="block1_conv1"),
            # BatchNormalization(momentum=0.9),
            
            # MaxPool2D(2, name="block1_pool"),
            # Dropout(0.4),

            # Conv2D(32, 3, 2, padding="same", activation="relu", name="block1_conv1"),
            # BatchNormalization(momentum=0.9),
            
            # MaxPool2D(2, name="block1_pool"),
            # Dropout(0.4),

            # LSTM(96, return_sequences=False),

            # Flatten(), Dense(500, activation="relu"), Dense(10, activation="softmax")
            Conv2D(32, 3, 2, padding="same", activation="relu", name="block1_conv1"),
            Conv2D(32, 3, 1, padding="same", activation="relu", name="block1_conv2"),
            BatchNormalization(momentum=0.9),
            MaxPool2D(2, name="block1_pool"),
            Dropout(0.4),
            # Block 2
            Conv2D(64, 3, 1, padding="same", activation="relu", name="block2_conv1"),
            Conv2D(64, 3, 1, padding="same", activation="relu", name="block2_conv2"),
            BatchNormalization(momentum=0.9),
            MaxPool2D(2, name="block2_pool"),
            Dropout(0.4),
            # Block 3 
            Conv2D(128, 3, 1, padding="same", activation="relu", name="block3_conv1"),
            Conv2D(128, 3, 1, padding="same", activation="relu", name="block3_conv2"),
            MaxPool2D(2, name="block3_pool"), Dropout(0.25),
            Flatten(), Dense(500, activation="relu"), Dense(10, activation="softmax")
            ]


        # ====================================================================

    def call(self, img):
        """ Passes input image through the network. """

        for layer in self.architecture:
            # print(img)
            img = layer(img)

        return img

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        return tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions, from_logits=False
        )
