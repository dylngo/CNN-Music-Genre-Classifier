

import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
import hyperparameters as hp


class Datasets:
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path):
        self.data_path = data_path

        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}
        self.class_to_idx = {}
        self.img_to_lbl = {}

        # For storing list of classes
        self.classes = [""] * hp.genre_num

        # Setup data generators
        self.train_data = self.get_data(os.path.join(self.data_path, "train/"), True, True)
        self.test_data = self.get_data(os.path.join(self.data_path, "test/"), False, False)

    def get_data(self, path, shuffle, augment):
        """ Returns spectrogram images
        """

        if augment:
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)
        else:
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator()

        # Form image data generator from directory structure
        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(hp.img_size, hp.img_size),
            class_mode="sparse",
            batch_size=hp.batch_size,
            shuffle=shuffle,
        )
        if not bool(self.idx_to_class):
            unordered_classes = []
            for dir_name in os.listdir(path):
                if os.path.isdir(os.path.join(path, dir_name)):
                    unordered_classes.append(dir_name)

            for img_class in unordered_classes:
                self.idx_to_class[data_gen.class_indices[img_class]] = img_class
                self.class_to_idx[img_class] = int(data_gen.class_indices[img_class])
                self.classes[int(data_gen.class_indices[img_class])] = img_class

        return data_gen
