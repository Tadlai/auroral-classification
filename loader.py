import tensorflow as tf
import os
import numpy as np

from machine import Machine


@tf.function
def same_shape(img, name):
    img = tf.reshape(img, shape=(tf.shape(img)[0], tf.shape(img)[1], 3))
    return img, name


@tf.function
def duplicate_img(img, name):
    return img, img


@tf.function
def get_single_image(img, name):
    return img


class Loader(Machine):
    def __init__(self, train_proportion=None, batch_size=None, buffer_size=None, n_samples=None, n_test_samples=None):
        super().__init__()
        self._name = "Loader"
        self.train_proportion = train_proportion
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.n_samples = n_samples
        self.n_test_samples = n_test_samples

    @tf.function
    def optimize(self, train, valid):
        # Optimization options :
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train = train.map(same_shape, num_parallel_calls=AUTOTUNE)
        # train = train.cache()
        train = train.shuffle(self.buffer_size)
        train = train.batch(self.batch_size, drop_remainder=True)
        # train = train.repeat()
        train = train.prefetch(buffer_size=AUTOTUNE)

        valid = valid.map(same_shape)
        valid = valid.batch(self.batch_size, drop_remainder=True)

        return train, valid

    @tf.function
    def optimize_test(self, test):
        test = test.map(same_shape)
        test = test.batch(self.batch_size)
        return test

    def train_valid_split(self, tfreader, input_path):
        file_names = os.listdir(input_path)
        file_paths = [os.path.join(input_path, name) for name in file_names]
        # train test validation split
        file_paths = file_paths[
                     :min(self.n_samples, len(file_paths) - self.n_test_samples)]  # avoid dataset overlapping
        np.random.shuffle(file_paths)
        train_paths = file_paths[:int(self.train_proportion * len(file_paths))]
        valid_paths = file_paths[int(self.train_proportion * len(file_paths)):]

        train = tfreader.read_tfrecords(train_paths)
        valid = tfreader.read_tfrecords(valid_paths)

        train, valid = self.optimize(train, valid)

        return train, valid, len(train_paths), len(valid_paths)

    def load_test(self, tfreader, input_path):
        file_names = os.listdir(input_path)
        file_paths = [os.path.join(input_path, name) for name in file_names]
        test_paths = file_paths[-self.n_test_samples:]
        np.random.shuffle(test_paths)
        test = tfreader.read_tfrecords(test_paths)
        test = self.optimize_test(test)
        return test
