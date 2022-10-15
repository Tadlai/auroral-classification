import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from preprocessor import Preprocessor

cosine_sim_1d = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
cosine_sim_2d = tf.keras.losses.CosineSimilarity(axis=2, reduction=tf.keras.losses.Reduction.NONE)


def _cosine_simililarity_dim1(x, y):
    v = cosine_sim_1d(x, y)
    return v


def _cosine_simililarity_dim2(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    v = cosine_sim_2d(tf.expand_dims(x, 1), tf.expand_dims(y, 0))
    return v


def _dot_simililarity_dim1(x, y):
    # x shape: (N, 1, C)
    # y shape: (N, C, 1)
    # v shape: (N, 1, 1)
    v = tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2))
    return v


def _dot_simililarity_dim2(x, y):
    v = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)
    # x shape: (N, 1, C)
    # y shape: (1, C, 2N)
    # v shape: (N, 2N)
    return v


def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)


# Augmentation utilities (differs from the original implementation)
# Referred from: https://arxiv.org/pdf/2002.05709.pdf (Appendxi A
# corresponding GitHub: https://github.com/google-research/simclr/)

class CustomAugment(object):

    def __call__(self, sample):
        samples1 = []
        samples2 = []
        for image in sample:
            image1 = self._random_apply(tf.image.flip_up_down, image, p=0.5)
            image1 = self._random_apply(tf.image.flip_left_right, image1, p=0.5)
            image1 = self._random_apply(self._crop_resize, image1, p=0.5)
            #image1 = self._random_apply(self._color_jitter, image1, p=0.8)
            #image1 = self._random_apply(self._color_drop, image1, p=0.2)
            samples1.append(image1)

            image2 = self._random_apply(tf.image.flip_up_down, image, p=0.5)
            image2 = self._random_apply(tf.image.flip_left_right, image2, p=0.5)
            image2 = self._random_apply(self._crop_resize, image2, p=0.5)
            #image2 = self._random_apply(self._color_jitter, image2, p=0.8)
            #image2 = self._random_apply(self._color_drop, image2, p=0.2)
            samples2.append(image2)

        sample1 = tf.stack(samples1)
        sample2 = tf.stack(samples2)
        return sample1, sample2

    def _color_jitter(self, x, s=0.5):
        # one can also shuffle the order of following augmentations
        # each time they are applied.
        x = tf.image.random_brightness(x, max_delta=0.8 * s)
        x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
        x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
        x = tf.image.random_hue(x, max_delta=0.2 * s)
        x = tf.clip_by_value(x, 0, 1)
        return x

    def _color_drop(self, x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [ 1, 1, 3])
        return x

    def _crop_resize(self, x):
        initial_shape = x.shape
        p = tf.random.uniform(shape=(2,), minval=0.3, maxval=0.9)
        x = tf.image.random_crop(x, size=(x.shape[0] * p[0], x.shape[1] * p[1], 3))
        x = tf.image.resize(x, tf.convert_to_tensor(initial_shape[0:2], dtype=tf.int32))
        return x

    def _random_apply(self, func, x, p):
        return tf.cond(
            tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                    tf.cast(p, tf.float32)),
            lambda: func(x),
            lambda: x)


if __name__ == "__main__":
    img = cv2.imread("/home/vincent/DeepLearning/Season_2019-2020/clean_dataset/LYR-Sony-010119_141333-ql.jpg")
    pr = Preprocessor()
    img = pr.preprocess(img)
    c = CustomAugment()
    img = np.expand_dims(img, axis=0)
    plt.figure()
    for i in range(50):
        plt.subplot(1, 50, i + 1)
        aug_img = c(img)
        plt.imshow(aug_img[0, :, :, :])
        plt.axis('off')
    plt.show()
