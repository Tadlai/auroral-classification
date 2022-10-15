import shutil

import cv2
import joblib
import numpy as np
import os
import glob
import tqdm
import tensorflow as tf
from matplotlib import pyplot as plt

from helper import TooDarkException
from machine import Machine
from path_dataclass import Paths


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _ndarray_feature(ndarray):
    tensor = tf.convert_to_tensor(ndarray, dtype=ndarray.dtype)
    string_tensor = tf.io.serialize_tensor(tensor)
    return _bytes_feature(string_tensor)


class Preprocessor(Machine):
    def __init__(self, ksize=5, threshold_dark=50, quantile_dark=0.9, crop_per_amount=0.5, crop_margin=5,
                 output_size=(256, 256), mode="full", data_path='', **kwargs):
        super().__init__(**kwargs)
        self._name = "Preprocessor"
        self.output_size = output_size
        self.crop_per_amount = crop_per_amount
        self.ksize = ksize
        self.quantile_dark = quantile_dark
        self.threshold_dark = threshold_dark
        self.crop_margin = crop_margin
        self.mode = mode
        self.data_path = data_path


    def filter_stars(self, image):
        # Blur for suppressing stars pixels
        image = cv2.medianBlur(image, self.ksize)
        return image

    def is_too_dark(self, image):
        # Filter out the images that are too dim
        image = image.astype('uint8')
        imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        maxi = np.nanquantile(imgHSV[:, :, 2], self.quantile_dark)
        if maxi < self.threshold_dark:
            return True
        else:
            return False

    def img_to_LAB(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return image.astype("float32")

    def img_to_HSV(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return image.astype("float32")

    def HSV_to_img(self, image):
        image = image.astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image

    def LAB_to_img(self, image):
        image = image.astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        return image

    def equalize_channel(self, image, num_channel):
        image = image.astype("float32")
        image[:, :, num_channel] = 255 * (image[:, :, num_channel] - np.nanmin(image[:, :, num_channel])) / (
                np.nanmax(image[:, :, num_channel]) - np.nanmin(image[:, :, num_channel]))
        # q1 = np.nanquantile(image[:, :, num_channel], 0.01)
        # image[:, :, num_channel] = image[:, :, num_channel] - q1
        # q99 = np.nanquantile(image[:, :, num_channel], 0.99)
        # image[:, :, num_channel] = image[:, :, num_channel] / q99
        # image[:, :, num_channel] = np.clip(image[:, :, num_channel], 0, 1)
        # image[:, :, num_channel] = image[:, :, num_channel] * 255
        return image

    def center_channel(self, image, num_channel):
        image[:, :, num_channel] = np.clip(image[:, :, num_channel] - np.nanmedian(image[:, :, num_channel]) + 128,
                                           0, 255)
        return image

    def crop_percentiles(self, image, num_channel):
        # compress the low and high luminance values
        q1 = np.nanpercentile(image[:, :, num_channel].flatten(), self.crop_per_amount)
        q99 = np.nanpercentile(image[:, :, num_channel].flatten(), 100 - self.crop_per_amount)
        image[:, :, num_channel] = np.clip(image[:, :, num_channel], q1, q99)
        return image

    def zero_to_nan(self, image):
        image[image <= 0.01] = np.nan
        return image

    def nan_to_zero(self, image):
        image[image == np.nan] = 0
        return image

    def crop_circle(self, image):
        ind_crop = 90 + self.crop_margin
        image = image[ind_crop:-ind_crop, ind_crop:-ind_crop, :]
        return image

    def remove_bottom_text(self, image):
        image = image[:480, :, :]
        return image

    def resize(self, image):
        image = cv2.resize(image, self.output_size)
        return image

    def normalize(self, image):
        image = image.astype(np.float32) / 255.
        return image

    def full_preprocess_chain(self, image):
        image = self.remove_bottom_text(image)

        if self.is_too_dark(image):
            raise TooDarkException("Image is too dim to be processed.")
        imageLAB = self.img_to_LAB(image)
        imageLAB = self.zero_to_nan(imageLAB)
        imageLAB = self.crop_percentiles(imageLAB, num_channel=0)
        # imageLAB = self.center_channel(imageLAB, num_channel=0)
        imageLAB = self.center_channel(imageLAB, num_channel=1)
        imageLAB = self.center_channel(imageLAB, num_channel=2)
        imageLAB = self.crop_circle(imageLAB)
        imageLAB = self.equalize_channel(imageLAB, num_channel=0)

        imageLAB = self.center_channel(imageLAB, num_channel=0)

        image = self.LAB_to_img(imageLAB)
        image = self.filter_stars(image)
        image = self.resize(image)
        image = self.normalize(image)
        return image

    def minimal_preprocess(self, image):
        # For pretrained networks with built-in preprocessings
        image = self.remove_bottom_text(image)

        if self.is_too_dark(image):
            raise TooDarkException("Image is too dim to be processed.")
        imageLAB = self.img_to_LAB(image)
        imageLAB = self.zero_to_nan(imageLAB)
        imageLAB = self.crop_percentiles(imageLAB, num_channel=0)
        imageLAB = self.crop_circle(imageLAB)
        image = self.LAB_to_img(imageLAB)
        image = self.filter_stars(image)
        image = self.resize(image)
        return image.astype(np.float32)

    def preprocess(self, image, exclude_dark=True):
        old_thresh = self.threshold_dark
        processed_img = None
        if not exclude_dark:
            self.threshold_dark = 0
        if self.mode == "full":
            processed_img = self.full_preprocess_chain(image)
        if self.mode == "minimal":
            processed_img = self.minimal_preprocess(image)

        self.threshold_dark = old_thresh
        return processed_img

    def save_data(self):
        # Save preprocessed data into another dir
        copy_path = os.path.join(self.data_dir, "tfrecords")
        if not os.path.isdir(copy_path):
            shutil.copytree(os.path.join(self.data_path, "tfrecords"), copy_path)
            print("Preprocessed images copied!")
        else:
            print("tfrecords files have already been saved !")





class TFWriter:
    def __init__(self, data_path, preprocessor):
        self.data_path = data_path
        self.aurora_path = os.path.join(self.data_path, "clean_dataset")
        self.dest_path = os.path.join(self.data_path, "tfrecords")
        self.preprocessor = preprocessor

    def write_directory(self, remove=True):
        list_samples = os.listdir(self.aurora_path)
        if remove:
            print("Removing existing files...")
            files = glob.glob(os.path.join(self.dest_path, "*"))
            for f in files:
                os.remove(f)
        print("Writing new files...")
        unprocessed_count = 0
        for image in tqdm.tqdm(list_samples):
            unprocessed_count += self.write_tfrecords(image)
        print(str(unprocessed_count) + " files out of " + str(len(list_samples)) + " were not written.")

    def write_tfrecords(self, img_name):
        try:
            img = cv2.imread(os.path.join(self.aurora_path, img_name))
            if img is None:
                return 1
            prep_img = self.preprocessor.preprocess(img)
            feature = {
                'image': _ndarray_feature(prep_img),
                'name': _bytes_feature(tf.io.serialize_tensor(img_name[:-4]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            record_file = os.path.join(self.dest_path, img_name[:-4] + ".tfrecords")
            with tf.io.TFRecordWriter(record_file) as writer:
                writer.write(example.SerializeToString())
            return 0
        except TooDarkException as e:
            return 1
            # print('File: ' + img_name + ' could not be processed:' + str(e))


class TFReader:
    def __init__(self, data_path="/home/vincent/DeepLearning/Season_2019-2020/clean_dataset"):
        self.data_path = data_path
        self.input_path = os.path.join(self.data_path, "tf_auroras")

    @tf.function
    def read_tfrecords(self, file_names):
        # Returns the two features of a tfrecords file: the image array, and the mask array
        dataset = tf.data.TFRecordDataset(filenames=file_names)

        # Create a dictionary describing the features.
        image_feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'name': tf.io.FixedLenFeature([], tf.string)
        }

        def _parse_image_function(example_proto):
            # Parse the input tf.train.Example proto using the dictionary above.
            example = tf.io.parse_single_example(example_proto, image_feature_description)
            tensor_image = tf.io.parse_tensor(example["image"], tf.float32)
            tensor_name = tf.io.parse_tensor(example["name"], tf.string)
            return tensor_image, tensor_name

        dataset = dataset.map(_parse_image_function)

        return dataset

    def read_directory(self):
        file_names = os.listdir(self.input_path)
        file_paths = [os.path.join(self.input_path, name) for name in file_names]
        dataset = self.read_tfrecords(file_paths)
        return dataset


if __name__ == '__main__':
    paths = Paths()
    img = cv2.imread("/home/vincent/DeepLearning/Season_2019-2020/clean_dataset/LYR-Sony-090119_205940-ql.jpg")
    p = Preprocessor(ksize=5,crop_margin=-10, output_size=(224, 224), mode="full", data_path=paths.root)
    imgp =  p.preprocess(img)
    plt.figure()
    plt.imshow(img)
    plt.figure()
    plt.imshow(imgp)
    plt.show()