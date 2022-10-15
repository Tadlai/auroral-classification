import os
import shutil

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from machine import Machine


class Extractor(Machine):
    def __init__(self, model_path=None, **kwargs):
        super().__init__(**kwargs)
        self._name = "Extractor"
        if model_path is not None:
            self.encoder = tf.keras.models.load_model(model_path)
        else:
            self.encoder = None

    def train_tf(self, train_ds, valid_ds, n_samples, valid_len, input_data_path):
        print("Extractor is an abstract class! Use subclasses instead.")
        return None, None

    def encode(self, batch):
        if self.encoder is None:
            raise Exception("No encoder found.")

        batch_preds = self.encoder.predict(batch)
        batch_preds.reshape((np.shape(batch_preds)[0], np.prod(np.shape(batch_preds)) // np.shape(batch_preds)[0]))
        return batch_preds

    def encode_dataset(self, dataset, dataset_size, batch_size, output_path):
        self.output_path = output_path
        npds = dataset.as_numpy_iterator()
        n_batches = dataset_size // batch_size
        print("Encoding...")
        encoded = []
        enc_names = []
        for k in tqdm(range(n_batches)):
            x_batch, names = next(npds)
            preds = self.encode(x_batch)
            encoded.append(preds)
            enc_names.append(names)
        enc = np.concatenate(encoded)
        names = np.concatenate(enc_names)

        print("Saving features array (shape: " + str(enc.shape) + ")")
        np.save(output_path, enc)
        np.save(output_path[:-4]+'_names.npy', names)

    def save_data(self):
        ef = os.path.join(self.data_dir,"encoded_features")
        if not os.path.isdir(ef):
            os.mkdir(ef)
            shutil.copy(self.output_path, os.path.join(ef,"features.npy"))
            shutil.copy(self.output_path[:-4]+'_names.npy', os.path.join(ef, "filenames.npy"))
        else:
            print("Encoded files have already been saved! Rewriting files...")
            shutil.copy(self.output_path, os.path.join(ef, "features.npy"))
            shutil.copy(self.output_path[:-4] + '_names.npy', os.path.join(ef, "filenames.npy"))






if __name__ == "__main__":
    e = Extractor()
    e.get_params()