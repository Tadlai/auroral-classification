from matplotlib import pyplot as plt
from tqdm import tqdm
from extractor import Extractor
import numpy as np
import cv2

from preprocessor import Preprocessor


class PolarHist(Extractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name = "PolarHist"

    def train_tf(self, train_ds, valid_ds, n_samples, valid_len, input_data_path=None):
        print("This extractor cannot be trained !")
        return None, None

    def polar_hist(self, image):
        image = image.astype(np.float32)
        value = np.sqrt(((image.shape[0] / 2.0) ** 2.0) + ((image.shape[1] / 2.0) ** 2.0))
        image = cv2.warpPolar(image, center=(image.shape[0] / 2, image.shape[1] / 2), dsize=np.shape(image[:, :, 0]),
                              maxRadius=value,
                              flags=cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG)
        image = image[:, :240]
        Ir = np.sum(image, axis=0)
        Itheta = np.sum(image, axis=1)
        Ir = np.mean(Ir, axis=1) / np.sum(Ir)
        Itheta = np.mean(Itheta, axis=1) / np.sum(Itheta)
        Ir = np.expand_dims(Ir, axis=-1)
        Itheta = np.expand_dims(Itheta, axis=-1)
        return np.concatenate((Ir.T, Itheta.T), axis=1)

    def encode(self, batch):
        preds = []
        for image in batch:
            pred = self.polar_hist(image)
            preds.append(pred)
        p = np.array(preds)
        p = p.reshape((len(preds), preds[0].shape[1]))
        return p

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
            preds.reshape((np.shape(preds)[0], np.prod(np.shape(preds)) // np.shape(preds)[0]))
            encoded.append(preds)
            enc_names.append(names)
        enc = np.concatenate(encoded, axis=0)
        names = np.concatenate(enc_names, axis=0)

        print("Saving features array (shape: " + str(enc.shape) + ")")
        np.save(output_path, enc)
        np.save(output_path[:-4] + '_names.npy', names)


if __name__ == "__main__":
    img = cv2.imread("/home/vincent/DeepLearning/Season_2019-2020/clean_dataset/LYR-Sony-010119_141333-ql.jpg")
    pl = PolarHist()
    pr = Preprocessor()
    img = pr.preprocess(img)
    vect = pl.polar_hist(img)
    print(vect.shape)
