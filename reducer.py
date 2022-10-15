import os
import shutil

import numpy as np
from umap import UMAP
import joblib
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import Isomap, TSNE, MDS

from machine import Machine


class Reducer(Machine):

    def __init__(self, n_dim=None, reducer_path=None, **kwargs):
        super().__init__(**kwargs)
        self._name = "Reducer"
        if reducer_path is not None:
            self.model = joblib.load(reducer_path)
            self.trained = True
            if n_dim is None:
                self._n_dim = int(reducer_path[:3])
            else:
                self._n_dim = n_dim
        else:
            self.model = None
            self.trained = False
            self._n_dim = n_dim

    def str_dim(self):
        if self.n_dim < 10:
            return '00' + str(self.n_dim)
        if 10 <= self.n_dim < 100:
            return '0' + str(self.n_dim)
        if 100 <= self.n_dim < 1000:
            return str(self.n_dim)

    def trainv2(self, features, n_samples, features_path, filename="temp_reducer.txt"):
        self.features_path = features_path
        print("Training reducer...")
        self.model.fit(features[:n_samples, :])
        print("Reducer trained! Saving model...")
        self.trained = True
        # Saving the model
        joblib.dump(self.model, self.str_dim() + '_' + filename)

    def train_reduce(self, features, n_samples, features_path=None, filename="temp_reducer.txt", dump=True,save=True):
        if features_path is not None:
            self.features_path = features_path
        reduced = self.model.fit_transform(features[:n_samples, :])
        self.trained = True
        if dump:
            joblib.dump(self.model, self.str_dim() + '_' + filename)
        if save:
            np.save('reduced_features.npy', reduced)
        return reduced

    def reduce_datav2(self, features, n_samples, save=True):
        if not self.trained:
            raise Exception("Reducer is not trained.")
        print("Reducing data...")
        reduced = self.model.transform(features[:n_samples, :])
        if save:
            np.save('reduced_features.npy', reduced)
        return reduced

    def save_machine(self, dir):
        super().save_machine(dir)
        joblib.dump(self.model, os.path.join(dir, "reducer.txt"))

    def save_data(self):
        shutil.copy('reduced_features.npy',self.data_dir)
        print("Reduced data saved.")

    @property
    def n_dim(self):
        return self._n_dim

    @n_dim.setter
    def n_dim(self,new_dim):
        self._n_dim = new_dim
        if self.model is not None:
            self.model.n_components = new_dim


class UmapReducer(Reducer):
    def __init__(self, n_neighbors=5, min_dist=0.1, n_dim=3, metric="euclidean", **kwargs):
        super().__init__(n_dim=n_dim, **kwargs)
        self._name = "UmapReducer"
        if self.model is None:
            self.model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_dim, metric=metric)
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_dim = n_dim


class PCAReducer(Reducer):
    def __init__(self, n_dim=3, kernel=None, **kwargs):
        super().__init__(n_dim=n_dim, **kwargs)
        self._name = "PCAReducer"
        if self.model is None:
            if kernel is not None:
                self.model = KernelPCA(n_components=n_dim, kernel=kernel)
                self.kernel = kernel
            self.model = PCA(n_components=n_dim)
        self.n_dim = n_dim

    def explained_variance(self):
        vars = self.model.explained_variance_ratio_
        print("Explained variance ratios: ", vars)

    def trainv2(self, features, n_samples, features_path, filename="temp_reducer.txt"):
        super(PCAReducer, self).trainv2(features, n_samples, features_path, filename)
        self.explained_variance()


class SVDReducer(Reducer):
    def __init__(self, n_dim=3, **kwargs):
        super().__init__(n_dim=n_dim, **kwargs)
        self._name = "SVDReducer"
        if self.model is None:
            self.model = TruncatedSVD(n_components=n_dim)
        self.n_dim = n_dim

    def explained_variance(self):
        vars = self.model.explained_variance_ratio_
        print("Explained variance ratios: ", vars)

    def trainv2(self, features, n_samples, features_path, filename="temp_reducer.txt"):
        super(SVDReducer, self).trainv2(features, n_samples, features_path, filename)
        self.explained_variance()


class IsomapReducer(Reducer):
    def __init__(self, n_neighbors=5, n_dim=3, **kwargs):
        super().__init__(n_dim=n_dim, **kwargs)
        self._name = "IsomapReducer"
        if self.model is None:
            self.model = Isomap(n_neighbors=n_neighbors, n_components=n_dim)
        self.n_neighbors = n_neighbors
        self.n_dim = n_dim


class TSNEReducer(Reducer):
    def __init__(self, n_dim=3, perplexity=30, early_exaggeration=12, lr="auto", **kwargs):
        super().__init__(n_dim=n_dim, **kwargs)
        self._name = "TSNEReducer"
        if self.model is None:
            self.model = TSNE(n_components=n_dim, perplexity=perplexity, early_exaggeration=early_exaggeration,
                              learning_rate=lr)
        self.perplexity = 30
        self.early_exaggeration = early_exaggeration
        self.lr = lr
        self.n_dim = n_dim


class Identity(Reducer):
    def __init__(self, n_dim=None, reducer_path=None, **kwargs):
        super().__init__(n_dim=0, reducer_path=None, **kwargs)
        self._name = "IdentityReducer"

    def trainv2(self, features, n_samples, features_path, filename="temp_reducer.txt"):
        self.features_path = features_path
        print("Identity reducer cannot be trained!")

    def train_reduce(self, features, n_samples, features_path=None, filename="temp_reducer.txt", dump=True,save=True):
        print("Identity reducer cannot be trained!")
        return self.reduce_datav2(features, n_samples, save)

    def reduce_datav2(self, features, n_samples, save=True):
        reduced = features[:n_samples, :]
        if save:
            np.save('reduced_features.npy', reduced)
        return reduced


if __name__ == "__main__":
    u = UmapReducer()
    u.get_params()
