import copy
import os
import shutil
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, \
    normalized_mutual_info_score, adjusted_mutual_info_score
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN, OPTICS, Birch, AffinityPropagation, AgglomerativeClustering, MeanShift
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from cop_kmeans import cop_kmeans
from machine import Machine
from reducer import Reducer, PCAReducer


class Grouper(Machine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name = "Grouper"
        self.model = None
        self.trained = False
        self.n_clusters = 0
        self.defined_n = False
        self.labels = None

    def fit_kneighbors(self, features):
        self.neigh = KNeighborsClassifier()
        self.neigh.fit(features, self.labels)

    def train(self, features, must_link, cannot_link):
        print("Beginning grouping...")
        self.model.fit(features)
        self.labels = np.array(self.model.labels_)
        self.trained = True
        self.fit_kneighbors(features)

    def find_groups(self, features, filename, must_link=None, cannot_link=None):
        self.train(features, must_link, cannot_link)
        scores = [0, 0, 0]
        try:
            scores[0] = silhouette_score(features, self.labels)
            scores[1] = calinski_harabasz_score(features, self.labels)
            scores[2] = davies_bouldin_score(features, self.labels)
        except ValueError as v:
            print(v)
            scores = [0, 0, 0]
        np.save(filename, self.labels)
        with open('scores.txt', 'w') as f:
            f.write(str(scores[0]) + ',' + str(scores[1]) + ',' + str(scores[2]) + '\n')
            f.write('Silhouette_score: ' + str(scores[0]) + '\n')
            f.write('Calinski_Harabasz_score: ' + str(scores[1]) + '\n')
            f.write('Davies_Bouldin_score: ' + str(scores[2]) + '\n')
        return scores

    def find_nclusters(self, features, nmin, nmax, nstep, must_link=None, cannot_link=None, show=True):
        if self.defined_n:
            scores = []
            scoredb = []
            scorech = []
            stabilities_mean = []
            stabilities_std = []
            for nclusters in range(nmin, nmax, nstep):
                self.model.n_clusters = nclusters
                self.train(features, must_link, cannot_link)
                score = silhouette_score(features, self.labels)
                scored = davies_bouldin_score(features, self.labels)
                scorec = calinski_harabasz_score(features, self.labels)
                mean, std, _, __ = self.measure_stability(features, 10, show=False, ml=must_link, cl=cannot_link)
                scores.append(score)
                scoredb.append(scored)
                scorech.append(scorec)
                stabilities_mean.append(mean)
                stabilities_std.append(std)

            stabilities_mean = np.array(stabilities_mean)
            stabilities_std = np.array(stabilities_std)

            plt.figure()
            plt.plot(range(nmin, nmax, nstep), scores, label="Silhouette score")
            plt.xlabel('Number of clusters')
            plt.legend()
            plt.savefig('sil.png')
            plt.figure()
            plt.plot(range(nmin, nmax, nstep), scorech, label="Calinsky-Harabasz score")
            plt.xlabel('Number of clusters')
            plt.legend()
            plt.savefig('calh.png')
            plt.figure()
            plt.plot(range(nmin, nmax, nstep), scoredb, label="Davies-Bouldin score")
            plt.xlabel('Number of clusters')
            plt.legend()
            plt.savefig('davb.png')
            plt.figure()
            plt.plot(range(nmin, nmax, nstep), stabilities_mean, label="mean stability")
            plt.plot(range(nmin, nmax, nstep), stabilities_mean + stabilities_std, label="std+")
            plt.plot(range(nmin, nmax, nstep), stabilities_mean - stabilities_std, label="std-")
            plt.xlabel('Number of clusters')
            plt.legend()
            plt.savefig('stab.png')

            ks = (np.argmax(scores)) * nstep + nmin
            kdb = (np.argmin(scoredb)) * nstep + nmin
            kch = (np.argmax(scorech)) * nstep + nmin  # does not seem to work
            kst = (np.argmax(stabilities_mean)) * nstep + nmin
            print(ks, kdb, kch, kst)
            if show:
                plt.show()
            else:
                np.save('nclust.npy', np.array([ks, kdb, kch, kst]))

            return ks
        else:
            print("Number of clusters cannot be defined")

    def plot_clusters(self, features, reducer=None):
        if reducer is not None:
            features = reducer.train_reduce(features, n_samples=len(features), save=False)
        # df = pd.DataFrame(features[:, :3], columns=["axis1", "axis2", "axis3"])
        df = pd.DataFrame(features[:, :2], columns=["axis1", "axis2"])
        symbols = ["circle" for i in range(max(self.labels) + 2)] + ["cross" for i in range(6)]
        sizes = [4 for i in range(max(self.labels) + 2)] + [14 for i in range(6)]
        # fig = px.scatter_3d(df, x='axis1', y='axis2', z='axis3', color=self.labels)
        colors = [str(label) for label in self.labels] + ['SingleArc'] * 10 + ['DiffuseAurora'] * 8 + [
            'PatchyAurora'] * 14 + ['LargeVortex'] * 12 + ['MultipleArc'] * 13 + ['Corona'] * 16
        colormap = ['#03579b', '#0488d1', '#03a9f4', '#4fc3f7', '#b3e5fc', '#253137',
                    '#455a64', '#607d8b', '#90a4ae', '#cfd8dc', '#19237e', '#303f9f',
                    '#3f51b5', '#7986cb', '#c5cae9', '#4a198c', '#7b21a2', '#9c27b0',
                    '#ba68c8', '#e1bee7', '#88144f', '#c21f5b', '#e92663', '#f06292',
                    '#f8bbd0', '#bf360c', '#e64a18', '#ff5722', '#ff8a65', '#ffccbc',
                    '#f67f17', '#fbc02c', '#ffec3a', '#fff177', '#fdf9c3', '#33691d',
                    '#689f38','#aed581','#ddedc8', '#333333', '#666666', '#999999','#CCCCCC','#FFFFFF']
        fig = px.scatter(df, x='axis1', y='axis2', color=colors,
                         color_discrete_sequence=colormap[:max(self.labels)+2]+['violet', 'blue', 'orange', 'red', 'green', 'brown'])
        for i, d in enumerate(fig.data):
            fig.data[i].marker.symbol = symbols[i]
            fig.data[i].marker.size = sizes[i]
            if i > len(fig.data) - 7:
                fig.data[i].marker.line = dict(width=1, color='black')
        # fig.update_traces(marker_size=6)
        fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
        fig.update_xaxes(linewidth=2, linecolor='black')
        fig.update_yaxes(linewidth=2, linecolor='black')
        fig.write_html("clusters_plot.html")
        fig.show()

    def plot_clusters_elements(self, filepaths, preprocessor, n_aff=6, show=True):

        j = 0
        for i in tqdm(range(self.n_clusters)):
            if i % 50 == 0:
                if i != 0:
                    plt.savefig("cluster_elements" + str(i // 50) + ".png")
                plt.figure(figsize=(100, 100))
                j = 0
            inds1 = np.argwhere(np.array(self.labels) == i)
            inds = copy.copy(inds1)
            # while inds.shape[0] < n_aff:
            #     inds = np.vstack((inds, inds1))
            #     print("WARNING: Some images have been duplicated for display of cluster " + str(i))
            paths = [filepaths[inds[k, 0]] for k in range(inds.shape[0])]
            for p in range(n_aff):

                j += 1
                if p >= inds.shape[0]:
                    continue
                plt.subplot(min(self.n_clusters, 50), n_aff, j)
                img = preprocessor.preprocess(cv2.imread(paths[p]), exclude_dark=False)
                if preprocessor.mode == "minimal":
                    img = preprocessor.normalize(img)
                plt.imshow(img)
                plt.axis('off')
                # plt.title(str(i))
            plt.subplots_adjust(bottom=0, top=1, wspace=0, hspace=0)

        plt.savefig("cluster_elements" + str((i // 50) + 1) + ".png")
        if show:
            plt.show()
        else:
            plt.close('all')
        self.max_savename = (i // 50) + 1

    def predict(self, samples):
        return self.model.predict(samples)

    def get_neighbors(self, sample, train_filepaths, preprocessor):
        dist, inds = self.neigh.kneighbors(sample, return_distance=True)
        paths = np.array(train_filepaths)[inds]
        labels = self.labels[inds]
        imgs = []

        for path_list in paths:
            img_list = []
            for path in path_list:
                img = cv2.imread(path)
                img = preprocessor.preprocess(img, exclude_dark=False)
                img_list.append(img)
            imgs.append(img_list)

        return np.array(imgs), labels

    def compute_dist_matrices(self, labels_list, n_tries):
        dist_matrix1 = np.zeros((n_tries, n_tries))
        dist_matrix2 = np.zeros((n_tries, n_tries))
        for i in tqdm(range(n_tries)):
            for j in range(n_tries):
                if i < j:
                    dist1 = adjusted_rand_score(labels_list[i], labels_list[j])
                    dist2 = adjusted_mutual_info_score(labels_list[i], labels_list[j])
                    dist_matrix1[i, j] = dist1
                    dist_matrix2[i, j] = dist2
                else:
                    dist_matrix1[i, j] = np.nan
                    dist_matrix2[i, j] = np.nan
        return dist_matrix1, dist_matrix2

    def subsample_constraints(self, ml, cl, rand_idx):
        new_ml = []
        for ctuple in ml:
            if ctuple[0] not in rand_idx or ctuple[1] not in rand_idx:
                pass
            else:
                new_ml.append((np.where(rand_idx == ctuple[0])[0][0], np.where(rand_idx == ctuple[1])[0][0]))
        new_cl = []
        for ctuple in cl:
            if ctuple[0] not in rand_idx or ctuple[1] not in rand_idx:
                pass
            else:
                new_cl.append((np.where(rand_idx == ctuple[0])[0][0], np.where(rand_idx == ctuple[1])[0][0]))
        return new_ml, new_cl

    def measure_stability(self, features, n_tries, division=10, show=True, ml=None, cl=None):
        idx = np.arange(0, len(features))
        labels_list = []
        for k in range(n_tries):
            # rand_idx = np.random.choice(idx, len(features) // division, replace=False)
            rand_idx = np.random.choice(idx, len(features) // n_tries, replace=False)
            subsampled_features = features[rand_idx]
            if ml is not None:
                ml, cl = self.subsample_constraints(ml, cl, rand_idx)
            self.train(subsampled_features, ml, cl)
            labels = self.predict(features)
            labels_list.append(labels)
            # idx = np.delete(idx, np.where(idx == rand_idx),axis=0)  # To have different samples for each iteration
            idx = np.setdiff1d(idx, rand_idx)

        dist_matrix1, dist_matrix2 = self.compute_dist_matrices(labels_list, n_tries)

        if show:
            print('Mean/std adjusted rand score: ')
            print(np.nanmean(dist_matrix1))
            print(np.nanstd(dist_matrix1))
            print('Mean/std adjusted mutual info score: ')
            print(np.nanmean(dist_matrix2))
            print(np.nanstd(dist_matrix2))
            plt.matshow(dist_matrix1, vmin=0, vmax=1)
            plt.colorbar()
            plt.title('Adjusted rand score of labels trained on subsampled sets (1:' + str(division) + ')')
            plt.matshow(dist_matrix2, vmin=0, vmax=1)
            plt.colorbar()
            plt.title('Adjusted mutual info score of labels trained on subsampled sets (1:' + str(division) + ')')
            plt.show()

        return np.nanmean(dist_matrix1), np.nanstd(dist_matrix1), np.nanmean(dist_matrix2), np.nanstd(dist_matrix2)

    def measure_reduction_stability(self, encoded_features, n_tries, reducer: Reducer, min_dim=1):
        min_dists = np.linspace(0, 1, n_tries + 2)
        labels_list = []
        sil = []
        for k in tqdm(range(min_dim, n_tries + min_dim)):
            # reducer.n_dim = k
            # reducer.model.n_neighbors = k
            reducer.model.min_dist = min_dists[k]
            # reducer.model.random_state = k * 1234
            reduced_features = reducer.train_reduce(encoded_features, len(encoded_features), save=False)
            self.train(reduced_features, None, None)
            labels_list.append(self.labels)
            sil.append(silhouette_score(reduced_features, self.labels))
            self.plot_clusters(reduced_features, reducer=PCAReducer(3))  # TODO remove this line

        dist_matrix1, dist_matrix2 = self.compute_dist_matrices(labels_list, n_tries)

        plt.plot(np.linspace(min_dim, n_tries + min_dim - 1, n_tries), sil)
        plt.title('Silhouette score of Kmeans according to the number of neighbors in UMAP')
        plt.show()

        plt.matshow(dist_matrix1, vmin=0, vmax=1)
        plt.colorbar()
        plt.title('Adjusted rand score of labels trained on reduced data with different random seeds')
        plt.matshow(dist_matrix2, vmin=0, vmax=1)
        plt.colorbar()
        plt.title('Adjusted mutual info score of labels trained on reduced data with different random seeds')
        plt.show()
        print('Mean adjusted rand score: ')
        print(np.nanmean(dist_matrix1))
        print('Mean normalized mutual info score: ')
        print(np.nanmean(dist_matrix2))

    def save_data(self):
        lbl = os.path.join(self.data_dir, "labels")
        if not os.path.isdir(lbl):
            os.mkdir(lbl)
            np.save(os.path.join(lbl, "group_labels.npy"), self.labels)
            try:
                for i in range(1, self.max_savename + 1):
                    shutil.copy("cluster_elements" + str(i) + ".png", lbl)
                for i in range(6):
                    shutil.copy("ref_neighbors" + str(i) + ".png", lbl)
                shutil.copy("clusters_plot.html", lbl)
                shutil.copy('scores.txt', lbl)
            except Exception as e:
                print(e, ": Display files were not found!")
        else:
            print("Encoded files have already been saved!")

    def add_ref_features(self, ref_dir, chain):
        new_features = copy.copy(chain.reduced_features)
        new_filepaths = copy.copy(chain.filepaths)
        for dir in os.listdir(ref_dir):
            dir_path = os.path.join(ref_dir, dir)
            if os.path.isdir(dir_path):
                list_imgs = os.listdir(dir_path)
                for img_name in list_imgs:
                    if ".jpg" not in img_name:
                        list_imgs.remove(img_name)

                list_paths = [os.path.join(dir_path, img_name) for img_name in list_imgs]
                dic_list = chain.process_imgs(list_paths, predict=False)
                for i in range(len(list_imgs)):
                    img_red = np.expand_dims(dic_list['red_img'][i], axis=0)
                    new_features = np.concatenate((new_features, img_red), axis=0)
                    new_filepaths.append(list_paths[i])
        return new_features, new_filepaths


class ExtendedGrouper(Grouper):
    def predict(self, samples):
        return self.neigh.predict(samples)


class KMeansGrouper(Grouper):
    def __init__(self, n_clusters, **kwargs):
        super().__init__(**kwargs)
        self._name = "KMeansGrouper"
        self.model = KMeans(n_clusters=n_clusters, random_state=0)
        self.n_clusters = n_clusters
        self.defined_n = True


class SpectralGrouper(ExtendedGrouper):  # pas de predict
    def __init__(self, n_clusters, **kwargs):
        super().__init__(**kwargs)
        self._name = "SpectralGrouper"
        self.model = SpectralClustering(n_clusters)
        self.n_clusters = n_clusters
        self.defined_n = True


class DBScanGrouper(ExtendedGrouper):  # pas de predict
    def __init__(self, eps, **kwargs):
        super().__init__(**kwargs)
        self._name = "DBScanGrouper"
        self.model = DBSCAN(eps=eps, min_samples=10)
        self.eps = eps

    def find_groups(self, features, filename, must_link=None, cannot_link=None):
        score = super().find_groups(features, filename)
        self.n_clusters = max(self.model.labels_) + 1
        return score


class OPTICSGrouper(ExtendedGrouper):  # pas de predict
    def __init__(self, min_samples=100, **kwargs):
        super().__init__(**kwargs)
        self._name = "OPTICSGrouper"
        self.model = OPTICS(min_samples=min_samples)

    def find_groups(self, features, filename, must_link=None, cannot_link=None):
        score = super().find_groups(features, filename)
        self.n_clusters = max(self.model.labels_) + 1
        return score


class BirchGrouper(Grouper):
    def __init__(self, n_clusters, **kwargs):
        super().__init__(**kwargs)
        self._name = "BirchGrouper"
        self.model = Birch(n_clusters=n_clusters, threshold=0.001)
        self.n_clusters = n_clusters
        self.defined_n = True


class AggloGrouper(ExtendedGrouper):
    def __init__(self, n_clusters, **kwargs):
        super().__init__(**kwargs)
        self._name = "AggloGrouper"
        self.model = AgglomerativeClustering(n_clusters=n_clusters)
        self.n_clusters = n_clusters
        self.defined_n = True


class AffinityGrouper(Grouper):
    def __init__(self, damping, **kwargs):
        super().__init__(**kwargs)
        self._name = "AffinityGrouper"
        self.damping = damping
        self.model = AffinityPropagation(damping=damping)

    def find_groups(self, features, filename, must_link=None, cannot_link=None):
        score = super().find_groups(features, filename)
        self.n_clusters = max(self.model.labels_) + 1
        return score


class MeanShiftGrouper(Grouper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name = "MeanShiftGrouper"
        self.model = MeanShift()

    def find_groups(self, features, filename, must_link=None, cannot_link=None):
        score = super().find_groups(features, filename)
        self.n_clusters = max(self.model.labels_) + 1
        return score


class MixtureGrouper(Grouper):
    def __init__(self, n_clusters, **kwargs):
        super().__init__(**kwargs)
        self._name = "MixtureGrouper"
        self.model = GaussianMixture(n_components=n_clusters)
        self.n_clusters = n_clusters
        self.defined_n = True


class COPKMeansGrouper(ExtendedGrouper):
    def __init__(self, n_clusters, **kwargs):
        super().__init__(**kwargs)
        self._name = "COPKMeansGrouper"
        self.n_clusters = n_clusters
        self.defined_n = True

    def train(self, features, must_link, cannot_link):
        print("Beginning clustering...")
        self.labels, centers = cop_kmeans(dataset=features, k=self.n_clusters, ml=must_link, cl=cannot_link)
        self.labels = np.array(self.labels)
        self.trained = True
        self.fit_kneighbors(features)


if __name__ == "__main__":
    gr = Grouper()
    gr.get_params()
    k = KMeansGrouper(2)
    k.get_params()
