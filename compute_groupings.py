import os
import shutil

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from chain import Chain
from ensemble import Ensemble
from extractor import Extractor
from machine_loader import load_machine
from path_dataclass import Paths
from reducer import Reducer, Identity, UmapReducer, IsomapReducer, PCAReducer
from simCLR_extractor import SimCLRExtractor
from preprocessor import TFWriter, TFReader, Preprocessor
from loader import Loader
from grouper import Grouper, SpectralGrouper, DBScanGrouper, OPTICSGrouper, BirchGrouper, AggloGrouper, \
    MeanShiftGrouper, AffinityGrouper, MixtureGrouper, COPKMeansGrouper
from grouper import KMeansGrouper
import numpy as np
import tensorflow as tf
from sklearn.metrics.cluster import contingency_matrix
import matplotlib.pyplot as plt
from pretrained_nets import ResNet, Inception, MobileNet


def compute_clusterings(extractor_path, reducer_name='Reducer', grouper: Grouper = KMeansGrouper(100),
                        ignore_existing=True):
    """
    For a considered Extractor type and on specified Reducer types, computes clustering results using a given Grouper
    :param extractor_path: path where the current Extractor results are saved
    :param reducer_name: string giving the type of reducer results on which we want to apply the clustering operation
    ('Reducer' means all reducer types)
    :param grouper: Grouper object which will be used for the clustering computation
    :param ignore_existing: if True, only compute when no results exist already
    """
    for red in os.listdir(extractor_path):
        if reducer_name not in red: # Specify here the type of reducer you want to apply the clustering on.
            continue
        print(red)
        paths.current_reducer = os.path.join(extractor_path, red)

        if ignore_existing: # existing results verification
            done = False
            for file in os.listdir(paths.current_reducer):
                if grouper.get_name() in file:
                    done = True
            if done:
                print('Grouping already done!')
                continue

        paths.current_grouper=None
        chain = Chain(prep, load, e, r, grouper, paths)
        chain.load_reduced_features()

        """List here all the computations from the Chain object to execute, at the Grouper level"""

        chain.find_groups(plot_elements=True, show=False)
        chain.plot_ref_samples(show=False)
        chain.save_grouper()


def compute_and_save_scores(features, labels):
    """
    Computes the three internal validation indices and saves the results in a temporary file (copied by save_grouper())
    :param features: reduced features array
    :param labels: list of labels produced by a Grouper object after clustering computation
    """
    scores = [0, 0, 0]
    try:
        scores[0] = silhouette_score(features, labels)
        scores[1] = calinski_harabasz_score(features, labels)
        scores[2] = davies_bouldin_score(features, labels)
    except ValueError as v:
        print(v)
        scores = [0, 0, 0]
    with open('scores.txt', 'w') as f:
        f.write(str(scores[0]) + ',' + str(scores[1]) + ',' + str(scores[2]) + '\n')
        f.write('Silhouette_score: ' + str(scores[0]) + '\n')
        f.write('Calinski_Harabasz_score: ' + str(scores[1]) + '\n')
        f.write('Davies_Bouldin_score: ' + str(scores[2]) + '\n')

if __name__ == "__main__":
    tf.random.set_seed(34567)
    np.random.seed(34567)
    """MACHINES DEFINITION"""
    paths = Paths()
    prep = Preprocessor(crop_margin=-10, output_size=(224, 224), mode="full", data_path=paths.root)
    load = Loader(train_proportion=0.9, batch_size=64, buffer_size=128, n_samples=20000, n_test_samples=50)
    e = Extractor()
    r = Reducer()
    g = Grouper()

    """PATHS"""
    paths.current_preprocessor = "/home/vincent/data/Preprocessor2"

    dims_80 = {"ResnetExtractor3": 11,
               'MobilenetExtractor10': 13,
               'AutoencoderExtractor9': 27,
               'SimCLRExtractor8': 3,
               'SimCLRExtractor11': 5,
               'InceptionExtractor9': 8}  # 80% of explained variance for basic PCA
    dims_85 = {"ResnetExtractor3": 15,
               'MobilenetExtractor10': 17,
               'AutoencoderExtractor9': 46,
               'SimCLRExtractor8': 4,
               'SimCLRExtractor11': 6,
               'InceptionExtractor9': 12} # 85%
    dims_90 = {"ResnetExtractor3": 23,
               'MobilenetExtractor10': 25,
               'AutoencoderExtractor9': 92,
               'SimCLRExtractor8': 4,
               'SimCLRExtractor11': 8,
               'InceptionExtractor9': 21} # 90%

    for dir in os.listdir(paths.current_preprocessor):
        if 'Extractor' not in dir: # Specify here the type of extractor you want to apply the reduction and clustering on.
            continue
        extractor_path = os.path.join(paths.current_preprocessor, dir)
        print(dir)

        """Specify here the type of reducer to apply"""
        # r = PCAReducer(n_dim=dims_80[dir])
        # r = IsomapReducer(n_dim=dims_80[dir])
        r = UmapReducer(n_dim=dims_80[dir], n_neighbors=20, min_dist=0)
        # r = Identity()
        paths.current_extractor = extractor_path
        g = KMeansGrouper(100)
        chain = Chain(prep, load, e, r, g, paths)

        """List here the computations from the Chain object to execute, at the reducer level"""

        chain.load_encoded_features()
        # chain.measure_reduction_stability(10, min_dim=0)
        # chain.train_reducer(20000)
        # chain.reduce_dataset(20000)
        # chain.save_reducer()
        # print(chain.encoded_features[0])
        # print(np.cumsum(chain.reducer.model.explained_variance_ratio_))
        # print(np.argwhere(np.cumsum(chain.reducer.model.explained_variance_ratio_)>0.8))

        """Computations at the Grouper level"""
        compute_clusterings(extractor_path, reducer_name='UmapReducer', grouper=SpectralGrouper(30),
                            ignore_existing=False)