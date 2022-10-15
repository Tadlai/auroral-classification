"""ENSEMBLE VOTING"""
import os
import numpy as np
from matplotlib import pyplot as plt
from chain import Chain
from ensemble import Ensemble
from grouper import KMeansGrouper, ExtendedGrouper
from loader import Loader
from path_dataclass import Paths
from preprocessor import Preprocessor
from reducer import UmapReducer, PCAReducer
from pretrained_nets import ResNet
import pandas as pd


def display_results(labels):
    """
    Displays the results of an ensemble voting (it can also add the reference samples to the displayed set)
    :param labels: List of labels computed by an ensemble voting process.
    """
    chain.grouper.labels = labels
    chain.grouper.n_clusters = np.max(labels) + 1

    chain.load_reduced_features()
    #chain.grouper.fit_kneighbors(chain.reduced_features)
    new_reduced_features, chain.new_filepaths = chain.grouper.add_ref_features(ref_dir=chain.paths.ref, chain=chain)
    print(chain.new_filepaths)
    chain.grouper.plot_clusters(new_reduced_features,reducer=PCAReducer(n_dim=2))
    # chain.grouper.plot_clusters_elements(chain.filepaths, chain.preprocessor, n_aff=20)
    #chain.plot_ref_samples(ref_dir='/home/vincent/AuroraShapes/AuroraClasses/')


def measure_loss(voting_paths, extractor_name, reducer_name, save_name, append=False):
    """
    Function allowing to compare and save voting results according to varying values of 'minimum cluster size',
    'majority threshold' and 'n_results', based on the defined ensemble loss function.
    :param voting_paths: list of pathnames of the folders containing all clustering results candidates for voting
    (ordered by increasing performance)
    :param extractor_name: Name of an extractor, if we wish to keep only the results based on this extractor, else put 'Extractor'
    :param reducer_name: Name of a reducer, if we wish to keep only the results based on this reducer, else put 'Reducer'
    :param save_name: Name of the file which will contain the loss according to the parameters values
    :param append: if True, appends the results to an existing file, specified by save_name
    """
    if os.path.exists(save_name) and append:
        data = np.load(save_name, allow_pickle=True).tolist()
    else:
        data = []
    for n_results in range(2, 20):
        voting_partitions = []
        for path in voting_paths:
            if extractor_name not in str(path) or reducer_name not in str(path):
                continue
            voting_partitions.append(np.load(str(path), allow_pickle=True))

        voting_partitions = voting_partitions[-n_results:]
        voting_partitions = np.array(voting_partitions)
        e = Ensemble(voting_partitions, majority_thresh=0.2, min_cluster_size=1)
        e.build_coassociation_matrix()
        coassoc = e.coassoc

        for mcs in range(2, 50, 4):
            for mt in np.arange(0, 1, 1 / n_results):
                if mt < 0.4 or mt > 0.9:
                    continue
                e = Ensemble(voting_partitions, majority_thresh=mt, min_cluster_size=mcs)
                e.coassoc = coassoc
                labels = e.compute_clusters()
                e.plot_cluster_count()
                loss = e.compute_loss(n_groups=25)
                print([n_results, mcs, mt, loss, e.n_excluded])
                data.append([n_results, mcs, mt, loss, e.n_excluded, e.n_clusters])

        np.save(save_name, np.array(data))


def single_fusion(voting_paths, n_results, mt, mcs, extractor_name, reducer_name):
    """
    Performs a single ensemble voting based on the specified parameters
    :param voting_paths: list of pathnames of the folders containing all clustering results candidates for voting
    (ordered by increasing performance)
    :param n_results: Number of results to take for voting (it takes the last 'n_results' elements of voting_paths)
    :param mt: co-occurrence majority threshold, above which two clusters are merged
    :param mcs: minimum cluster size
    :param extractor_name: Name of an extractor, if we wish to keep only the results based on this extractor, else put 'Extractor'
    :param reducer_name: Name of a reducer, if we wish to keep only the results based on this reducer, else put 'Reducer'
    """
    voting_partitions = []
    for path in voting_paths:
        if extractor_name not in str(path) or reducer_name not in str(path):
            continue
        voting_partitions.append(np.load(str(path), allow_pickle=True))

    voting_partitions = voting_partitions[-n_results:]
    voting_partitions = np.array(voting_partitions)

    e = Ensemble(voting_partitions, majority_thresh=mt, min_cluster_size=mcs)
    e.build_coassociation_matrix()
    labels = e.compute_clusters()
    e.plot_cluster_count()

    plt.figure()
    plt.hist((e.coassoc + e.coassoc.T).flatten(), bins=20)
    plt.show()
    e.compute_loss(25)
    return labels

if __name__ == "__main__":
    paths = Paths()
    paths.current_preprocessor = "/home/vincent/data/Preprocessor2"
    paths.current_extractor = "/home/vincent/data/Preprocessor2/SimCLRExtractor11"
    paths.current_reducer = "/home/vincent/data/Preprocessor2/SimCLRExtractor11/UmapReducer55"
    prep = Preprocessor(crop_margin=-10, output_size=(224, 224), mode="full", data_path=paths.root)
    load = Loader(train_proportion=0.9, batch_size=64, buffer_size=128, n_samples=20000, n_test_samples=50)
    kmeans = ExtendedGrouper()
    resnet = ResNet(size=224)
    umap = UmapReducer(n_dim=3, n_neighbors=20, min_dist=0)
    chain = Chain(prep, load, resnet, umap, kmeans, paths)

    # voting_paths = Path('/home/vincent/data/Preprocessor2/').rglob('group_labels.npy')
    voting_paths100 = [
        '/home/vincent/data/Preprocessor2/SimCLRExtractor8/PCAReducer26/AggloGrouper11/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/ResnetExtractor3/UmapReducer53/SpectralGrouper10/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor8/IsomapReducer38/AggloGrouper11/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor11/IsomapReducer36/KMeansGrouper9/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor8/PCAReducer26/KMeansGrouper9/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor8/PCAReducer32/KMeansGrouper9/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor8/IsomapReducer38/KMeansGrouper9/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/AutoencoderExtractor9/UmapReducer56/SpectralGrouper10/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/InceptionExtractor9/UmapReducer58/COPKMeansGrouper13/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/ResnetExtractor3/UmapReducer53/COPKMeansGrouper13/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor8/UmapReducer57/COPKMeansGrouper13/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/ResnetExtractor3/UmapReducer53/AggloGrouper11/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/InceptionExtractor9/UmapReducer58/SpectralGrouper10/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/InceptionExtractor9/UmapReducer58/AggloGrouper11/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/MobilenetExtractor10/UmapReducer54/SpectralGrouper10/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/AutoencoderExtractor9/UmapReducer56/COPKMeansGrouper13/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/MobilenetExtractor10/UmapReducer54/COPKMeansGrouper13/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/ResnetExtractor3/UmapReducer53/KMeansGrouper9/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor11/UmapReducer55/COPKMeansGrouper13/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor8/UmapReducer57/AggloGrouper11/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor8/UmapReducer57/SpectralGrouper10/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor11/UmapReducer55/SpectralGrouper10/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/MobilenetExtractor10/UmapReducer54/AggloGrouper11/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/InceptionExtractor9/UmapReducer58/KMeansGrouper9/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor11/UmapReducer55/AggloGrouper11/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/MobilenetExtractor10/UmapReducer54/KMeansGrouper9/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/AutoencoderExtractor9/UmapReducer56/AggloGrouper11/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor8/UmapReducer57/KMeansGrouper9/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor11/UmapReducer55/KMeansGrouper9/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/AutoencoderExtractor9/UmapReducer56/KMeansGrouper9/labels/group_labels.npy']

    voting_paths30 = [
        '/home/vincent/data/Preprocessor2/ResnetExtractor3/UmapReducer53/SpectralGrouper17/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor8/UmapReducer57/COPKMeansGrouper16/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/ResnetExtractor3/UmapReducer53/COPKMeansGrouper16/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/ResnetExtractor3/UmapReducer53/AggloGrouper15/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/InceptionExtractor9/UmapReducer58/SpectralGrouper17/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor8/UmapReducer57/AggloGrouper15/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/InceptionExtractor9/UmapReducer58/COPKMeansGrouper16/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor8/UmapReducer57/SpectralGrouper17/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/MobilenetExtractor10/UmapReducer54/SpectralGrouper17/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/MobilenetExtractor10/UmapReducer54/COPKMeansGrouper16/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/AutoencoderExtractor9/UmapReducer56/COPKMeansGrouper16/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor11/UmapReducer55/SpectralGrouper17/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor11/UmapReducer55/COPKMeansGrouper16/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/AutoencoderExtractor9/UmapReducer56/SpectralGrouper17/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor11/UmapReducer55/AggloGrouper15/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/MobilenetExtractor10/UmapReducer54/AggloGrouper15/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/InceptionExtractor9/UmapReducer58/AggloGrouper15/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor8/UmapReducer57/KMeansGrouper14/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/AutoencoderExtractor9/UmapReducer56/AggloGrouper15/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/ResnetExtractor3/UmapReducer53/KMeansGrouper14/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/MobilenetExtractor10/UmapReducer54/KMeansGrouper14/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/InceptionExtractor9/UmapReducer58/KMeansGrouper14/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/AutoencoderExtractor9/UmapReducer56/KMeansGrouper14/labels/group_labels.npy',
        '/home/vincent/data/Preprocessor2/SimCLRExtractor11/UmapReducer55/KMeansGrouper14/labels/group_labels.npy']

    voting_paths10030 = np.load('/home/vincent/PycharmProjects/auroraclassif/candidates_ranking_100_30.npy',
                                allow_pickle=True)

    # measure_loss(voting_paths10030,
    #              extractor_name='MobilenetExtractor10',
    #              reducer_name='Reducer',
    #              save_name='fusion_results_mobilenet.npy',
    #              append=False)
    # labels = single_fusion(voting_paths10030,
    #                        n_results=12,
    #                        mt=0.8333,
    #                        mcs=22,
    #                        extractor_name='SimCLRExtractor11',
    #                        reducer_name='Reducer')
    labels = np.load('/home/vincent/aurora project/fusion_results/fused_simCLR11/nr12/labels.npy')
    # np.save('labels.npy',labels)
    display_results(labels)
