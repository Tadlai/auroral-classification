import os
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from chain import Chain
from extractor import Extractor
from grouper import Grouper, KMeansGrouper, SpectralGrouper, COPKMeansGrouper, BirchGrouper, AggloGrouper, \
    MixtureGrouper, DBScanGrouper, OPTICSGrouper, AffinityGrouper
from loader import Loader
from path_dataclass import Paths
from preprocessor import Preprocessor
from reducer import Reducer, UmapReducer, IsomapReducer
import pandas as pd


def compute_stability(preprocessor_path="/home/vincent/data/Preprocessor2"):
    """
    For every result in a preprocessor folder, compute the clustering stability and saves it in the corresponding folder
    :param preprocessor_path: preprocessor folder containing the clustering results to assess
    """
    paths = Paths()
    paths.current_preprocessor = preprocessor_path
    paths_to_compute = Path(paths.current_preprocessor).rglob('labels')
    for path in paths_to_compute:
        print(path)
        txt = os.path.join(str(path), 'stability.txt')
        if os.path.exists(txt):
            print('Stability already computed!')
            continue
        paths.current_extractor = path.parents[2]
        paths.current_reducer = path.parents[1]
        paths.current_grouper = path.parents[0]
        chain = Chain(Preprocessor(), Loader(), Extractor(), Reducer(), Grouper(), paths)
        chain.load_reduced_features()
        ari_m, ari_s, ami_m, ami_s = chain.measure_clustering_stability(50, show=False)
        print(ari_m, ari_s, ami_m, ami_s)
        with open(txt, 'w') as f:
            f.write(str(ari_m) + ',' + str(ari_s) + ',' + str(ami_m) + ',' + str(ami_s))


def select_results(preprocessor_path='/home/vincent/data/Preprocessor2/'):
    """
    From a set of clustering results, plots and saves a ranking based on 4 clustering validity indices
    :param preprocessor_path: preprocessor folder containing the clustering results to assess
    """

    possible_paths = Path(preprocessor_path).rglob('scores.txt')
    possible_paths = [str(path) for path in possible_paths]
    candidates_paths= []
    for path in possible_paths:
        """Here specify the types of clustering results to keep"""
        #if 'KMeansGrouper14' in path or 'SpectralGrouper17' in path or 'COPKMeansGrouper16' in path or 'AggloGrouper15' in path:
        candidates_paths.append(path)

    """Gathering scores"""
    s = []
    db = []
    ch = []
    st = []
    for path in candidates_paths:
        print(path)
        with open(path, 'r') as f:
            a = f.readline()
        a = a.split(sep=',')
        a[2] = a[2][:-2]
        print(a)
        try:
            with open(path[:-10]+'stability.txt') as f:
                b = f.readline()
            b = b.split(sep=',')

            s.append(float(a[0]))
            db.append(float(a[2]))
            ch.append(float(a[1]))
            st.append((float(b[2])+float(b[0]))/2)
        except FileNotFoundError as e:
            print(e)
            candidates_paths.remove(path)
    list_paths = [str(path)[len(preprocessor_path):-len('/labels/scores.txt')] for path
                  in candidates_paths]
    df = pd.DataFrame(np.array([s, db, ch, st]).T,list_paths,['Silhouette','Davies_Bouldin','Calinski_Harabasz','Stability'])

    """Sorting results"""
    inds = df.Silhouette.argsort()
    inddb = df.Davies_Bouldin.argsort()
    indch = df.Calinski_Harabasz.argsort()
    indst = df.Stability.argsort()

    order = np.zeros(len(inds))
    for ind in [inds,inddb,indch,indst]:
        for i in ind:
            order[ind[i]] += i

    indx = np.argsort(order)

    sn = (np.array(s) ) / (np.max(s))
    chn = (np.array(ch) ) / (np.max(ch))
    dbn = 1 - (np.array(db) ) / (np.max(db))
    stn = (st) / (np.max(st))

    paths_to_save = [str(path)[:-len('scores.txt')] + 'group_labels.npy' for path
                  in candidates_paths]

    np.save('candidates_ranking_100_30.npy',np.array(paths_to_save)[indx])

    plt.figure()
    plt.xticks(range(len(list_paths)), np.array(list_paths)[indx], rotation=90)
    plt.plot(range(len(list_paths)), np.array(sn)[indx] ,label='Silhouette score')
    plt.plot(range(len(list_paths)), np.array(chn)[indx],label='Calinsky-Harabasz score')
    plt.plot(range(len(list_paths)), np.array(dbn)[indx] ,label='Davies-Bouldin score')
    plt.plot(range(len(list_paths)), np.array(stn)[indx],label='Stability score')
    plt.legend()
    plt.subplots_adjust(top=0.98,bottom=0.41,left=0.1,right=0.9,hspace=0.2,wspace=0.2)


    plt.figure()
    plt.xticks(range(len(list_paths)), np.array(list_paths)[indx], rotation=90)
    plt.plot(range(len(list_paths)), np.array(s)[indx], label='Silhouette score')
    plt.legend()
    plt.subplots_adjust(top=0.98, bottom=0.41, left=0.1, right=0.9, hspace=0.2, wspace=0.2)

    plt.figure()
    plt.xticks(range(len(list_paths)), np.array(list_paths)[indx], rotation=90)
    plt.plot(range(len(list_paths)), np.array(ch)[indx],label='Calinsky-Harabasz score')
    plt.legend()
    plt.subplots_adjust(top=0.98, bottom=0.41, left=0.1, right=0.9, hspace=0.2, wspace=0.2)

    plt.figure()
    plt.xticks(range(len(list_paths)), np.array(list_paths)[indx], rotation=90)
    plt.plot(range(len(list_paths)), np.array(db)[indx] ,label='Davies-Bouldin score')
    plt.legend()
    plt.subplots_adjust(top=0.98, bottom=0.41, left=0.1, right=0.9, hspace=0.2, wspace=0.2)

    plt.figure()
    plt.xticks(range(len(list_paths)), np.array(list_paths)[indx], rotation=90)
    plt.plot(range(len(list_paths)), np.array(st)[indx],label='Stability score')
    plt.legend()
    plt.subplots_adjust(top=0.98, bottom=0.41, left=0.1, right=0.9, hspace=0.2, wspace=0.2)

    plt.figure()
    plt.xticks(range(len(list_paths)), np.array(list_paths)[indx], rotation=90)
    plt.plot(range(len(list_paths)), np.array(order/4)[indx], label='Order')
    plt.legend()
    plt.subplots_adjust(top=0.98, bottom=0.41, left=0.1, right=0.9, hspace=0.2, wspace=0.2)

    plt.show()

if __name__ == "__main__":
    compute_stability()
    select_results()
