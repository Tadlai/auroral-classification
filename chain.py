import os
import cv2
import joblib
from pathlib import Path
from tqdm import tqdm
from CNN_autoencoder import CNNAutoencoder
from ensemble import Ensemble
from extractor import Extractor
from machine_loader import load_machine
from path_dataclass import Paths
from reducer import Reducer, Identity, UmapReducer, IsomapReducer, PCAReducer, TSNEReducer
from simCLR_extractor import SimCLRExtractor
from preprocessor import TFWriter, TFReader, Preprocessor
from loader import Loader, get_single_image, duplicate_img
from grouper import Grouper, SpectralGrouper, DBScanGrouper, OPTICSGrouper, BirchGrouper, AggloGrouper, \
    MeanShiftGrouper, AffinityGrouper, MixtureGrouper, COPKMeansGrouper
from grouper import KMeansGrouper
from groupings_fusion import display_results,single_fusion,measure_loss
import numpy as np
import tensorflow as tf
from sklearn.metrics.cluster import contingency_matrix
import matplotlib.pyplot as plt
from pretrained_nets import ResNet, Inception, MobileNet


class Chain:
    """Class implementing the main functions from the whole processing chain, including preprocessing,
     feature extraction, dimension reduction, clustering, displaying some results and loading/saving the results
      and methods"""
    def __init__(self, preprocessor: Preprocessor, loader: Loader, extractor: Extractor, reducer: Reducer,
                 grouper: Grouper, paths: Paths):
        """
        All parameters here are objects which must be instanciated and contain the desired parameters for each step.
        The 'paths' parameter contains all the filepaths necessary for the processing chain.
        """
        self.cl = None
        self.ml = None
        self.new_filepaths = None
        self.labels = None
        self.filepaths = []
        self.filenames = None
        self.encoded_features = None
        self.reduced_features = None
        self.features = None
        self.valid_len = None
        self.train_len = None
        self.valid_ds = None
        self.train_ds = None
        self.reducer_trained = False
        self.encoder = None
        self.min_loss = None
        self.preprocessor = preprocessor
        self.loader = loader
        self.extractor = extractor
        self.reducer = reducer
        self.grouper = grouper
        self.paths = paths
        # if a path is specified in the Paths object, then the corresponding Machine is loaded instead of the initial parameter
        if paths.current_preprocessor is not None:
            self.preprocessor = load_machine(paths.preprocessors,
                                             os.path.basename(os.path.normpath(paths.current_preprocessor)))
        if paths.current_extractor is not None:
            self.extractor = load_machine(paths.extractors, os.path.basename(os.path.normpath(paths.current_extractor)))
        if paths.current_reducer is not None:
            self.reducer = load_machine(paths.reducers, os.path.basename(os.path.normpath(paths.current_reducer)))
        if paths.current_grouper is not None:
            self.grouper = load_machine(paths.groupers, os.path.basename(os.path.normpath(paths.current_grouper)))

    def write_files(self):
        """
        Preprocesses the raw dataset and writes the output in .tfrecords format in a new folder
        """
        tfwr = TFWriter(data_path=self.preprocessor.data_path, preprocessor=self.preprocessor)
        tfwr.write_directory()

    def train_extractor(self, plot=True):
        """
        Train the specified feature extractor using the preprocessed dataset
        :param plot: boolean indicating if we want to show the training scores plot at the end of training.
        """
        self.load_datasets()
        loss, val_loss = self.extractor.train_tf(self.train_ds, self.valid_ds, self.loader.n_samples, self.valid_len,
                                                 self.paths.current_preprocessor)
        if loss is not None:
            self.min_loss = min(val_loss)
            print(self.min_loss)
            plt.ioff()
            plt.figure()
            plt.plot(loss, label="Training Loss")
            plt.plot(val_loss, label="Validation Loss")
            plt.legend()
            if plot:
                plt.show()
            else:
                plt.savefig("loss_plot.png")

    def load_datasets(self):
        """
        Loads the preprocessed dataset into memory and splits it into training and validation data
        """
        if self.train_len is None:
            tfr = TFReader()
            tfrecords_path = os.path.join(self.paths.current_preprocessor, "tfrecords")
            self.train_ds, self.valid_ds, self.train_len, self.valid_len = self.loader.train_valid_split(tfr,
                                                                                                         tfrecords_path)

    def load_encoded_features(self):
        """
        Loads the specified (in self.paths) encoded dataset along with the corresponding filenames, and converts them
        into usable filepaths.
        """
        input_path = os.path.join(self.paths.current_extractor, "encoded_features/features.npy")
        input_path_names = os.path.join(self.paths.current_extractor, "encoded_features/filenames.npy")
        self.encoded_features = np.load(input_path)
        self.filenames = np.load(input_path_names, allow_pickle=True)
        self.filepaths = self.get_filepaths(self.filenames)

    def load_reduced_features(self):
        """
        Loads the specified (in self.paths) reduced dataset in memory.
        """
        if self.filenames is None:
            self.load_encoded_features()
        input_path = os.path.join(self.paths.current_reducer, "reduced_features.npy")
        self.reduced_features = np.load(input_path)

    def get_filepaths(self, filenames):
        """
        Get usable filepaths from a list of filenames.
        :param filenames: numpy list of the images filenames
        :return: the list of the corresponding filepaths
        """
        filepaths = []
        root_path = os.path.join(self.paths.root, 'clean_dataset')
        for name in filenames:
            path = os.path.join(root_path, str(name)[2:-1] + '.jpg')
            filepaths.append(path)
        return filepaths

    def encode_dataset(self, dataset, dataset_len):
        """
        Uses a trained feature extractor to encode a preprocessed dataset. Saves it in the location specified in self.paths
        :param dataset: The loaded tf.Dataset to be encoded
        :param dataset_len: The lenght of the current dataset
        """
        output_path = os.path.join(self.paths.root, "encoded.npy")
        self.extractor.encode_dataset(dataset, dataset_len, self.loader.batch_size, output_path)

    def train_reducer(self, n_samples, filename="temp_reducer.sav"):
        """
        Trains the specified dimension reducer.
        :param n_samples: Maximum number of samples to use from the encoded dataset for training the dimension reducer
        :param filename: output name of the saved trained reducer
        """
        self.load_encoded_features()
        self.reducer.trainv2(self.encoded_features, n_samples, self.paths.current_extractor, filename)

    def reduce_dataset(self, n_samples, reducer_path=None):
        """
        Uses a trained reducer to load and reduce an encoded dataset.
        :param n_samples: Maximum number of samples to reduce
        :param reducer_path: Filepath of a trained reducer, if chosen different from the one specified in self.paths
        """
        if reducer_path is not None:
            self.reducer = Reducer(reducer_path=reducer_path)
        self.load_encoded_features()

        self.reduced_features = self.reducer.reduce_datav2(self.encoded_features, n_samples)

    def train_and_reduce_dataset(self, n_samples, reducer_path=None):
        """
        Trains and reduces an encoded dataset at the same time.
        :param n_samples: Maximum number of samples to reduce
        :param reducer_path: Filepath of a trained reducer, if chosen different from the one specified in self.paths
        """
        if reducer_path is not None:
            self.reducer = Reducer(reducer_path=reducer_path)
        self.load_encoded_features()
        self.reduced_features = self.reducer.train_reduce(self.encoded_features, n_samples,
                                                          self.paths.current_extractor)

    def plot_transformed_data(self):
        """
        Displays in 3d the first 3 dimensions of the current reduced dataset.
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.reduced_features[:, 0], self.reduced_features[:, 1], self.reduced_features[:, 2])
        ax.set_xlabel('axis 1')
        ax.set_ylabel('axis 2')
        ax.set_zlabel('axis 3')
        plt.show()

    def load_constraints(self):
        """
        Loads the 'must-link' and 'cannot-link' constraints used for constrained clustering.
        """
        ml = joblib.load(os.path.join(self.paths.ref, "must_link.txt"))
        cl = joblib.load(os.path.join(self.paths.ref, "cannot_link.txt"))
        self.ml = [(i + len(self.filepaths), j + len(self.filepaths)) for i, j in ml]
        self.cl = [(i + len(self.filepaths), j + len(self.filepaths)) for i, j in cl]

    def find_groups(self, plot_elements=True, display_reducer=None, show=True):
        """
        Apply the specified clustering method to the reduced dataset.
        :param plot_elements: Boolean indicating if we want to fetch random elements from each cluster for display.
        :param display_reducer: a Reducer object which can be used for further reducing the data before display
        :param show: if True, the custer elements are displayed, if not the final plot is only saved as png.
        """
        if self.grouper.get_name() == "COPKMeansGrouper":
            new_reduced_features, self.new_filepaths = self.grouper.add_ref_features(ref_dir=self.paths.ref, chain=self)
            self.load_constraints()
            scores = self.grouper.find_groups(new_reduced_features,
                                              filename=os.path.join(self.paths.root, "temp_labels.npy"),
                                              must_link=self.ml,
                                              cannot_link=self.cl)
        else:
            scores = self.grouper.find_groups(self.reduced_features,
                                              filename=os.path.join(self.paths.root, "temp_labels.npy"))
        print("Silhouette score : ", scores[0])
        print("Calinski Harabasz score : ", scores[1])
        print("Davies Bouldin score : ", scores[2])
        if self.grouper.get_name() == "COPKMeansGrouper":
            self.grouper.plot_clusters(new_reduced_features, display_reducer)
            if plot_elements:
                self.grouper.plot_clusters_elements(self.new_filepaths, self.preprocessor, n_aff=50, show=show)
        else:
            self.grouper.plot_clusters(self.reduced_features, display_reducer)
            if plot_elements:
                self.grouper.plot_clusters_elements(self.filepaths, self.preprocessor, n_aff=50, show=show)

    def find_number_of_groups(self, nmin, nmax, nstep, show=True):
        """
        Automatically find the best number of groups according to the Silhouette score (unstable).
        :param nmin: minimum number of groups to search
        :param nmax: maximum number of groups to search
        :param nstep: step to use for searching the number of groups
        :param show: if True, shows the scores plot according to the number of groups
        """
        if self.grouper.get_name() == "COPKMeansGrouper":
            new_reduced_features, self.new_filepaths = self.grouper.add_ref_features(ref_dir=self.paths.ref, chain=self)
            self.load_constraints()
            n_groups = self.grouper.find_nclusters(new_reduced_features, nmin, nmax, nstep, must_link=self.ml,
                                                   cannot_link=self.cl, show=show)
        else:
            n_groups = self.grouper.find_nclusters(self.reduced_features, nmin, nmax, nstep, show=show)
        print("Best number of groups: " + str(n_groups))

    def process_imgs(self, img_paths, predict=True):
        """
        Apply the whole processing chain on a list of input images.
        :param img_paths: Input list of images paths.
        :param predict: if true, the cluster label is computed.
        :return: a dictionary containing each intermediate result
        """
        imgs = []
        pr_imgs = []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            imgs.append(img)
            pr_imgs.append(self.preprocessor.preprocess(img, exclude_dark=False))
        pr_imgs = np.array(pr_imgs)
        enc_imgs = self.extractor.encode(pr_imgs)
        red_imgs = self.reducer.reduce_datav2(enc_imgs, len(imgs), save=False)
        if predict:
            label = self.grouper.predict(red_imgs)
            if self.new_filepaths is None:  # to avoid bug when dataset is augmented with ref samples
                neighbors, neighbors_labels = self.grouper.get_neighbors(red_imgs, self.filepaths, self.preprocessor)
            else:
                neighbors, neighbors_labels = self.grouper.get_neighbors(red_imgs, self.new_filepaths,
                                                                         self.preprocessor)
        else:
            label, neighbors, neighbors_labels = [None, None, None]
        return {'img': imgs, 'pr_img': pr_imgs, 'enc_img': enc_imgs, 'red_img': red_imgs, 'label': label,
                'neighbors': neighbors, 'neighbors_labels': neighbors_labels}

    def plot_ref_samples(self, show=True):
        """
        Displays the preprocessed reference samples as well as their nearest neighbors, with their computed cluster labels.
        :param show: if true, shows the results instead of just saving it as a png file.
        """
        pred_labels = []
        ref_labels = np.load(os.path.join(self.paths.ref, 'labels.npy'), allow_pickle=True)
        j = 0
        for dir in os.listdir(self.paths.ref):
            dir_path = os.path.join(self.paths.ref, dir)
            if os.path.isdir(dir_path):
                fig = plt.figure()
                list_imgs = os.listdir(dir_path)
                for img_name in list_imgs:
                    if ".jpg" not in img_name:
                        list_imgs.remove(img_name)

                list_paths = [os.path.join(dir_path, img_name) for img_name in list_imgs]
                dic_list = self.process_imgs(list_paths)
                pred_labels = pred_labels + [label for label in dic_list['label']]
                for i in range(len(list_imgs)):
                    # display
                    ax = fig.add_subplot(len(list_imgs), 6, i * 6 + 1)
                    ax.imshow(dic_list['pr_img'][i])
                    ax.set_title("REF: " + str(dic_list['label'][i]), x=-2, y=0.2)
                    ax.axis('off')
                    for k in range(1, 6):
                        ax = fig.add_subplot(len(list_imgs), 6, i * 6 + 1 + k)
                        ax.imshow(dic_list['neighbors'][i][k - 1])
                        ax.set_title(str(dic_list['neighbors_labels'][i][k - 1]), x=-1, y=0.1)
                        ax.axis('off')
                fig.suptitle(dir)
                plt.savefig("ref_neighbors" + str(j) + ".png")
                j += 1
        print(contingency_matrix(ref_labels, pred_labels))
        if show:
            plt.show()
        else:
            plt.close('all')

    def measure_clustering_stability(self, n_tries, show=True):
        """
        Gives the stability measure (robustness to subsampling) of the specified clustering method.
        :param n_tries: Number of times the subsampling+clustering is made
        :param show: if true, shows plots of the results
        :return: The value for the stability measure
        """
        if self.grouper.get_name() == "COPKMeansGrouper":
            new_reduced_features, self.new_filepaths = self.grouper.add_ref_features(ref_dir=self.paths.ref, chain=self)
            self.load_constraints()
            return self.grouper.measure_stability(new_reduced_features, n_tries, show=show, ml=self.ml, cl=self.cl)
        else:
            return self.grouper.measure_stability(self.reduced_features, n_tries, show=show)

    def measure_reduction_stability(self, n_tries):
        """
        Measures the stability according to changes in parameters of the reduction method.
        :param n_tries: Number of times the clustering labels are computed
        """
        self.grouper.measure_reduction_stability(self.encoded_features, n_tries, self.reducer)

    def save_preprocessor(self):
        """
        Saves the Preprocessor machine as well as the preprocessed data into the folder tree structure.
        """
        self.preprocessor.verify_and_save_machine(self.paths.preprocessors)
        self.preprocessor.verify_and_save_data(self.paths.data)
        self.paths.current_preprocessor = self.preprocessor.data_dir

    def save_extractor(self):
        """
        Saves the Extractor machine as well as the encoded data into the folder tree structure.
        """
        self.extractor.verify_and_save_machine(self.paths.extractors)
        self.extractor.verify_and_save_data(self.paths.current_preprocessor)
        self.paths.current_extractor = self.extractor.data_dir

    def save_reducer(self):
        """
        Saves the Reducer machine as well as the reduced data into the folder tree structure.
        """
        self.reducer.verify_and_save_machine(self.paths.reducers)
        self.reducer.verify_and_save_data(self.paths.current_extractor)
        self.paths.current_reducer = self.reducer.data_dir

    def save_grouper(self):
        """
        Saves the Grouper machine as well as the labels data into the folder tree structure.
        """
        self.grouper.verify_and_save_machine(self.paths.groupers)
        self.grouper.verify_and_save_data(self.paths.current_reducer)
        self.paths.current_grouper = self.grouper.data_dir


if __name__ == "__main__":
    tf.random.set_seed(34567)
    np.random.seed(34567)
    """MACHINES DEFINITION"""
    paths = Paths()
    prep = Preprocessor(crop_margin=-10, output_size=(224, 224), mode="full", data_path=paths.root)
    load = Loader(train_proportion=0.9, batch_size=64, buffer_size=128, n_samples=20000, n_test_samples=50)

    # cnn = CNNAutoencoder(n_filters=2, n_layers=2, lr=0.1, batch_size=128, n_epochs=2, input_size=224, encoded_size=128)
    # polhist = PolarHist()
    resnet = ResNet(size=224)
    # inception = Inception(size=224)
    # mobile = MobileNet(size=224)
    # simclr = SimCLRExtractor(lr=0.001, batch_size=64, n_epochs=10, size=224,
    #                          transformations={'fliplr': True, 'flipud': True, 'crop': True, 'colorj': False,
    #                                           'colord': False})  # transformations dict has no effect
    # # ,model_path='/home/vincent/models/06_11_2022_18_32_05_enc'
    umap = UmapReducer(n_dim=3, n_neighbors=20, min_dist=0)
    # tsne = TSNEReducer()
    # pca = PCAReducer(n_dim=3)
    # kpca = PCAReducer(kernel="rbf", n_dim=10)
    # iso = IsomapReducer(30)
    # id = Identity()
    kmeans = KMeansGrouper(100)
    # cop = COPKMeansGrouper(100)
    # spg = SpectralGrouper(30)
    # dbs = DBScanGrouper(eps=0.5)
    # op = OPTICSGrouper()
    # bg = BirchGrouper(30)
    # mi = MixtureGrouper(30)

    """PATHS"""
    # paths.current_extractor = paths.root
    paths.current_preprocessor = "/home/vincent/data/Preprocessor2"

    """PROCESSING CHAIN"""
    chain = Chain(prep, load, resnet, umap, kmeans, paths)
    chain.write_files()
    chain.save_preprocessor()

    chain.train_extractor(plot=False)
    chain.load_datasets()
    chain.encode_dataset(chain.train_ds, chain.train_len)
    chain.save_extractor()
    chain.train_reducer(20000)
    chain.reduce_dataset(20000)
    chain.save_reducer()
    # chain.train_and_reduce_dataset(n_samples=8000)
    chain.plot_transformed_data()
    chain.load_encoded_features()
    # chain.measure_reduction_stability(10)
    # chain.measure_clustering_stability(100)
    chain.find_groups(plot_elements=False,display_reducer=None)
    chain.plot_ref_samples()
    chain.save_grouper()
    # chain.find_number_of_groups(nmin=70,nmax=120,nstep=2)

    """ENSEMBLE VOTING"""
    voting_paths10030 = np.load(
        '/home/vincent/PycharmProjects/auroraclassif/candidates_ranking_100_30.npy',
        allow_pickle=True)

    measure_loss(voting_paths10030,
                 extractor_name='MobilenetExtractor10',
                 reducer_name='Reducer',
                 save_name='fusion_results_mobilenet.npy',
                 append=False)
    labels = single_fusion(voting_paths10030,
                           n_results=12,
                           mt=0.8333,
                           mcs=22,
                           extractor_name='SimCLRExtractor11',
                           reducer_name='Reducer')
    np.save('labels.npy',labels)
    display_results(labels)
