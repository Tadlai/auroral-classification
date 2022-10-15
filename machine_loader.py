from CNN_autoencoder import CNNAutoencoder
from classic_extractor import PolarHist
from preprocessor import Preprocessor
from simCLR_extractor import SimCLRExtractor
from pretrained_nets import ResNet, Inception, MobileNet
from grouper import *
from reducer import *


def load_machine(main_dir, machine_name):
    """
    Allows to load a Machine object from a saved directory.
    :param main_dir: saving subfolder corresponding to one of the four types of Machines: (Preprocessor, Extractor, Reducer, Grouper)
    :param machine_name: name of the machine to load (ex: MobileNetExtractor10)
    :return: the loaded Machine object
    """
    dir = os.path.join(main_dir, machine_name)
    dicpath = os.path.join(dir, "dict.txt")
    dic = joblib.load(dicpath)
    name = dic['_name']
    del dic['_name']
    new_machine = None
    if name == 'Preprocessor':
        new_machine = Preprocessor(**dic)
    elif "Extractor" in name:
        del dic['encoder'], dic['output_path']
        if name == "AutoencoderExtractor":
            del dic['model'], dic['decoder']
            new_machine = CNNAutoencoder(model_path=os.path.join(dir, 'encoder'), **dic)
        elif name == "SimCLRExtractor":
            del dic['criterion'], dic['optimizer'], dic['data_augmentation'], dic['negative_mask'], dic['projection']
            del dic['input_data_path']
            new_machine = SimCLRExtractor(model_path=os.path.join(dir, 'encoder'), **dic)
        elif name == "ResnetExtractor":
            new_machine = ResNet(**dic)
        elif name == "InceptionExtractor":
            new_machine = Inception(**dic)
        elif name == "MobilenetExtractor":
            new_machine = MobileNet(**dic)
    elif name == "PolarHist":
        del dic['encoder']
        new_machine = PolarHist(**dic)
    elif "Grouper" in name:
        del dic['model'], dic['trained'], dic['defined_n'],dic['labels'],dic['neigh'],dic['max_savename']
        if name == "KMeansGrouper":
            new_machine = KMeansGrouper(**dic)
        elif name == "SpectralGrouper":
            new_machine = SpectralGrouper(**dic)
        elif name == "DBScanGrouper":
            del dic['n_clusters']
            new_machine = DBScanGrouper(**dic)
        elif name == "OPTICSGrouper":
            del dic['n_clusters']
            new_machine = OPTICSGrouper(**dic)
        elif name == "BirchGrouper":
            new_machine = BirchGrouper(**dic)
        elif name == "AggloGrouper":
            new_machine = AggloGrouper(**dic)
        elif name == "AffinityGrouper":
            del dic['n_clusters']
            new_machine = AffinityGrouper(**dic)
        elif name == "MeanShiftGrouper":
            del dic['n_clusters']
            new_machine = MeanShiftGrouper(**dic)
        elif name == "MixtureGrouper":
            new_machine = MixtureGrouper(**dic)
        elif name == "COPKMeansGrouper":
            new_machine = COPKMeansGrouper(**dic)
    elif "Reducer" in name:
        del dic['model'], dic['trained'], dic['features_path']
        if name == "IdentityReducer":
            del dic['_n_dim']
            new_machine = Identity(**dic)
        else:
            dic['n_dim'] = dic.pop('_n_dim')
            dic['reducer_path'] = os.path.join(dir, 'reducer.txt')
            print(dic)
            if name == "UmapReducer":
                new_machine = UmapReducer(**dic)
            elif name == "PCAReducer":
                new_machine = PCAReducer(**dic)
            elif name == "SVDReducer":
                new_machine = SVDReducer(**dic)
            elif name == "IsomapReducer":
                new_machine = IsomapReducer(**dic)
            elif name == "TSNEReducer":
                new_machine = TSNEReducer(**dic)

    return new_machine


if __name__ == "__main__":
    """Tests of the loading function"""
    p = Preprocessor(ksize=1000)
    p.data_dir = "test"
    p.verify_and_save_machine("/home/vincent/machines/preprocessors")
    p2 = load_machine("/home/vincent/machines/preprocessors", p.save_name)

    e = CNNAutoencoder(1, 2, 3, 4, 5)
    e.data_dir = "test"
    e.verify_and_save_machine("/home/vincent/machines/extractors")
    e2 = load_machine("/home/vincent/machines/extractors", e.save_name)

    # s = SimCLRExtractor(1,2,3)
    # s.data_dir = "test"
    # s.verify_and_save_machine("/home/vincent/machines/extractors")
    # s2 = load_machine("/home/vincent/machines/extractors", "SimCLRExtractor2")

    r = ResNet(224, data_dir="test")
    r.verify_and_save_machine("/home/vincent/machines/extractors")
    r2 = load_machine("/home/vincent/machines/extractors", r.save_name)

    pl = PolarHist(data_dir="test")
    pl.verify_and_save_machine("/home/vincent/machines/extractors")
    pl2 = load_machine("/home/vincent/machines/extractors", pl.save_name)

    k = KMeansGrouper(5, data_dir="test")
    k.verify_and_save_machine("/home/vincent/machines/groupers")
    k2 = load_machine("/home/vincent/machines/groupers", k.save_name)

    db = DBScanGrouper(data_dir="test")
    db.verify_and_save_machine("/home/vincent/machines/groupers")
    db2 = load_machine("/home/vincent/machines/groupers", db.save_name)

    rd = TSNEReducer(n_dim=3, perplexity=4, early_exaggeration=3, data_dir="test")
    rd.verify_and_save_machine("/home/vincent/machines/reducers")
    rd2 = load_machine("/home/vincent/machines/reducers", rd.save_name)

