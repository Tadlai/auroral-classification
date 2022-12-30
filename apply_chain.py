from chain import Chain, tf, np, Paths, Preprocessor, Loader, Extractor, Reducer, ExtendedGrouper
from sklearn.neighbors import KNeighborsClassifier

tf.random.set_seed(34567)
np.random.seed(34567)

"""MACHINES DEFINITION"""
paths = Paths()
prep = Preprocessor(crop_margin=-10, output_size=(224, 224), mode="full", data_path=paths.root)
load = Loader(train_proportion=1, batch_size=64, buffer_size=128, n_samples=20000, n_test_samples=0)
extractor = Extractor()
reducer = Reducer()
grouper = ExtendedGrouper()

"""PATHS"""
# keep this line commented if the preprocessing step hasn't been done yet; otherwise, put the location of the saved results
paths.current_preprocessor = "/home/vincent/data/Preprocessor3"
# keep these lines the same (just change 'vincent...') if the corresponding processing step hasn't been done yet; otherwise, put the location of the saved results
paths.current_extractor = "/home/vincent/data/Preprocessor2/SimCLRExtractor11/"
paths.current_reducer = "/home/vincent/data/Preprocessor2/SimCLRExtractor11/UmapReducer55"
# put here the locations of the reference labels and dataset
reference_labels_path = '/home/vincent/data/fusion_results/fused_simCLR_NCT/nr4/labels.npy'
reference_dataset_path = "/home/vincent/data/Preprocessor2/SimCLRExtractor11/UmapReducer55/reduced_features.npy"
# <!> Don't forget to change the path names to your local paths in path_dataclass.py

"""PROCESSING CHAIN"""
chain1 = Chain(prep, load, extractor, reducer, grouper, paths)

# Get initial dataset and labels
reference_labels = np.load(reference_labels_path)
reference_dataset = np.load(reference_dataset_path)


# Preprocessing (keep the 2 lines commented if preprocessing has already been done)
#chain1.write_files()
#chain1.save_preprocessor()

# Feature extraction
chain1.load_datasets()
chain1.encode_dataset(chain1.train_ds, chain1.train_len)
chain1.save_extractor()

# Data reduction
chain1.reduce_dataset(20000) # upper bound for the number of samples to reduce
chain1.save_reducer()

# Fit K neighbours model
chain1.load_reduced_features()
neigh = KNeighborsClassifier()
neigh.fit(reference_dataset, reference_labels)
predicted_labels = neigh.predict(chain1.reduced_features)
corresponding_filenames = chain1.filenames # To keep the right order in the images labels

print(predicted_labels)
print(corresponding_filenames)

