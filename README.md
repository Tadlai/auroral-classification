# Automatic morphological classification of auroral structures

This repository contains the python scripts used in my end-of-studies project, which consisted in automatically classifying auroral images based on their morphologies. You can find here the report itself as well. 

## Abstract from the report

In this project we wanted to automatically find groupings in auroral all-sky color images from the Kjell Henriksen Observatory in Svalbard, according to the morphology of the auroras, and this without any manual labelling.  

The data set used contained only clear skies with auroras and sometimes with the Moon. We used 2 feature extractors based on unsupervised deep-learning (Convolutional Autoencoder, SimCLR network) and 3 Convolutional neural networks pretrained on the ImageNet dataset (Resnet-50, Inception-v3, MobileNet-v2). We used 4 different dimension reduction methods on the feature data (PCA, Kernel PCA, Isomap and UMAP), followed by 3 simple clustering methods (K-means, Spectral and Hierarchical clustering) and a constrained clustering method (COPK-means).  All computed combinations of the listed methods were ranked according to internal validation indices and validated by manual visual inspection of the data. For each feature extractor, the best results were combined using a majority voting principle in order to provide more complex groupings of the data.  

The fusion result which originates from a SimCLR feature extractor gave the most relevant partition of the data, by grouping together images containing similar auroral features such as vortex-like structures or patchy texture.  However the presence of the Moon in the image constituted an important bias in the final feature data representation.

## Code organization

The main script is the chain.py file, which contains all functions necessary to compute clustering results based on an auroral image dataset. The file apply_chain is readily usable to compute new labels from an existing solution.

Other independent scripts are : candidates_election.py, compute_groupings.py, display_fusion_results.py, display_preprocessed_labels.py, write_readable_txt_files.py. 

The other files mainly define classes and methods which are used in the scripts cited above.

This repository does not contain the image data as it is too big to upload.

## Requirements
All scripts are written in Python 3.7.

Main libraries used: opencv-python, joblib, tqdm, numpy, tensorflow, scikit-learn, matplotlib, datetime

If you have questions about my work, I'll be glad to answer, at: vtsr @ gmx.fr
