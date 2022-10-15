import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from path_dataclass import Paths
from preprocessor import Preprocessor

"""Simple display of the original vs preprocessed versions of the reference image samples."""

paths = Paths()
prep = Preprocessor(crop_margin=-10, output_size=(224, 224), mode="full", data_path=paths.root)
ref_dir = paths.ref
for dir in os.listdir(ref_dir):
    dir_path = os.path.join(ref_dir, dir)
    if os.path.isdir(dir_path):
        fig = plt.figure()
        list_imgs = os.listdir(dir_path)
        for img_name in list_imgs:
            if ".jpg" not in img_name:
                list_imgs.remove(img_name)

        list_paths = [os.path.join(dir_path, img_name) for img_name in list_imgs]
        imgs = []
        pr_imgs = []
        for img_path in list_paths:
            img = cv2.imread(img_path)
            imgs.append(img)
            pr_imgs.append(prep.preprocess(img, exclude_dark=False))
        imgs = np.array(imgs)
        pr_imgs = np.array(pr_imgs)

        for j in range(1, len(imgs) + 1):
            plt.subplot(2, len(imgs), j)
            plt.imshow(imgs[j-1])
            plt.axis('off')
            plt.subplot(2, len(imgs), len(imgs)+j)
            plt.imshow(pr_imgs[j-1])
            plt.axis('off')
        plt.suptitle(dir)
        plt.savefig(dir)
plt.show()

