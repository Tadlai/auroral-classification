import os
import joblib
import numpy as np


"""
Conversion of the subfolders containing the reference samples into 'cannot link' and 'must link' constraints, saved as txt files.
"""
dataset = "/home/vincent/DeepLearning/Season_2019-2020/clean_dataset"
root = "/home/vincent/AuroraShapes/AuroraClasses/"
filepaths = []
labels = []
must_link = []
cannot_link = []
key_imgs = []
l = 0
i = 0
print(os.listdir(root))
for dire in os.listdir(root):
    if os.path.isdir(root + dire):
        k = 0
        for img in os.listdir(root + dire):
            if img[-3:] == "jpg":
                filepaths.append(img)
                labels.append(l)
                if img in os.listdir(dataset):
                    os.remove(os.path.join(dataset,img))
                if k == 0:
                    key_imgs.append(i)
                else:
                    must_link.append((i-1,i))
                k += 1
                i += 1
        l += 1
for j in range(1,len(key_imgs)):
    for i in range(1,j):
        cannot_link.append((key_imgs[j-i],key_imgs[j]))
#np.save(root + "filepaths.npy", filepaths)
#np.save(root + "labels.npy", labels)
joblib.dump(must_link, root + "must_link.txt")
joblib.dump(cannot_link, root + "cannot_link.txt")
print(filepaths)
print(labels)
print(must_link)
print(cannot_link)