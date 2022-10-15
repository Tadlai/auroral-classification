import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter


class Ensemble:
    def __init__(self, voting_partitions, partition_weights=None, majority_thresh=0.5, min_cluster_size=20):
        self.n_excluded = None
        self.mean_cluster_size = None
        self.med_cluster_size = None
        self.n_clusters = None
        self.trash_cluster = False
        self.voting_partitions = voting_partitions
        self.n_partitions = len(voting_partitions)
        self.n_samples = np.min([len(voting_partitions[k]) for k in range(self.n_partitions)])
        self.coassoc = np.zeros((self.n_samples, self.n_samples))
        if partition_weights is None:
            self.partition_weights = np.ones(self.n_partitions)
        self.dic_clusters = {i: [i] for i in range(self.n_samples)}
        self.list_clusters = np.array([i for i in range(self.n_samples)])
        self.majority_thresh = majority_thresh
        self.min_cluster_size = min_cluster_size

    def update_coassociation_matrix(self, partition, partition_weight):
        vote = partition_weight / self.n_partitions
        for i in range(self.n_samples):
            partitioni = partition[i]
            for j in range(i):
                if partitioni == partition[j]:
                    self.coassoc[i, j] = self.coassoc[i, j] + vote

    def merge_clusters(self, i, j):
        label_i = self.list_clusters[i]
        label_j = self.list_clusters[j]
        if label_j != label_i:
            new_label = min(label_i, label_j)
            old_label = max(label_i, label_j)
            samples_to_move = self.dic_clusters[old_label]
            self.dic_clusters[new_label] = self.dic_clusters[new_label] + samples_to_move  # merge clusters
            self.list_clusters[samples_to_move] = new_label  # update list
            self.dic_clusters[old_label] = []  # old cluster is now empty

    def majority_vote(self):
        for i in range(self.n_samples):
            for j in range(i):
                if self.coassoc[i, j] > self.majority_thresh:
                    self.merge_clusters(i, j)

    def build_coassociation_matrix(self):
        for i in tqdm(range(self.n_partitions)):
            partition = self.voting_partitions[i]
            weight = self.partition_weights[i]
            self.update_coassociation_matrix(partition, weight)

    def compute_clusters(self):
        self.majority_vote()
        self.assemble_single_labels()
        self.n_clusters = self.rearrange_labels()
        return self.list_clusters

    def get_voted_partition(self):
        self.build_coassociation_matrix()
        return self.compute_clusters()

    def assemble_single_labels(self):  # not in original algorithm
        for k, v in self.dic_clusters.items():
            if len(v) < self.min_cluster_size:
                self.list_clusters[self.list_clusters == k] = -1
        self.trash_cluster = True

    def rearrange_labels(self):
        u = np.unique(self.list_clusters)
        print('Number of clusters:', len(u))
        if self.trash_cluster:
            map_dic = {u[i]: i - 1 for i in range(len(u))}
        else:
            map_dic = {u[i]: i for i in range(len(u))}
        newlist = np.copy(self.list_clusters)
        for k, v in map_dic.items():
            newlist[self.list_clusters == k] = v
        self.list_clusters = newlist
        return len(u)

    def plot_cluster_count(self):
        counts = []
        copy_dic = self.dic_clusters.copy()
        for key, cl_list in self.dic_clusters.items():
            if len(cl_list) != 0:
                counts.append(len(cl_list))
            else:
                del copy_dic[key]
        print('median before rearrangement: ', np.median(counts))
        print('mean before rearrangement: ', np.mean(counts))
        c = Counter(self.list_clusters)
        print(list(c.values()))
        self.med_cluster_size = np.median(list(c.values()))
        self.mean_cluster_size = np.mean(list(c.values()))
        print('median after rearrangement: ', self.med_cluster_size)
        print('mean after rearrangement: ', self.mean_cluster_size)
        self.n_excluded = c[-1]
        print('Number of excluded samples:',self.n_excluded)
        # plt.figure()
        # plt.bar(list(copy_dic.keys()), counts, width=1)
        # plt.title("Cluster sizes before label rearrangement")
        # plt.xlabel('Label number')
        # plt.ylabel('Cluster size')
        # plt.figure()
        # plt.hist(self.list_clusters, bins=np.max(self.list_clusters) + 2, range=(-1, np.max(self.list_clusters) + 1),
        #          align='left')
        # plt.title("Cluster sizes after label rearrangement")
        # plt.xlabel('Label number')
        # plt.ylabel('Cluster size')
        # plt.show()


    def train_grouper(self, features):
        self.neigh = KNeighborsClassifier()
        self.neigh.fit(features, self.list_clusters)

    def predict(self, batch):
        return self.neigh.predict(batch)

    def compute_loss(self,group_size=None,n_groups=None):
        c = Counter(self.list_clusters)
        N = len(self.list_clusters)
        if group_size is None and n_groups is not None:
            group_size = N/n_groups
        elif group_size is not None and n_groups is None:
            n_groups = N//group_size
        else:
            print('Only one parameter should be set at the same time!')
        print(group_size,n_groups)
        target = np.zeros(N)
        target[:n_groups] = group_size
        current_sizes = np.array(sorted(list(c.values()), reverse=True))
        comparable = np.zeros_like(target)
        comparable[:len(current_sizes)] = current_sizes
        mse_loss = (1/n_groups)*np.sum(np.abs(comparable-target))

        return mse_loss





if __name__ == '__main__':
    a = np.random.randint(0, 10, (2, 1000))
    e = Ensemble(a, majority_thresh=0.5, min_cluster_size=0)
    print(e.get_voted_partition())
    e.plot_cluster_count()
    plt.figure()
    plt.imshow(e.coassoc)
    plt.figure()
    plt.imshow(e.coassoc > 0.5)
    plt.show()
