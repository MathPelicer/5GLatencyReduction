import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


def main_DBSCAN():
    print("Hello World DBSCAN")
    
    # X = np.array([[1, 0], [2, 0], [2, 0],
    #             [8, 0], [8, 0], [25, 0]])
    # clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    
    # print(clustering.get_params())
    
    centers = [[1, 1], [-1, -1], [1, -1]]    
    X, labels_true = make_blobs(
        n_samples=750, centers=centers, cluster_std=0.4, random_state=0
    )
    
    # print("X:", X)
    
    # X: [[ 0.84022039  1.14802236]
    #     [-1.15474834 -1.2041171 ]
    #     [ 0.67863613  0.72418009]
    #     ...
    #     [ 0.26798858 -1.27833405]
    #     [-0.88628813 -0.30293249]
    #     [ 0.60046048 -1.29605472]]
    
    # print("labels_true:", labels_true)
    
    # labels_true: [0 1 0 2 0 1 1 2 0 0 1 1 1 2 1 0 1 1 2 2 2 2 2 2 1 1 2 0 0 2 0 1 1 0 1 0 2
    #                 0 0 2 2 1 1 1 1 1 0 2 0 1 2 2 1 1 2 2 1 0 2 1 2 2 2 2 2 0 2 2 0 0 0 2 0 0
    #                 2 1 0 1 0 2 1 1 0 0 0 0 1 2 1 2 2 0 1 0 1 0 1 1 0 0 2 1 2 0 2 2 2 2 0 0 0
    #                 1 1 1 1 0 0 1 0 1 2 1 0 0 1 2 1 0 0 2 0 2 2 2 0 1 2 2 0 1 0 2 0 0 2 2 2 2
    #                 1 0 2 1 1 2 2 2 0 1 0 1 0 1 0 2 2 1 1 2 2 1 0 1 2 2 2 1 1 2 2 0 1 2 0 0 2
    #                 0 0 1 0 1 0 1 1 2 2 0 0 1 1 2 1 2 2 2 2 0 2 0 2 2 0 2 2 2 0 0 1 1 1 2 2 2
    #                 2 1 2 2 0 0 2 0 0 0 1 0 1 1 1 2 1 1 0 1 2 2 1 2 2 1 0 0 1 1 1 0 1 0 2 0 2
    #                 0 2 2 2 1 1 0 0 1 1 0 0 2 1 2 2 1 1 2 1 2 0 2 2 0 1 2 2 0 2 2 0 0 2 0 2 0
    #                 2 1 0 0 0 1 2 1 2 2 0 2 2 0 0 2 1 1 1 1 1 0 1 1 1 1 0 0 1 1 1 0 2 0 1 2 2
    #                 0 0 2 0 2 1 0 2 0 2 0 2 2 0 1 0 1 0 2 2 1 1 1 2 0 2 0 2 1 2 2 0 1 0 1 0 0
    #                 0 0 2 0 2 0 1 0 1 2 1 1 1 0 1 1 0 2 1 0 2 2 1 1 2 2 2 1 2 1 2 0 2 1 2 1 0
    #                 1 0 1 1 0 1 2 0 1 0 0 2 1 2 2 2 2 1 0 0 0 0 1 0 2 1 0 1 2 0 0 1 0 1 1 0 2
    #                 0 2 2 2 1 1 2 0 1 0 0 1 0 1 1 2 2 1 0 1 2 2 1 1 1 1 0 0 0 2 2 1 2 1 0 0 1
    #                 2 1 0 0 2 0 1 0 2 1 0 2 2 1 0 2 0 2 1 1 0 2 0 0 1 1 1 1 0 1 0 1 0 0 2 0 1
    #                 1 2 1 1 0 1 0 2 1 0 0 1 0 1 1 2 2 1 2 2 1 2 1 1 1 1 2 0 0 0 1 2 2 0 2 0 2
    #                 1 0 1 1 0 0 1 2 1 2 2 0 2 1 1 1 2 0 0 2 0 2 2 0 2 0 1 1 1 1 0 0 0 2 1 1 1
    #                 1 2 2 2 0 2 1 1 0 0 1 0 2 1 2 1 0 2 2 0 0 1 0 0 2 0 0 0 2 0 2 0 0 1 1 0 0
    #                 1 2 2 0 0 0 0 2 1 1 1 2 1 0 0 2 2 0 1 2 0 1 2 2 1 0 0 0 1 2 0 0 0 2 2 2 0
    #                 1 1 1 1 1 0 0 2 1 2 0 1 1 1 0 2 1 1 1 2 1 2 0 2 2 1 0 0 0 1 1 2 0 0 2 2 1
    #                 2 2 2 0 2 1 2 1 1 1 2 0 2 0 2 2 0 0 2 1 2 0 2 0 0 0 1 0 2 1 2 0 1 0 0 2 0
    #                 2 1 1 2 1 0 1 2 1 2]

    X = StandardScaler().fit_transform(X)
    
    # print("X_Standardized:", X)
    
    # X_Standardized: [[ 0.49426097  1.45106697]
    #                 [-1.42808099 -0.83706377]
    #                 [ 0.33855918  1.03875871]
    #                 ...
    #                 [-0.05713876 -0.90926105]
    #                 [-1.16939407  0.03959692]
    #                 [ 0.26322951 -0.92649949]]
    
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    
    # print("DBSCAN:", db)
    
    # DBSCAN: DBSCAN(eps=0.3, min_samples=10)
    
    # print("db.labels_:", db.labels_)
    
    # db.labels_: [ 0  1  0  2  0  1  1  2  0  0  1  1  1  2  1  0 -1  1  1  2  2  2  2  2
    #             1  1  2  0  0  2  0  1  1  0  1  0  2  0  0  2  2  1  1  1  1  1  0  2
    #             0  1  2  2  1  1  2  2  1  0  2  1  2  2  2  2  2  0  2  2  0  0  0  2
    #             0  0  2  1 -1  1  0  2  1  1  0  0  0  0  1  2  1  2  2  0  1  0  1 -1
    #             1  1  0  0  2  1  2  0  2  2  2  2 -1  0 -1  1  1  1  1  0  0  1  0  1
    #             2  1  0  0  1  2  1  0  0  2  0  2  2  2  0 -1  2  2  0  1  0  2  0  0
    #             2  2 -1  2  1 -1  2  1  1  2  2  2  0  1  0  1  0  1  0  2  2 -1  1  2
    #             2  1  0  1  2  2  2  1  1  2  2  0  1  2  0  0  2  0  0  1  0  1  0  1
    #             1  2  2  0  0  1  1  2  1  2  2  2  2  0  2  0  2  2  0  2  2  2  0  0
    #             1  1  1  2  2  2  2  1  2  2  0  0  2  0  0  0  1  0  1  1  1  2  1  1
    #             0  1  2  2  1  2  2  1  0  0  1  1  1  0  1  0  2  0  2  2  2  2  2  1
    #             1  0  0  1  1  0  0  2  1 -1  2  1  1  2  1  2  0  2  2  0  1  2  2  0
    #             2  2  0  0  2  0  2  0  2  1  0  0  0  1  2  1  2  2  0  2  2  0  0  2
    #             1  1  1  1  1  0  1  1  1  1  0  0  1  1  1  0  2  0  1  2  2  0  0  2
    #             0  2  1  0  2  0  2  0  2  2  0  1  0  1  0  2  2  1  1  1  2  0  2  0
    #             2  1  2  2  0  1  0  1  0  0  0  0  2  0  2  0  1  0  1  2  1  1  1  0
    #             1  1  0  2  1  0  2  2  1  1  2  2  2  1  2  1  2  0  2  1  2  1  0  1
    #             0  1  1  0  1  2 -1  1  0  0  2  1  2  2  2  2  1  0  0  0  0  1  0  2
    #             1  0  1  2  0  0  1  0  1  1  0 -1  0  2  2  2  1  1  2  0  1  0  0  1
    #             0  1  1  2  2 -1  0  1  2  2  1  1  1  1  0  0  0  2  2  1  2  1  0  0
    #             1  2  1  0  0  2  0  1  0  2  1  0  2  2  1  0  0  0  2  1  1  0  2  0
    #             0  1  1  1  1  0  1  0  1  0  0  2  0  1  1  2  1  1  0  1  0  2  1  0
    #             0  1  0  1  1  2  2  1  2  2  1  2  1  1  1  1  2  0  0  0  1  2  2  0
    #             2  0  2  1  0  1  1  0  0  1  2  1  2  2  0  2  1  1  1  2  0  0  2  0
    #             2  2  0  2  0  1  1  1  1  0  0  0  2  1  1  1  1  2  2  2  0  2  1  1
    #             0  0  1  0  2  1  2  1  0  2  2  0  0  1  0  0  2  0  0  0  2  0  2  0
    #             0  1  1  0  0  1  2  2  0  0  0  0  2 -1  1  1  2  1  0  0  2  2  0  1
    #             2  0  1  2  2  1  0  0 -1 -1  2  0  0  0  2 -1  2  0  1  1  1  1  1  0
    #             0  2  1  2  0  1  1  1  0  2  1  1 -1  2  1  2  0  2  2  1  0  0  0  1
    #             1  2  0  0  2  2  1  2  2  2  0  2  1  2  1  1  1  2  0  2  0  2  2  0
    #             0  2  1  2  0  2  0  0  0  1  0  2  1  2  0  1  0  0  2  0  2  1  1  2
    #             1  0  1  2  1  2]
    
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    
    # print("core_samples_mask:", core_samples_mask)
    
    # core_samples_mask: [False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False False False False False False False
    #                     False False False False False False]
    
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    # print("labels:", labels)
    
    # labels: [ 0  1  0  2  0  1  1  2  0  0  1  1  1  2  1  0 -1  1  1  2  2  2  2  2
    #         1  1  2  0  0  2  0  1  1  0  1  0  2  0  0  2  2  1  1  1  1  1  0  2
    #         0  1  2  2  1  1  2  2  1  0  2  1  2  2  2  2  2  0  2  2  0  0  0  2
    #         0  0  2  1 -1  1  0  2  1  1  0  0  0  0  1  2  1  2  2  0  1  0  1 -1
    #         1  1  0  0  2  1  2  0  2  2  2  2 -1  0 -1  1  1  1  1  0  0  1  0  1
    #         2  1  0  0  1  2  1  0  0  2  0  2  2  2  0 -1  2  2  0  1  0  2  0  0
    #         2  2 -1  2  1 -1  2  1  1  2  2  2  0  1  0  1  0  1  0  2  2 -1  1  2
    #         2  1  0  1  2  2  2  1  1  2  2  0  1  2  0  0  2  0  0  1  0  1  0  1
    #         1  2  2  0  0  1  1  2  1  2  2  2  2  0  2  0  2  2  0  2  2  2  0  0
    #         1  1  1  2  2  2  2  1  2  2  0  0  2  0  0  0  1  0  1  1  1  2  1  1
    #         0  1  2  2  1  2  2  1  0  0  1  1  1  0  1  0  2  0  2  2  2  2  2  1
    #         1  0  0  1  1  0  0  2  1 -1  2  1  1  2  1  2  0  2  2  0  1  2  2  0
    #         2  2  0  0  2  0  2  0  2  1  0  0  0  1  2  1  2  2  0  2  2  0  0  2
    #         1  1  1  1  1  0  1  1  1  1  0  0  1  1  1  0  2  0  1  2  2  0  0  2
    #         0  2  1  0  2  0  2  0  2  2  0  1  0  1  0  2  2  1  1  1  2  0  2  0
    #         2  1  2  2  0  1  0  1  0  0  0  0  2  0  2  0  1  0  1  2  1  1  1  0
    #         1  1  0  2  1  0  2  2  1  1  2  2  2  1  2  1  2  0  2  1  2  1  0  1
    #         0  1  1  0  1  2 -1  1  0  0  2  1  2  2  2  2  1  0  0  0  0  1  0  2
    #         1  0  1  2  0  0  1  0  1  1  0 -1  0  2  2  2  1  1  2  0  1  0  0  1
    #         0  1  1  2  2 -1  0  1  2  2  1  1  1  1  0  0  0  2  2  1  2  1  0  0
    #         1  2  1  0  0  2  0  1  0  2  1  0  2  2  1  0  0  0  2  1  1  0  2  0
    #         0  1  1  1  1  0  1  0  1  0  0  2  0  1  1  2  1  1  0  1  0  2  1  0
    #         0  1  0  1  1  2  2  1  2  2  1  2  1  1  1  1  2  0  0  0  1  2  2  0
    #         2  0  2  1  0  1  1  0  0  1  2  1  2  2  0  2  1  1  1  2  0  0  2  0
    #         2  2  0  2  0  1  1  1  1  0  0  0  2  1  1  1  1  2  2  2  0  2  1  1
    #         0  0  1  0  2  1  2  1  0  2  2  0  0  1  0  0  2  0  0  0  2  0  2  0
    #         0  1  1  0  0  1  2  2  0  0  0  0  2 -1  1  1  2  1  0  0  2  2  0  1
    #         2  0  1  2  2  1  0  0 -1 -1  2  0  0  0  2 -1  2  0  1  1  1  1  1  0
    #         0  2  1  2  0  1  1  1  0  2  1  1 -1  2  1  2  0  2  2  1  0  0  0  1
    #         1  2  0  0  2  2  1  2  2  2  0  2  1  2  1  1  1  2  0  2  0  2  2  0
    #         0  2  1  2  0  2  0  0  0  1  0  2  1  2  0  1  0  0  2  0  2  1  1  2
    #         1  0  1  2  1  2]

    # # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)

    # print("Estimated number of clusters: %d" % n_clusters_)
    # print("Estimated number of noise points: %d" % n_noise_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    # print(
    #     "Adjusted Mutual Information: %0.3f"
    #     % metrics.adjusted_mutual_info_score(labels_true, labels)
    # )
    # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

    # # Black removed and is used for noise instead.
    # unique_labels = set(labels)
    # colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = [0, 0, 0, 1]

    #     class_member_mask = labels == k

    #     xy = X[class_member_mask & core_samples_mask]
    #     plt.plot(
    #         xy[:, 0],
    #         xy[:, 1],
    #         "o",
    #         markerfacecolor=tuple(col),
    #         markeredgecolor="k",
    #         markersize=14,
    #     )

    #     xy = X[class_member_mask & ~core_samples_mask]
    #     plt.plot(
    #         xy[:, 0],
    #         xy[:, 1],
    #         "o",
    #         markerfacecolor=tuple(col),
    #         markeredgecolor="k",
    #         markersize=6,
    #     )

    # plt.title("Estimated number of clusters: %d" % n_clusters_)
    # plt.show()
    
main_DBSCAN()