import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt

import math

import random

from collections import Counter

import json

import pandas as pd

def example_DBSCAN():
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

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    # print(n_clusters_)
    
    # n_clusters_ = 3
    
    n_noise_ = list(labels).count(-1)
    
    # print(n_noise_)
    
    # n_noise_ = 18

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    print(
        "Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(labels_true, labels)
    )
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()
    
def read_workload():
    list_of_devices = []
    
    f = open('workload.json')
    data = json.load(f)

    for i in data:
        for t in data[i]:

            device = []
            device.append(i)
            device.append(t['device_id'])
            device.append(10 if t['class_of_service'] == 'standard' else 100)
            for fog_latency in range(len(t['latency'])):
                device.append(t['latency'][fog_latency])
            device.append(t['request_size'])
            device.append(t['region'])

            list_of_devices.append(device)

    devicesDf = pd.DataFrame(list_of_devices, columns=['Hour', 'Device_id', 'Class_of_service',
                            'Fog_latency_1', 'Fog_latency_2', 'Fog_latency_3', 'Cloud_latency', 'Request_size', 'Regions'])
    
    df_by_hour = dict(tuple(devicesDf.groupby('Hour')))
    # print(list(range(len(df_by_hour))))
    
    return devicesDf, df_by_hour

def generate_points(amount=100, seed='',minValue=-1, maxValue=3):
    
    if seed != '':
        random.seed(seed)
    else:
        random.seed(5)

    points = []
    priority = []

    for i in range(amount):
        point = []
        x = random.uniform(minValue,maxValue)
        y = random.uniform(minValue,maxValue)
        point.append(round(x,3))
        # point.append(round(y,3))
        points.append(point)
        
        priority.append(random.randint(0,1))

    return (points)

def calculate_params(dataset,number_of_neighbors):
    print("calculating params")
    neighbors = NearestNeighbors(number_of_neighbors)
    neighbors_fit = neighbors.fit(dataset)
    distances, indices = neighbors_fit.kneighbors(dataset)
    
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.show()
    
def calculate_distribution(dataset):
    distribution_dict = {}
    
    distribution_dict['data'] = dataset
    
    distribution_dict['data_mean'] = np.mean(dataset)
    distribution_dict['data_sd'] = np.std(dataset)
    
    # distribution_dict['prob_density'] = (np.pi*distribution_dict['data_sd']) * np.exp(-0.5*((distribution_dict['data']-distribution_dict['data_mean'])/distribution_dict['data_sd'])**2)
    
    # print("Ploting probality density")
    
    # #Plotting the Results
    # plt.plot(distribution_dict['data'], distribution_dict['prob_density'], color = 'red')
    # plt.xlabel('Data points')
    # plt.ylabel('Probability Density')
    
    # plt.show()
    
    distribution_dict['rounded_prob_density'] = []
    
    # for i in range(len(distribution_dict['prob_density'])):
    #     rounded_value = round(distribution_dict['prob_density'][i][0], 1)
    #     distribution_dict['rounded_prob_density'].append(rounded_value)
        
    for i in range(len(distribution_dict['data'])):
        rounded_value = round(distribution_dict['data'][i][0], 1)
        distribution_dict['rounded_prob_density'].append(rounded_value)
        
    distribution_dict['distribution'] = dict(Counter(distribution_dict['rounded_prob_density']))
    
    rounded_prob_density_keys = list(distribution_dict['distribution'].keys())
    
    distribution_dict['most_repetition'] = 0
    
    for key in rounded_prob_density_keys:
        if(distribution_dict['distribution'][key] > distribution_dict['most_repetition']):
            distribution_dict['most_repetition'] = distribution_dict['distribution'][key]
    
    return distribution_dict
    
def sort_array(arr):
    n = len(arr)
    # optimize code, so if the array is already sorted, it doesn't need
    # to go through the entire process
    swapped = False
    # Traverse through all array elements
    for i in range(n-1):
        # range(n) also work but outer loop will
        # repeat one time more than needed.
        # Last i elements are already in place
        for j in range(0, n-i-1):
 
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
         
        if not swapped:
            # if we haven't needed to make a single swap, we
            # can just exit the main loop.
            return

def calculate_distance(point_A, point_B):
    return math.sqrt(math.pow(point_B[0] - point_A[0], 2))

def calculate_distances_between_points(dataset):
    distance_between_all_points = []
    
    for fixed_point in dataset:
        point_distances = []
        for variable_point in dataset:
            if(fixed_point == variable_point):
                continue
            point_distances.append(calculate_distance(fixed_point, variable_point))
        distance_between_all_points.append(point_distances)
            
    smallest_n_distances = []
            
    distribution_dict = calculate_distribution(dataset)
    
    closest_n = distribution_dict['most_repetition']
    
    for point_distances in distance_between_all_points:
        sort_array(point_distances)
        closest_n_distances_list = []
        for i in range(closest_n):
            closest_n_distances_list.append(point_distances[i])
        smallest_n_distances.append(closest_n_distances_list)
    
    average_points_distances = []
    
    for closest_n_distances_list in smallest_n_distances:
        single_point_distance_sum = 0
        for point_distance in closest_n_distances_list:
            single_point_distance_sum += point_distance
        average_distance = single_point_distance_sum / closest_n
        average_points_distances.append(average_distance)
        
    all_points_distance_sum = 0
        
    for i in range(len(average_points_distances)):
        all_points_distance_sum += average_points_distances[i]
        
    all_points_average_distance = all_points_distance_sum / len(average_points_distances)
    
    return math.fabs(all_points_average_distance)
    
def calculate_clusters_centroid(dataset, dataset_labels, n_clusters):
    clusters_dict = {}
    
    # Builds a dictionary for each cluster with the points that belong to it and the amount of points
    for cluster in range(n_clusters):
        clusters_dict[cluster] = {}
        clusters_dict[cluster]['points'] = []
        clusters_dict[cluster]['n_points'] = 0
        for i in range(len(dataset_labels)):
            if(dataset_labels[i] == cluster):
                clusters_dict[cluster]['points'].append(dataset[i])
                clusters_dict[cluster]['n_points'] += 1
    
    clusters_centroids = []
    
    # Calculate the Centroids of wach cluster
    for cluster in range(n_clusters):
        centroid_point = []
        clusters_dict[cluster]['points_sum'] = 0
        for i in range(clusters_dict[cluster]['n_points']):
            clusters_dict[cluster]['points_sum'] += clusters_dict[cluster]['points'][i][0]
        centroid_point.append(clusters_dict[cluster]['points_sum']/clusters_dict[cluster]['n_points'])
        clusters_centroids.append(centroid_point)
            
    return clusters_centroids

def test_DBSCAN():
    points = generate_points()
    
    sample_dimensionality = 2
    
    average_point_distance = calculate_distances_between_points(points)
    
    default_db = DBSCAN(eps=0.044, min_samples=sample_dimensionality).fit(points)
    
    default_clusters_labels = default_db.labels_
    
    print("Labels (Default):", default_clusters_labels)
    
    default_n_clusters_ = len(set(default_clusters_labels)) - (1 if -1 in default_clusters_labels else 0)
    
    print("Clusters (Default):",default_n_clusters_)
    
    print("\n====================================================================\n")
    
    db_calculated_eps = DBSCAN(eps=average_point_distance, min_samples=sample_dimensionality).fit(points)
    
    calculated_eps_clusters_labels = db_calculated_eps.labels_
    
    print("Labels (Calculated eps):", calculated_eps_clusters_labels)
    
    calculated_eps_n_clusters_ = len(set(calculated_eps_clusters_labels)) - (1 if -1 in calculated_eps_clusters_labels else 0)
    
    print("Clusters (Calculated eps):",calculated_eps_n_clusters_)
    
    print("\n====================================================================\n")
    
    # clusters_labels_prdct = db_predicted
    
    # print("Labels Predicted:", clusters_labels_prdct)
    
    # n_clusters_prdct = len(set(clusters_labels_prdct)) - (1 if -1 in clusters_labels_prdct else 0)
    
    # print("Clusters Predicted:",n_clusters_prdct)
    
    # Joining clusters
    
    # clusters_centroids = calculate_clusters_centroid(points, clusters_labels, n_clusters_)
    
    # print("Cluster Centroids:", clusters_centroids)
    
    # print("N Centroids:", len(clusters_centroids))
    
    # db_centroids = DBSCAN(eps=0.163, min_samples=2).fit(clusters_centroids)
    
    # clusters_centroids_labels = db_centroids.labels_
    
    # print("Centroids Labels:", clusters_centroids_labels)
    
    # n_clusters_centroids_ = len(set(clusters_centroids_labels)) - (1 if -1 in clusters_centroids_labels else 0)
    
    # print("Centroids Clusters:",n_clusters_centroids_)
    
# example_DBSCAN()

test_DBSCAN()