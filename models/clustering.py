from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import silhouette_score

import numpy as np
import math

from util.visualizer import *


def cluster(train, val, type, number_of_clusters, plot_folder):
    # todo this should be a class
    if type == "spectral_clustering":
        clustering_model = SpectralClustering(n_clusters=number_of_clusters, assign_labels="discretize", random_state=0).fit(train["data"])
    elif type == "kmeans":
        clustering_model = KMeans(n_clusters=number_of_clusters, random_state=0).fit(train["data"])
    else:
        raise NotImplementedError
    # compute metrics
    accuracies = {}
    random_array = np.random.randint(9, size=train["labels"].shape)
    centroids = find_centroids(number_of_clusters, train, clustering_model.labels_)
    visualize_clustering(train, clustering_model.labels_, "spectral clustering", plot_folder, centroids)
    test_classifications = cluster_test(val, centroids)
    accuracies["random_score"] = homogeneity_score(train["labels"], random_array)
    accuracies["v_measure_score"] = v_measure_score(train["labels"], clustering_model.labels_)
    accuracies["homogeneity_score"] = homogeneity_score(train["labels"], clustering_model.labels_)
    accuracies["completeness_score"] = completeness_score(train["labels"], clustering_model.labels_)
    accuracies["silhouette_score"] = silhouette_score(train["data"], clustering_model.labels_)

    accuracies["v_measure_score_test"] = v_measure_score(val["labels"], test_classifications)
    accuracies["homogeneity_score_test"] = homogeneity_score(val["labels"], test_classifications)
    accuracies["completeness_score_test"] = completeness_score(val["labels"], test_classifications)
    accuracies["silhouette_score_test"] = silhouette_score(val["data"], test_classifications)


def cluster_test(val, centroids):
    classifications = []
    for val_point in val["data"]:
        distances = []
        for centroid in centroids:
            distances.append(eucledian(val_point, centroid))
        classifications.append(np.argmin(distances))
    return classifications

def find_centroids(n_clusters, train, prediction):
    centroids = []
    for i in range(n_clusters):
        x0 = train["data"][prediction == i]
        center = np.mean(x0, axis=0)
        centroids.append(center)
    return centroids


def eucledian(point1, point2):
    return math.sqrt((point1[0] - point2[0])*(point1[0] - point2[0]) + (point1[1] - point2[1])*(point1[1] - point2[1]))



