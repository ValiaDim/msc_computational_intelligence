from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score

import numpy as np


def cluster(train, type, number_of_clusters):
    # todo this should be a class
    if type == "spectral_clustering":
        clustering_model = SpectralClustering(n_clusters=number_of_clusters, assign_labels="discretize", random_state=0).fit(train["data"])
    elif type == "kmeans":
        clustering_model = KMeans(n_clusters=number_of_clusters, random_state=0).fit(train["data"])
    else:
        raise NotImplementedError
    # compute metrics
    accuracies = {}
    accuracies["homogeneity_score"] = homogeneity_score(train["labels"], clustering_model.labels_)
    accuracies["completeness_score"] = completeness_score(train["labels"], clustering_model.labels_)
    print("hey")



