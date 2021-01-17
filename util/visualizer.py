import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import numpy as np

DEBUG = False

def visualize_groundTruth(data, method, plot_folder, classes, centroids=[]):
    # method to visualize the 2D embedding, using different colors for each ground truth value
    colours = ListedColormap(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00', '#f09c9c'])
    plt.clf()
    for i in range(len(classes)):
        plt.scatter(data["data"][data["labels"] == i, 0], data["data"][data["labels"] == i, 1], cmap=colours, zorder=1)
    plt.legend(labels=classes)
    # scatter = plt.scatter(data['data'][:, 0], data['data'][:, 1], c=data['labels'], cmap=colours, zorder=1)
    if len(centroids) != 0:
        assert(len(centroids) == len(classes)), "The lenght of centroids should be the same as the number of classes"
        for i in range(len(classes)):
            plt.scatter(np.asarray(centroids)[:, 0], np.asarray(centroids)[:, 1], marker="X", zorder=2, c="#73ff00", s=200)
    plot_save_path = os.path.join(plot_folder, ("embedding_{}.png".format(method)))
    plt.savefig(plot_save_path)
    if DEBUG:
        plt.show(block=True)


def visualize_clustering(data, prediction, method, plot_folder, number_of_clusters, centroids=[]):
    # method to visualize the clustering after 2D embedding, using different colors for cluster
    # classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    if number_of_clusters == 5:
        colours = ['#377eb8', '#ff7f00', '#4daf4a',
                   '#f781bf', '#a65628']
    elif number_of_clusters == 10:
        colours = ['#377eb8', '#ff7f00', '#4daf4a',
                   '#f781bf', '#a65628', '#984ea3',
                   '#999999', '#e41a1c', '#dede00', '#f09c9c']
    elif number_of_clusters == 15:
        colours = ['#377eb8', '#ff7f00', '#4daf4a',
                   '#f781bf', '#a65628', '#984ea3',
                   '#999999', '#e41a1c', '#dede00', '#f09c9c', '#4287f5', '#bff542', '#f5b642', '#f7434c', '#06c736' ]
    plt.clf()
    for i in range(number_of_clusters):
        plt.scatter(data["data"][prediction == i, 0], data["data"][prediction == i, 1], c=colours[i])
        plt.scatter(np.asarray(centroids)[i, 0], np.asarray(centroids)[i, 1], c=colours[i], marker='x', s=200, linewidths=2)
    plot_save_path = os.path.join(plot_folder, ("clustering_{}.png".format(method)))
    plt.savefig(plot_save_path)
    if DEBUG:
        plt.show(block=True)

