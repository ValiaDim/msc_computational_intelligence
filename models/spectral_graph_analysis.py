from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle, islice

import os

def spectral_embedding(train, method, plot_folder, dimensions=2):
    projected_train = train.copy()
    if method == "Isomap":
        embedding = Isomap(n_components=dimensions)
    elif method == "TSNE":
        embedding = TSNE(n_components=dimensions)
    elif method == "LLE":
        embedding = LocallyLinearEmbedding(n_components=dimensions)
    projected_train['data'] = embedding.fit_transform(train['data'])
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00','#34eb34']),
                                  int(max(train['labels']) + 1))))
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    scatter = plt.scatter(projected_train['data'][:, 0], projected_train['data'][:,1], s=10, color=colors[train['labels']])
    plt.legend(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"), loc="center left")
    plt.show(block=True)
    plt.title("Scatter plot of 2-dimensional embeddings using spectral embedding technique: {}".format(method))
    plot_save_path = os.path.join(plot_folder, ("spectral_{}.png".format(method)))
    plt.savefig(plot_save_path)
    return projected_train
