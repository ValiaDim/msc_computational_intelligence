import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import numpy as np

def visualize(data, method, plot_folder, centroids=[]):
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    colours = ListedColormap(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00', '#f09c9c'])
    scatter = plt.scatter(data['data'][:, 0], data['data'][:, 1], c=data['labels'], cmap=colours, zorder=1)
    if len(centroids) != 0:
        assert(len(centroids) == len(classes)), "The lenght of centroids should be the same as the number of classes"
        plt.scatter(np.asarray(centroids)[:, 0], np.asarray(centroids)[:, 1], marker="X", zorder=2, c='#34eb34')
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plot_save_path = os.path.join(plot_folder, ("spectral_{}.png".format(method)))
    plt.savefig(plot_save_path)
    plt.show(block=True)

