import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os


def visualize(data, method, plot_folder):
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    colours = ListedColormap(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                          '#999999', '#e41a1c', '#dede00', '#34eb34'])
    scatter = plt.scatter(data['data'][:, 0], data['data'][:, 1], c=data['labels'], cmap=colours)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plot_save_path = os.path.join(plot_folder, ("spectral_{}.png".format(method)))
    plt.savefig(plot_save_path)
    plt.show(block=True)

