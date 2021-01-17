from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding


from util.visualizer import *


def spectral_embedding(train, val, method, plot_folder, neighbors, classes, dimensions=2):
    projected_train = train.copy()
    projected_val = val.copy()
    if method == "Isomap":
        embedding = Isomap(n_neighbors=neighbors, n_components=dimensions).fit(train['data'])
        projected_train['data'] = embedding.transform(train['data'])
        projected_val['data'] = embedding.transform(val['data'])
    elif method == "TSNE":
        embedding = TSNE(n_components=dimensions)
        projected_train['data'] = embedding.fit_transform(train['data'])
        projected_val['data'] = embedding.fit_transform(val['data'])
    elif method == "LLE":
        embedding = LocallyLinearEmbedding(n_neighbors=neighbors, n_components=dimensions).fit(train['data'])
        projected_train['data'] = embedding.transform(train['data'])
        projected_val['data'] = embedding.transform(val['data'])
    elif method == "modified_LLE":
        embedding = LocallyLinearEmbedding(n_neighbors=neighbors, n_components=dimensions, method="modified").fit(train['data'])
        projected_train['data'] = embedding.transform(train['data'])
        projected_val['data'] = embedding.transform(val['data'])
    elif method == "hessian_LLE":
        embedding = LocallyLinearEmbedding(n_neighbors=neighbors, n_components=dimensions, method="hessian").fit(train['data'])
        projected_train['data'] = embedding.transform(train['data'])
        projected_val['data'] = embedding.transform(val['data'])
    elif method == "laplacian_eigenmaps":
        embedding = SpectralEmbedding(n_components=dimensions)
        projected_train['data'] = embedding.fit_transform(train['data'])
        projected_val['data'] = embedding.fit_transform(val['data'])
    visualize_groundTruth(projected_train, method+"_training", plot_folder, classes)
    visualize_groundTruth(projected_val, method+"_validation", plot_folder, classes)

    return projected_train, projected_val
