from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding


from util.visualizer import visualize

def spectral_embedding(train, val, method, plot_folder, dimensions=2):
    projected_train = train.copy()
    projected_val = val.copy()
    if method == "Isomap":
        embedding = Isomap(n_neighbors=6, n_components=dimensions).fit(train['data'])
    elif method == "TSNE":
        embedding = TSNE(n_components=dimensions).fit(train['data'])
    elif method == "LLE":
        embedding = LocallyLinearEmbedding(n_neighbors=6, n_components=dimensions).fit(train['data'])
    elif method == "modified_LLE":
        embedding = LocallyLinearEmbedding(n_neighbors=6, n_components=dimensions, method="modified").fit(train['data'])
    elif method == "hessian_LLE":
        embedding = LocallyLinearEmbedding(n_neighbors=6, n_components=dimensions, method="hessian").fit(train['data'])
    elif method == "laplacian_eigenmaps":
        embedding = SpectralEmbedding(n_components=dimensions).fit(train['data'])
    projected_train['data'] = embedding.transform(train['data'])
    projected_val['data'] = embedding.transform(val['data'])
    # visualize(projected_train, method, plot_folder)

    return projected_train, projected_val
