from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def PCA_fun(train, val):
    projected_train = train
    projected_val = val
    assert (train['data'].shape[0] <= train['data'].shape[1], "The data samples needs to be more than the features!")
    pca = PCA().fit(train['data'])
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    # plt.show()
    # Reduce to 500 dimensions
    pca = PCA(n_components=500)
    pca.fit(train['data'])
    projected_train['data'] = pca.transform(train['data'])
    projected_val['data'] = pca.transform(val['data'])
    print("PCA OK")
    print(projected_train['data'].shape)
    return projected_train, projected_val