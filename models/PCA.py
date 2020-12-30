from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import numpy as np
import warnings

def PCA_fun(train, val, explained_variance=0.95):
    projected_train = train
    projected_val = val
    print(train['data'].shape[0])
    print(train['data'].shape[1])
    # https://faithfull.me/using-pca-when-there-are-less-samples-than-dimensions
    if train['data'].shape[0] < train['data'].shape[1]:
        warnings.warn("The data samples are less than the features! Only data samples - 1 principal components \
        can be derived")
    if explained_variance is None:
        pca = PCA().fit(train['data'])
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()
        invalid_input = True
        while invalid_input:
            components_str = input("Type the number of components you want to keep: ")
            try:
                components_int = int(components_str)
            except ValueError:
                continue
            if 0 <= components_int <= train['data'].shape[1]:
                pca = PCA(n_components=components_int)
                pca.fit(train['data'])
                projected_train['data'] = pca.transform(train['data'])
                projected_val['data'] = pca.transform(val['data'])
                break
    else:
        pca = PCA(n_components=explained_variance)
        pca.fit(train['data'])
        projected_train['data'] = pca.transform(train['data'])
        projected_val['data'] = pca.transform(val['data'])
    print("PCA is finished, selected explained variance {}. New data shape is {}".format(explained_variance, projected_train['data'].shape))
    return projected_train, projected_val


def KPCA_fun(train, val, kernel="rbf", gamma=None, components=None):
    projected_train = train
    projected_val = val
    print(train['data'].shape[0])
    print(train['data'].shape[1])
    # https://faithfull.me/using-pca-when-there-are-less-samples-than-dimensions
    if train['data'].shape[0] < train['data'].shape[1]:
        warnings.warn("The data samples are less than the features! Only data samples - 1 principal components \
        can be derived")
    if components is None:
        kpca = KernelPCA().fit(train['data'])
        plt.plot(np.cumsum(kpca.lambdas_)/sum(kpca.lambdas_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()
        invalid_input = True
        while invalid_input:
            components_str = input("Type the number of components you want to keep: ")
            try:
                components_int = int(components_str)
            except ValueError:
                continue
            if 0 <= components_int <= train['data'].shape[1]:
                kpca = KernelPCA(n_components=components_int)
                kpca.fit(train['data'])
                projected_train['data'] = kpca.transform(train['data'])
                projected_val['data'] = kpca.transform(val['data'])
                break
    else:
        kpca = KernelPCA(kernel=kernel, n_components=components, gamma=gamma)
        kpca.fit(train['data'])
        projected_train['data'] = kpca.transform(train['data'])
        projected_val['data'] = kpca.transform(val['data'])
    return projected_train, projected_val


















