from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def PCA_fun(train, val, explained_variance=0.95):
    projected_train = train
    projected_val = val
    assert train['data'].shape[0] <= train['data'].shape[1], "The data samples needs to be more than the features!"
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
        projected_val['data'] = pca.transform(train['data'])
    print("PCA is finished, selected explained variance {}. New data shape is {}".format(explained_variance, projected_train['data'].shape))
    return projected_train, projected_val














