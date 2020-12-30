from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid

import numpy as np


def nearest_neighbor_classifier(train, validation, k=1, verbose=False):
    nearest_neightbor = KNeighborsClassifier(n_neighbors=k)
    nearest_neightbor.fit(train['data'], train['labels'])
    # Find the prediction and accuracy on the training set.
    Yhat_svc_linear_train = nearest_neightbor.predict(train['data'])
    acc_train = np.mean(Yhat_svc_linear_train == train['labels'])

    # Find the prediction and accuracy on the test set.
    Yhat_svc_linear_test = nearest_neightbor.predict(validation['data'])
    acc_validation = np.mean(Yhat_svc_linear_test == validation['labels'])
    if verbose:
        print('Train Accuracy for lda classifier, = {0:f}'.format(acc_train))
        print('Validation Accuracy for lda classifier, = {0:f}'.format(acc_validation))
    return acc_train, acc_validation


def nearest_centroid_classifier(train, validation, verbose=False):
    nearest_centroid = NearestCentroid()
    nearest_centroid.fit(train['data'], train['labels'])
    # Find the prediction and accuracy on the training set.
    Yhat_svc_linear_train = nearest_centroid.predict(train['data'])
    acc_train = np.mean(Yhat_svc_linear_train == train['labels'])

    # Find the prediction and accuracy on the test set.
    Yhat_svc_linear_test = nearest_centroid.predict(validation['data'])
    acc_validation = np.mean(Yhat_svc_linear_test == validation['labels'])
    if verbose:
        print('Train Accuracy for lda classifier, = {0:f}'.format(acc_train))
        print('Validation Accuracy for lda classifier, = {0:f}'.format(acc_validation))
    return acc_train, acc_validation