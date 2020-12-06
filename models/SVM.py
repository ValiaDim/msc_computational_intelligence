from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVR

import numpy as np


def train_svm(train, validation, c, kernel, type):
    # todo this should be a class
    if type == "classification":
        return svm_classifier(train, validation, c, kernel)
    if type == "regression":
        return svr_regressor(train, validation, c, kernel) # todo regression has more arguments!
    else:
        raise NotImplementedError


def svm_classifier(train, validation, c, kernel, verbose=False):
    svc = svm.SVC(probability=False, kernel=kernel, C=c, verbose=False, max_iter=100000)

    svc.fit(train['data'], train['labels'])

    # Find the prediction and accuracy on the training set.
    Yhat_svc_linear_train = svc.predict(train['data'])
    acc_train = np.mean(Yhat_svc_linear_train == train['labels'])


    # Find the prediction and accuracy on the test set.
    Yhat_svc_linear_test = svc.predict(validation['data'])
    acc_validation = np.mean(Yhat_svc_linear_test == validation['labels'])
    if verbose:
        print('Train Accuracy with kernel {0} and C {1}, = {2:f}'.format(kernel, c, acc_train))
        print('Validation Accuracy with kernel {0} and C {1}, = {2:f}'.format(kernel, c, acc_validation))
    return acc_train, acc_validation


def svr_regressor(train, validation, c, kernel, verbose=False):
    clf = SVR(kernel=kernel, C=c, gamma=0.1, epsilon=.1, max_iter=100000)
    clf.fit(train['data'], train['labels'])
    # Find the prediction and accuracy on the training set.
    Yhat_svc_linear_train = clf.predict(train['data'])
    acc_train = np.mean(abs(Yhat_svc_linear_train - train['labels']))

    # Find the prediction and accuracy on the test set.
    Yhat_svc_linear_test = clf.predict(validation['data'])
    acc_validation = np.mean(abs(Yhat_svc_linear_test - validation['labels']))
    if verbose:
        print('Train error with kernel {0} and C {1}, = {2:f}'.format(kernel, c, acc_train))
        print('Validation error with kernel {0} and C {1}, = {2:f}'.format(kernel, c, acc_validation))
    return acc_train, acc_validation


def svm_fun2(train, validation):
    clf = LinearSVC(C=1)
    clf.fit(train['data'], train['labels'])
    # Find the prediction and accuracy on the training set.
    Yhat_svc_linear_train = clf.predict(train['data'])
    acc_train = np.mean(Yhat_svc_linear_train == train['labels'])
    print('Train Accuracy LINEAR c 1, = {0:f}'.format(acc_train))

    # Find the prediction and accuracy on the test set.
    Yhat_svc_linear_test = clf.predict(validation['data'])
    acc_validation = np.mean(Yhat_svc_linear_test == validation['labels'])
    print('Validation Accuracy LINEAR c 1, = {0:f}'.format(acc_validation))
    return acc_train, acc_validation