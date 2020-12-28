from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import numpy as np


def lda_classifier(train, validation, solver="svd", shrinkage=None, verbose=False):
    lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
    lda.fit(train['data'], train['labels'])
    # Find the prediction and accuracy on the training set.
    Yhat_svc_linear_train = lda.predict(train['data'])
    acc_train = np.mean(Yhat_svc_linear_train == train['labels'])

    # Find the prediction and accuracy on the test set.
    Yhat_svc_linear_test = lda.predict(validation['data'])
    acc_validation = np.mean(Yhat_svc_linear_test == validation['labels'])
    if verbose:
        print('Train Accuracy for lda classifier, = {0:f}'.format(acc_train))
        print('Validation Accuracy for lda classifier, = {0:f}'.format(acc_validation))
    return acc_train, acc_validation

