from sklearn import svm
import numpy as np


def svm_fun(train, validation, c, kernel):
    svc = svm.SVC(probability=False, kernel=kernel, C=c, verbose=True)

    svc.fit(train['data'], train['labels'])

    # Find the prediction and accuracy on the training set.
    Yhat_svc_linear_train = svc.predict(train['data'])
    acc_train = np.mean(Yhat_svc_linear_train == train['labels'])
    print('Train Accuracy with kernel {0} and C {1}, = {2:f}'.format(kernel, c, acc_train))

    # Find the prediction and accuracy on the test set.
    Yhat_svc_linear_test = svc.predict(validation['data'])
    acc_validation = np.mean(Yhat_svc_linear_test == validation['labels'])
    print('Validation Accuracy with kernel {0} and C {1}, = {2:f}'.format(kernel, c, acc_validation))
    return acc_train, acc_validation