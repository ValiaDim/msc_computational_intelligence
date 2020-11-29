from dataloaders import CIFAR_10
from models import SVM

import matplotlib.pyplot as plt
import matplotlib as mpl


def train():
    cifar_10_dir = '/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py'
    dataloader = CIFAR_10.cifar_dataloader(cifar_10_dir, 0.2, i_preprocess=True)
    train, test, validation, label_names = dataloader.get_cifar_10()
    c_svm = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    kernel_svm = ['linear', 'poly', 'rbf', 'sigmoid']
    acc_train_svm = {}
    acc_test_svm = {}
    for kernel in kernel_svm:
        acc_train_svm[kernel] = []
        acc_test_svm[kernel] = []
        for c in c_svm:
            acc1, acc2 = SVM.svm_fun(train, validation, c, kernel)
            acc_train_svm[kernel].append(acc1)
            acc_test_svm[kernel].append(acc2)

    print("Done!")
    for key in acc_train_svm.keys():
        plt.plot(c_svm, acc_train_svm[key], '.-', color='red')
        plt.plot(c_svm, acc_test_svm[key], '.-', color='orange')
        plt.xlabel('c')
        plt.ylabel('Accuracy')
        plt.title("Plot of accuracy vs c for training and validation data for {} kernel".format(key))
        plt.grid()
        plt.show()


if __name__ == "__main__":
    train()