from dataloaders import CIFAR_10
from models import SVM
from models import PCA
from options.train_options import TrainOptions
from util import util


import matplotlib.pyplot as plt
import matplotlib as mpl
import os


def train(opt, training_folder):
    log_folder = os.path.join(training_folder, 'logs')
    checkpoint_folder = os.path.join(training_folder, 'checkpoints')
    plot_folder = os.path.join(training_folder, 'plots')

    dataloader = CIFAR_10.cifar_dataloader(opt.train_path, opt.validation_percentage, i_preprocess=True,
                                           i_reduced_training_dataset=opt.reduced_training_dataset)
    train, test, validation, label_names = dataloader.get_cifar_10()
    # c_svm = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    # kernel_svm = ['linear', 'poly', 'rbf', 'sigmoid']
    c_svm = [0.01, 0.1, 1, 10, 100]
    kernel_svm = ['rbf', 'sigmoid']
    acc_train_svm = {}
    acc_test_svm = {}
    train, val = PCA.PCA_fun(train, validation)
    for kernel in kernel_svm:
        acc_train_svm[kernel] = []
        acc_test_svm[kernel] = []
        for c in c_svm:
            acc1, acc2 = SVM.svm_fun(train, validation, c, kernel)
            log_message = ("SVM kernel: {},\t SVM c parameter: {}\n".format(kernel, c))
            log_message = log_message + ("Training accuracy: {},\t Validation accuracy: {}\n".format(acc1, acc2))
            util.logger(log_message, log_folder)
            acc_train_svm[kernel].append(acc1)
            acc_test_svm[kernel].append(acc2)

    for key in acc_train_svm.keys():
        plt.plot(c_svm, acc_train_svm[key], '.-', color='red')
        plt.plot(c_svm, acc_test_svm[key], '.-', color='orange')
        plt.xlabel('c')
        plt.ylabel('Accuracy')
        plt.title("Plot of accuracy vs c for training and validation data for {} kernel".format(key))
        plt.grid()
        plot_save_path = os.path.join(plot_folder,("svm_{}.png".format(key)))
        plt.savefig(plot_save_path)
        # plt.show()


if __name__ == "__main__":
    opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
    training_folder = util.create_folders_for_training(opt.train_experiment_name)
    train(opt, training_folder)
