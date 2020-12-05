from dataloaders import CIFAR_10
from dataloaders import IMDB_Wiki
from dataloaders import feature_extraction
from models import SVM
from models import PCA
from options.train_options import TrainOptions
from util import util


from tqdm import tqdm
import matplotlib.pyplot as plt
import os


class trainer():
    def __init__(self,opt, training_folder):
        self.log_folder = os.path.join(training_folder, 'logs')
        self.checkpoint_folder = os.path.join(training_folder, 'checkpoints')
        self.plot_folder = os.path.join(training_folder, 'plots')
        self.train_data = {}
        self.test_data = {}
        self.validation_data = {}
        self.svm_type = "classification"

        self.dataset = opt.dataset
        self.train_path = opt.train_path
        self.validation_percentage = opt.validation_percentage
        self.normalize_data = opt.normalize_data
        self.reduced_training_dataset = opt.reduced_training_dataset
        self.feature_extraction = opt.feature_extraction
        self.use_PCA = opt.use_PCA
        self.grid_search = opt.grid_search

    def perform_grid_search(self, c_svm, kernel_svm):
        acc_train_svm = {}
        acc_test_svm = {}
        progress_bar = tqdm(total=len(c_svm) * len(kernel_svm), desc='Grid searching for best svm')
        for kernel in kernel_svm:
            acc_train_svm[kernel] = []
            acc_test_svm[kernel] = []
            for c in c_svm:
                acc1, acc2 = SVM.train_svm(self.train_data, self.validation_data, c, kernel, self.svm_type)
                log_message = ("SVM kernel: {},\t SVM c parameter: {}\n".format(kernel, c))
                log_message = log_message + ("Training accuracy: {},\t Validation accuracy: {}\n".format(acc1, acc2))
                util.logger(log_message, self.log_folder)
                acc_train_svm[kernel].append(acc1)
                acc_test_svm[kernel].append(acc2)
                progress_bar.update(1)

        for key in acc_train_svm.keys():
            plt.plot(c_svm, acc_train_svm[key], '.-', color='red')
            plt.plot(c_svm, acc_test_svm[key], '.-', color='orange')
            plt.xlabel('c')
            plt.ylabel('Accuracy')
            plt.title("Plot of accuracy vs c for training and validation data for {} kernel".format(key))
            plt.grid()
            plot_save_path = os.path.join(self.plot_folder, ("svm_{}.png".format(key)))
            plt.savefig(plot_save_path)

    def train(self):
        # todo should create a wrapper to remove this if else
        if self.dataset == "CIFAR10":
            dataloader = CIFAR_10.cifar_dataloader(self.train_path, self.validation_percentage, i_normalize=self.normalize_data,
                                                   i_reduced_training_dataset=self.reduced_training_dataset)
            self.train_data, self.test_data, self.validation_data, label_names = dataloader.get_cifar_10()
            self.svm_type = "classification"
        elif self.dataset == "IMDB_WIKI":
            dataloader = IMDB_Wiki.imdb_wiki_dataloader(self.train_path, self.validation_percentage,
                                                        i_normalize=self.normalize_data,
                                                        i_reduced_training_dataset=self.reduced_training_dataset)
            self.train_data, self.test_data, self.validation_data = dataloader.get_imdb_wiki()
            self.svm_type = "regression"

        else:
            print("Selected dataset: {} is not implemented".format(self.dataset))
            raise NotImplementedError
            return
        if self.feature_extraction != "off":
            train_feature, validation_feature = feature_extraction.get_features(self.train_data, self.validation_data,
                                                                                feature_type=opt.feature_extraction)
            self.train_data["data"] = train_feature
            self.validation_data["data"] = validation_feature
        if self.use_PCA:
            self.train_data, self.validation_data = PCA.PCA_fun(self.train_data, self.validation_data)
        if self.grid_search:
            c_svm = [0.01, 0.1, 1, 10, 100]
            kernel_svm = ['linear', 'poly', 'rbf', 'sigmoid']
            self.perform_grid_search(c_svm, kernel_svm)


def train_CIFAR(opt, training_folder):
    log_folder = os.path.join(training_folder, 'logs')
    checkpoint_folder = os.path.join(training_folder, 'checkpoints')
    plot_folder = os.path.join(training_folder, 'plots')

    dataloader = CIFAR_10.cifar_dataloader(opt.train_path, opt.validation_percentage, i_normalize=opt.normalize_data,
                                           i_reduced_training_dataset=opt.reduced_training_dataset)
    train, test, validation, label_names = dataloader.get_cifar_10()
    train_feature, validation_feature = feature_extraction.get_features(train, validation)
    train["data"] = train_feature
    validation["data"] = validation_feature
    # c_svm = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    # kernel_svm = ['linear', 'poly', 'rbf', 'sigmoid']
    c_svm = [ 1]
    kernel_svm = ['linear']
    acc_train_svm = {}
    acc_test_svm = {}
    acc11, acc21 = SVM.svm_fun2(train, validation)
    if opt.use_PCA:
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


def train_imdb_wiki(opt, training_folder):
    log_folder = os.path.join(training_folder, 'logs')
    checkpoint_folder = os.path.join(training_folder, 'checkpoints')
    plot_folder = os.path.join(training_folder, 'plots')

    dataloader = IMDB_Wiki.imdb_wiki_dataloader(opt.train_path, opt.validation_percentage, i_normalize=opt.normalize_data,
                                           i_reduced_training_dataset=opt.reduced_training_dataset)
    train, test, validation = dataloader.get_imdb_wiki()
    # train_feature, validation_feature = feature_extraction.get_features(train, validation)
    # train["data"] = train_feature
    # validation["data"] = validation_feature
    c_svm = [0.01, 0.1, 1, 10, 100]
    kernel_svm = ['linear', 'poly', 'rbf', 'sigmoid']
    pbar = tqdm(total=len(c_svm) * len(kernel_svm), desc='Grid searching for best svr')
    acc_train_svm = {}
    acc_test_svm = {}
    if opt.use_PCA:
        train, val = PCA.PCA_fun(train, validation, None)
    for kernel in kernel_svm:
        acc_train_svm[kernel] = []
        acc_test_svm[kernel] = []
        for c in c_svm:
            acc1, acc2 = SVM.svr(train, validation, c, kernel)
            log_message = ("SVM kernel: {},\t SVM c parameter: {}\n".format(kernel, c))
            log_message = log_message + ("Training accuracy: {},\t Validation accuracy: {}\n".format(acc1, acc2))
            util.logger(log_message, log_folder)
            acc_train_svm[kernel].append(acc1)
            acc_test_svm[kernel].append(acc2)
            pbar.update(1)

    for key in acc_train_svm.keys():
        plt.plot(c_svm, acc_train_svm[key], '.-', color='red')
        plt.plot(c_svm, acc_test_svm[key], '.-', color='orange')
        plt.xlabel('c')
        plt.ylabel('Accuracy')
        plt.title("Plot of accuracy vs c for training and validation data for {} kernel".format(key))
        plt.grid()
        plot_save_path = os.path.join(plot_folder, ("svm_{}.png".format(key)))
        plt.savefig(plot_save_path)


if __name__ == "__main__":
    opt = TrainOptions().parse()
    training_folder = util.create_folders_for_training(opt.train_experiment_name)
    trainer_obj = trainer(opt, training_folder)
    trainer_obj.train()
