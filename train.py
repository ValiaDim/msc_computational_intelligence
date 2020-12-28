from dataloaders import CIFAR_10
from dataloaders import IMDB_Wiki
from dataloaders import feature_extraction
from models import SVM
from models import PCA
from models import LDA
from options.train_options import TrainOptions
from util import util

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
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
        self.bin_ages = opt.bin_ages
        self.train_path = opt.train_path
        self.validation_percentage = opt.validation_percentage
        self.normalize_data = opt.normalize_data
        self.reduced_training_dataset = opt.reduced_training_dataset
        self.feature_extraction = opt.feature_extraction
        self.pca_type = opt.pca_type
        self.classifier_type = opt.classifier_type
        self.grid_search = opt.grid_search
        self.load_raw_images = (opt.feature_extraction != "off")
        self.max_number_of_iter = opt.max_number_of_iter
        self.feature_layer = opt.feature_layer

    def perform_svm_grid_search(self, c_svm, kernel_svm):
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
            plt.clf()
            plt.plot(c_svm, acc_train_svm[key], '.-', color='red')
            plt.plot(c_svm, acc_test_svm[key], '.-', color='orange')
            plt.xlabel('c')
            plt.ylabel('Accuracy')
            plt.title("Plot of accuracy vs c for training and validation data for {} kernel".format(key))
            plt.grid()
            plot_save_path = os.path.join(self.plot_folder, ("svm_{}.png".format(key)))
            plt.savefig(plot_save_path)

    def perform_lda_grid_search(self):
        acc1, acc2 = LDA.lda_classifier(self.train_data, self.validation_data)
        log_message = "LDA classifier: "
        log_message = log_message + ("Training accuracy: {},\t Validation accuracy: {}\n".format(acc1, acc2))
        util.logger(log_message, self.log_folder)

    def perform_kpca_lda_grid_search(self, kernel_kpca, gamma_kpca, number_of_components_kpca, solver_lda, shrinkage_lda):
        acc_train_kpca_lda = {}
        acc_test_kpca_lda = {}
        experiment_number = len(kernel_kpca) * len(gamma_kpca) * len(number_of_components_kpca) * \
                                                len(solver_lda) * len(shrinkage_lda)
        if "svd" in solver_lda:
            experiment_number = len(kernel_kpca) * len(gamma_kpca) * len(number_of_components_kpca) * \
                                ((len(solver_lda) -1) * len(shrinkage_lda) + 1)
        progress_bar = tqdm(total=experiment_number, desc='Grid searching for best kpca+lda ')
        for kernel in kernel_kpca:
            acc_train_kpca_lda[kernel] = []
            acc_test_kpca_lda[kernel] = []
            for gamma in gamma_kpca:
                for com_num in number_of_components_kpca:
                    reduced_train_data, reduced_validation_data = PCA.KPCA_fun(self.train_data.copy(), self.validation_data.copy(),
                                                                         kernel=kernel, gamma=gamma, components=com_num)
                    log_message = "Used dimensionality reduction, via kernel PCA with kernel: {}, \t gamma: {}, " \
                                  "\t number of components: {}\n".format(kernel, gamma, com_num)
                    util.logger(log_message, self.log_folder, change_classifier=False)
                    for solver in solver_lda:
                        if solver == "svd":
                            acc1, acc2 = LDA.lda_classifier(reduced_train_data, reduced_validation_data, solver)
                            log_message = ("LDA solver: {}\n".format(solver))
                            log_message = log_message + ("Training accuracy: {},\t Validation accuracy: {}\n".format(acc1, acc2))
                            util.logger(log_message, self.log_folder)
                        else:
                            for shrinkage in shrinkage_lda:
                                acc1, acc2 = LDA.lda_classifier(reduced_train_data, reduced_validation_data, solver, shrinkage)
                                log_message = ("LDA solver: {} \t shrinkage: {} \n".format(solver, shrinkage))
                                log_message = log_message + ("Training accuracy: {},\t Validation accuracy: {}\n".format(acc1, acc2))
                                util.logger(log_message, self.log_folder)
                        acc_train_kpca_lda[kernel].append(acc1)
                        acc_test_kpca_lda[kernel].append(acc2)
                        progress_bar.update(1)


    def train(self):
        # todo should create a wrapper to remove this if else
        if self.dataset == "CIFAR10":
            dataloader = CIFAR_10.cifar_dataloader(self.train_path, self.validation_percentage,
                                                   i_normalize=self.normalize_data,
                                                   i_reduced_training_dataset=self.reduced_training_dataset,
                                                   i_raw_images=self.load_raw_images)
            self.train_data, self.test_data, self.validation_data, label_names = dataloader.get_cifar_10()
            self.svm_type = "classification"
        elif self.dataset == "IMDB_WIKI" and not self.bin_ages:
            dataloader = IMDB_Wiki.imdb_wiki_dataloader(self.train_path, self.validation_percentage,
                                                        i_normalize=self.normalize_data,
                                                        i_reduced_training_dataset=self.reduced_training_dataset,
                                                        i_raw_images=self.load_raw_images,
                                                        i_feature_type=self.feature_extraction)
            self.train_data, self.test_data, self.validation_data = dataloader.get_imdb_wiki()
            self.svm_type = "regression"
        elif self.dataset == "IMDB_WIKI" and self.bin_ages:
            dataloader = IMDB_Wiki.imdb_wiki_dataloader(self.train_path, self.validation_percentage,
                                                        i_normalize=self.normalize_data,
                                                        i_reduced_training_dataset=self.reduced_training_dataset,
                                                        i_raw_images=self.load_raw_images,
                                                        i_feature_type=self.feature_extraction, bin_ages=True)
            self.train_data, self.test_data, self.validation_data = dataloader.get_imdb_wiki()
            self.svm_type = "classification"

        else:
            print("Selected dataset: {} is not implemented".format(self.dataset))
            raise NotImplementedError
            return
        if self.feature_extraction != "off":
            train_feature, validation_feature = feature_extraction.get_features(self.train_data, self.validation_data,
                                                                                feature_type=self.feature_extraction,
                                                                                feature_layer=self.feature_layer)
            self.train_data["data"] = np.stack(train_feature, axis=0)
            self.validation_data["data"] = np.stack(validation_feature, axis=0)
        if self.pca_type == "PCA":
            self.train_data, self.validation_data = PCA.PCA_fun(self.train_data, self.validation_data)
        elif self.pca_type == "KPCA":
            log_message = "Used dimensionality reduction, via kernel PCA with kernel {}\n".format("rbf")
            util.logger(log_message, self.log_folder, change_classifier=False)
            self.train_data, self.validation_data = PCA.KPCA_fun(self.train_data, self.validation_data)
        else:
            if self.pca_type != "off":
                print("Selected dimensionality reduction: {} is not implemented".format(self.pca_type))
                raise NotImplementedError
                return
        if self.grid_search:
            if self.classifier_type == "svm":
                c_svm = [0.01, 0.1, 1, 10, 100]
                kernel_svm = ['linear', 'poly', 'rbf', 'sigmoid']
                self.perform_svm_grid_search(c_svm, kernel_svm)
            elif self.classifier_type == "lda":
                self.perform_lda_grid_search()
            elif self.classifier_type == "kpca_lda":
                kernel_kpca = ['poly', 'rbf', 'sigmoid']
                gamma_kpca = [None] #[None, 0.01, 0.1, 1]
                number_of_components_kpca = [None] #[None, 10, 50, 100] # [None,10,20,30,40,50]#
                solver_lda = ['svd']#['svd', 'lsqr', 'eigen']
                shrinkage_lda =['auto']#['auto', 0, 1, 0.01]
                self.perform_kpca_lda_grid_search(kernel_kpca, gamma_kpca, number_of_components_kpca, solver_lda, shrinkage_lda)


if __name__ == "__main__":
    opt = TrainOptions().parse()
    training_folder = util.create_folders_for_training(opt)
    print("Starting experiment named: {}".format(opt.train_experiment_name))
    trainer_obj = trainer(opt, training_folder)
    trainer_obj.train()
