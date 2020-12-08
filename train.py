from dataloaders import CIFAR_10
from dataloaders import IMDB_Wiki
from dataloaders import feature_extraction
from models import SVM
from models import PCA
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
        self.train_path = opt.train_path
        self.validation_percentage = opt.validation_percentage
        self.normalize_data = opt.normalize_data
        self.reduced_training_dataset = opt.reduced_training_dataset
        self.feature_extraction = opt.feature_extraction
        self.use_PCA = opt.use_PCA
        self.grid_search = opt.grid_search
        self.load_raw_images = (opt.feature_extraction != "off")
        self.max_number_of_iter = opt.max_number_of_iter
        self.feature_layer = opt.feature_layer

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
            plt.clf()
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
            dataloader = CIFAR_10.cifar_dataloader(self.train_path, self.validation_percentage,
                                                   i_normalize=self.normalize_data,
                                                   i_reduced_training_dataset=self.reduced_training_dataset,
                                                   i_raw_images=self.load_raw_images)
            self.train_data, self.test_data, self.validation_data, label_names = dataloader.get_cifar_10()
            self.svm_type = "classification"
        elif self.dataset == "IMDB_WIKI":
            dataloader = IMDB_Wiki.imdb_wiki_dataloader(self.train_path, self.validation_percentage,
                                                        i_normalize=self.normalize_data,
                                                        i_reduced_training_dataset=self.reduced_training_dataset,
                                                        i_raw_images=self.load_raw_images,
                                                        i_feature_type=self.feature_extraction)
            self.train_data, self.test_data, self.validation_data = dataloader.get_imdb_wiki()
            self.svm_type = "regression"

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
        if self.use_PCA:
            self.train_data, self.validation_data = PCA.PCA_fun(self.train_data, self.validation_data)
        if self.grid_search:
            c_svm = [0.01, 0.1, 1, 10, 100]
            kernel_svm = ['linear', 'poly', 'rbf', 'sigmoid']
            self.perform_grid_search(c_svm, kernel_svm)


if __name__ == "__main__":
    opt = TrainOptions().parse()
    training_folder = util.create_folders_for_training(opt)
    print("Starting experiment named: {}".format(opt.train_experiment_name))
    trainer_obj = trainer(opt, training_folder)
    trainer_obj.train()
