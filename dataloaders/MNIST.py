from keras.datasets import mnist
from matplotlib import pyplot

import numpy as np


class MNIST_dataloader():
    def __init__(self, i_percentage_validation, i_normalize=True, i_reduced_training_dataset=None,
                 i_raw_images=False):
        self.MNIST_train = {}
        self.MNIST_test = {}
        self.MNIST_validation = {}
        self.MNIST_label_names = [0,1,2,3,4,5,6,7,8,9]
        self.percentage_validation = i_percentage_validation
        self.normalize = i_normalize
        self.reduced_training_dataset = i_reduced_training_dataset
        self.raw_images = i_raw_images

    def preprocess_data(self):
        self.MNIST_train['data'] = self.MNIST_train['data'].astype(np.float)
        self.MNIST_test['data'] = self.MNIST_test['data'].astype(np.float)
        self.MNIST_validation['data'] = self.MNIST_validation['data'].astype(np.float)

        self.MNIST_train['data'] = np.reshape(self.MNIST_train['data'], (self.MNIST_train['data'].shape[0], -1))
        self.MNIST_test['data'] = np.reshape(self.MNIST_test['data'], (self.MNIST_test['data'].shape[0], -1))
        self.MNIST_validation['data'] = np.reshape(self.MNIST_validation['data'], (self.MNIST_validation['data'].shape[0], -1))

        # Normalize
        if self.normalize:
            self.MNIST_train['data'] = (self.MNIST_train['data'] / 255)
            self.MNIST_test['data'] = (self.MNIST_test['data'] / 255)
            self.MNIST_validation['data'] = (self.MNIST_validation['data'] / 255)

    def create_validation_set(self):
        train_before_length = len(self.MNIST_train['labels'])
        validation_length = int(self.percentage_validation * train_before_length)
        training_length = train_before_length - validation_length
        # validation_length = 10000
        self.MNIST_validation['data'] = self.MNIST_train['data'][:validation_length, :, :]
        self.MNIST_validation['labels'] = self.MNIST_train['labels'][:validation_length]

        self.MNIST_train['data'] = self.MNIST_train['data'][validation_length:, :, :]
        self.MNIST_train['labels'] = self.MNIST_train['labels'][validation_length:]

    def reduce_dataset(self):
        training_lenght = self.reduced_training_dataset
        validation_lenght = int(self.percentage_validation*training_lenght)
        self.MNIST_validation['data'] = self.MNIST_validation['data'][:validation_lenght, :, :]
        self.MNIST_validation['labels'] = self.MNIST_validation['labels'][:validation_lenght]

        self.MNIST_train['data'] = self.MNIST_train['data'][:training_lenght, :, :]
        self.MNIST_train['labels'] = self.MNIST_train['labels'][:training_lenght]

    def load_MNIST_data(self):
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        # add them in dict for better code
        self.MNIST_train['data'] = train_X
        self.MNIST_train['labels'] = train_y
        self.MNIST_test['data'] = test_X
        self.MNIST_test['labels'] = test_y
        if self.percentage_validation:
            assert 0 <= self.percentage_validation <= 1, "Percentage validation needs to be less than 1"
            if self.percentage_validation < 0 or self.percentage_validation >= 1:
                self.percentage_validation = 0.2
                print("Setting to 0.2")
            self.create_validation_set()
        if self.reduced_training_dataset is not None:
            self.reduce_dataset()

    def get_MNIST(self):
        self.load_MNIST_data()
        if not self.raw_images:
            self.preprocess_data()
        print("Loaded dataset with sizes:")
        print("Training: {}".format(self.MNIST_train["data"].shape))
        print("Testing: {}".format(self.MNIST_test["data"].shape))
        print("Validation: {}".format(self.MNIST_validation["data"].shape))
        return self.MNIST_train, self.MNIST_test, self.MNIST_validation, self.MNIST_label_names


if __name__ == "__main__":
    """show it works"""

    dataloader = MNIST_dataloader(i_percentage_validation=0.2, i_raw_images=True)
    MNIST_train, MNIST_test, MNIST_validation, label_names = dataloader.get_MNIST()
    for i in range(9):
        pyplot.subplot(330 + 1 + i)
    pyplot.imshow(MNIST_train['data'][i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()


