import numpy as np
import matplotlib.pyplot as plt
from dataloaders import utils


# code from https://github.com/snatch59/load-cifar-10
"""
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 
training images and 10000 test images.
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains 
exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random 
order, but some training batches may contain more images from one class than another. Between them, the training 
batches contain exactly 5000 images from each class.
"""

class cifar_dataloader():
    def __init__(self, i_data_dir, i_percentage_validation, i_preprocess):
        self.cifar_train = {}
        self.cifar_test = {}
        self.cifar_validation = {}
        self.cifar_label_names = []
        self.percentage_validation = i_percentage_validation
        self.data_dir = i_data_dir
        self.preprocess = i_preprocess

    def preprocess_data(self):
        self.cifar_train['data'] = self.cifar_train['data'].astype(np.float)
        self.cifar_test['data'] = self.cifar_test['data'].astype(np.float)
        self.cifar_validation['data'] = self.cifar_validation['data'].astype(np.float)
        print(self.cifar_train['data'].shape)
        print(self.cifar_test['data'].shape)
        print(self.cifar_validation['data'].shape)

        self.cifar_train['data'] = np.reshape(self.cifar_train['data'], (self.cifar_train['data'].shape[0], -1))
        self.cifar_test['data'] = np.reshape(self.cifar_test['data'], (self.cifar_test['data'].shape[0], -1))
        self.cifar_validation['data'] = np.reshape(self.cifar_validation['data'], (self.cifar_validation['data'].shape[0], -1))
        print(self.cifar_train['data'].shape)
        print(self.cifar_train['data'][0])

        # Normalize
        self.cifar_train['data'] = ((self.cifar_train['data'] / 255) * 2) - 1  # better to 0..1 think of it
        self.cifar_test['data'] = ((self.cifar_test['data'] / 255) * 2) - 1  # better to 0..1 think of it
        self.cifar_validation['data'] = ((self.cifar_validation['data'] / 255) * 2) - 1  # better to 0..1 think of it

        print(self.cifar_train['data'].shape)
        print(self.cifar_train['data'][0])

    def create_validation_set(self):
        train_before_length = len(self.cifar_train['filenames'])
        validation_length = int(self.percentage_validation * train_before_length)
        training_length = train_before_length - validation_length
        # validation_length = 10000
        self.cifar_validation['data'] = self.cifar_train['data'][:validation_length, :, :, :]
        self.cifar_validation['filenames'] = self.cifar_train['filenames'][:validation_length]
        self.cifar_validation['labels'] = self.cifar_train['labels'][:validation_length]

        self.cifar_train['data'] = self.cifar_train['data'][validation_length:, :, :, :]
        self.cifar_train['filenames'] = self.cifar_train['filenames'][validation_length:]
        self.cifar_train['labels'] = self.cifar_train['labels'][validation_length:]

    def load_cifar_10_data(self, negatives=False):
        """
        Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
        """

        # get the meta_data_dict
        # num_cases_per_batch: 1000
        # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # num_vis: :3072

        meta_data_dict = utils.unpickle(self.data_dir + "/batches.meta")
        self.cifar_label_names = meta_data_dict[b'label_names']
        self.cifar_label_names = np.array(self.cifar_label_names)

        # training data
        cifar_train_data = None
        cifar_train_filenames = []
        cifar_train_labels = []

        # cifar_train_data_dict
        # 'batch_label': 'training batch 5 of 5'
        # 'data': ndarray
        # 'filenames': list
        # 'labels': list

        for i in range(1, 6):
            cifar_train_data_dict = utils.unpickle(self.data_dir + "/data_batch_{}".format(i))
            if i == 1:
                cifar_train_data = cifar_train_data_dict[b'data']
            else:
                cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
            cifar_train_filenames += cifar_train_data_dict[b'filenames']
            cifar_train_labels += cifar_train_data_dict[b'labels']

        cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
        if negatives:
            cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
        else:
            cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
        cifar_train_filenames = np.array(cifar_train_filenames)
        cifar_train_labels = np.array(cifar_train_labels)

        # test data
        # cifar_test_data_dict
        # 'batch_label': 'testing batch 1 of 1'
        # 'data': ndarray
        # 'filenames': list
        # 'labels': list

        cifar_test_data_dict = utils.unpickle(self.data_dir + "/test_batch")
        cifar_test_data = cifar_test_data_dict[b'data']
        cifar_test_filenames = cifar_test_data_dict[b'filenames']
        cifar_test_labels = cifar_test_data_dict[b'labels']

        cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
        if negatives:
            cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
        else:
            cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
        cifar_test_filenames = np.array(cifar_test_filenames)
        cifar_test_labels = np.array(cifar_test_labels)

        # add them in dict for better code
        self.cifar_train['data'] = cifar_train_data
        self.cifar_train['filenames'] = cifar_train_filenames
        self.cifar_train['labels'] = cifar_train_labels
        self.cifar_test['data'] = cifar_test_data
        self.cifar_test['filenames'] = cifar_test_filenames
        self.cifar_test['labels'] = cifar_test_labels
        if self.percentage_validation:
            assert (self.percentage_validation <= 1, "Percentage validation needs to be less than 1")
            self.create_validation_set()

    def get_cifar_10(self):
        self.load_cifar_10_data()
        if self.preprocess:
            self.preprocess_data()
        return self.cifar_train, self.cifar_test, self.cifar_validation, self.cifar_label_names


if __name__ == "__main__":
    """show it works"""

    cifar_10_dir = '/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py'
    dataloader = cifar_dataloader(cifar_10_dir, 0.2, i_preprocess=False)
    cifar_train, cifar_test, cifar_validation, label_names = dataloader.get_cifar_10()

    print("Train data: ", cifar_train['data'].shape)
    print("Train filenames: ", cifar_train['filenames'].shape)
    print("Train labels: ", cifar_train['labels'].shape)
    print("Test data: ", cifar_test['data'].shape)
    print("Test filenames: ", cifar_test['filenames'].shape)
    print("Test labels: ", cifar_test['labels'].shape)
    if cifar_validation:
        print("Validation data: ", cifar_validation['data'].shape)
        print("Validation filenames: ", cifar_validation['filenames'].shape)
        print("Validation labels: ", cifar_validation['labels'].shape)
    print("Label names: ", label_names.shape)

    # Don't forget that the label_names and filesnames are in binary and need conversion if used.

    # display some random training images in a 25x25 grid
    num_plot = 5
    f, ax = plt.subplots(num_plot, num_plot)
    for m in range(num_plot):
        for n in range(num_plot):
            idx = np.random.randint(0, cifar_train['data'].shape[0])
            ax[m, n].imshow(cifar_train['data'][idx])
            ax[m, n].get_xaxis().set_visible(False)
            ax[m, n].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0)
    plt.show()