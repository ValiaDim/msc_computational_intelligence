import argparse
import os
import warnings
from util import util


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        mode_options = ["train", "test", "inference"]
        dataset_options = ["CIFAR10", "IMDB_WIKI"]
        self.parser.add_argument('--mode', choices=mode_options, help='Select mode: ' + ' | '.join(mode_options))
        self.parser.add_argument('--dataset', choices=dataset_options, help='Select dataset: ' + ' | '.join(dataset_options))
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        if self.opt.classifier_type == "kpca_lda" and self.opt.pca_type != "off":
            warnings.warn("Cannot use kpca_lda as classifier & have a pca type on. Setting pca to off.")
            self.opt.pca_type = "off"
        self.opt.isTrain = self.isTrain   # train or test
        return self.opt
