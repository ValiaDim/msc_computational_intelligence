from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # classifier types contain both classification and clustering methods
        classifier_types = ["svm", "lda", "kpca_lda", "nearest_neighbor", "nearest_centroid", "kmeans", "spectral_clustering"]
        dimentionality_reduction_methods = ["off", "PCA", "KPCA", "Isomap", "LLE", "modified_LLE", "hessian_LLE", "TSNE", "laplacian_eigenmaps"]
        feature_extraction_methods = ["off", "HOG", "DEX", "VGG19", "MobileNetV2"]
        feature_layer = ["fc1", "fc2", "cls", "conv", "vgg_layer1", "vgg_layer2", "vgg_layer3", "vgg_layer4",
                         "vgg_layer5", "MobileNet_layer1", "MobileNet_layer2", "MobileNet_layer3", "MobileNet_layer4",
                         "MobileNet_layer5", "MobileNet_layer6"]
        self.parser.add_argument('--train_path', help="The root path of dataset")
        self.parser.add_argument('--grid_search', action='store_true', help="Activate grid search for the "
                                                                            "hyperparameter search")
        self.parser.add_argument('--reduced_training_dataset', default=None, type=int, help="Use reduced dataset for "
                                                                                         "hyperparameter search. "
                                                                                         "Works only combined with "
                                                                                         "grid_search parameter. "
                                                                                  "Default is None (using all the data)")
        self.parser.add_argument('--validation_percentage', default=0.2, help="Add the percentage of the validation "
                                                                              "set. Default is: 0.2")
        self.parser.add_argument('--normalize_data', action='store_true', help="If you want to normalize the data "
                                                                                   "to 0...1.")
        self.parser.add_argument('--train_experiment_name', help="Optionally add the experiment name. This will be the "
                                                                 "folder name under .trainings folder")
        self.parser.add_argument('--classifier_type', default="svm", choices=classifier_types,
                                 help='Select classifier type: ' + ' | '.join(classifier_types) )
        self.parser.add_argument('--dimentionality_reduction', default="off", choices=dimentionality_reduction_methods,
                                 help='Select dimensionality reduction method: ' + ' | '.join(dimentionality_reduction_methods))
        self.parser.add_argument('--feature_extraction', default="off", choices=feature_extraction_methods,
                                 help='Select feature extraction method: ' + ' | '.join(feature_extraction_methods))
        self.parser.add_argument('--feature_layer', default="conv", choices=feature_layer,
                                 help='Select from where to take the features: ' + ' | '.join(feature_layer))
        self.parser.add_argument('--max_number_of_iter', default=100000, help="Add the max number of iterations for "
                                                                              "the svm. Default is: 100000")
        self.isTrain = True
