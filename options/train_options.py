from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--train_path', required=True, help="The root path of dataset")
        self.parser.add_argument('--grid_search', action='store_true', help="Activate grid search for the "
                                                                            "hyperparameter search")
        self.parser.add_argument('--reduced_training_dataset', action='store_true', help="Use reduced dataset for "
                                                                                         "hyperparameter search. "
                                                                                         "Works only combined with "
                                                                                         "grid_search parameter")
        self.parser.add_argument('--validation_percentage', default=0.2, help="Add the percentage of the validation "
                                                                              "set. Default is: 0.2")
        self.parser.add_argument('--train_experiment_name', help="Optionally add the experiment name. This will be the "
                                                                 "folder name under .trainings folder")
        self.parser.add_argument('--use_PCA', action='store_true', help="Use PCA for dimensionality reduction")
        self.isTrain = True
