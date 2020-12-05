# CIFAR_10 dataset:
python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset --train_experiment_name "CIFAR10_normalized"
python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset --normalize_data False --train_experiment_name "CIFAR10_non_normalized"
python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset --use_PCA --train_experiment_name "CIFAR10_normalized_PCA"
python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset --feature_extraction HOG --train_experiment_name "CIFAR10_normalized_HOG"
python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset --feature_extraction HOG --use_PCA --train_experiment_name "CIFAR10_normalized_HOG_PCA"



# imdb_wiki dataset:
python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset --train_experiment_name "IMDB_WIKI_normalized"
python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset --normalize_data False --train_experiment_name "IMDB_WIKI_non_normalized"
python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset --use_PCA --train_experiment_name "IMDB_WIKI_normalized_PCA"
python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset --feature_extraction HOG --train_experiment_name "IMDB_WIKI_normalized_HOG"
python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset --feature_extraction HOG --use_PCA --train_experiment_name "IMDB_WIKI_normalized_HOG_PCA"
