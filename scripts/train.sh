# CIFAR_10 dataset:
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset 10000 --normalize_data --train_experiment_name "CIFAR10_normalized"
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset 10000 --train_experiment_name "CIFAR10_non_normalized"
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset 10000 --normalize_data --use_PCA --train_experiment_name "CIFAR10_normalized_PCA"
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset 10000 --normalize_data --feature_extraction HOG --train_experiment_name "CIFAR10_normalized_HOG"
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset 10000 --normalize_data --feature_extraction HOG --use_PCA --train_experiment_name "CIFAR10_normalized_HOG_PCA"

# kpca
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset 10000 --normalize_data --train_experiment_name "CIFAR10_normalized_kpca_lda_gridGamma_Components" --classifier_type kpca_lda
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset 10000 --normalize_data --feature_extraction HOG --train_experiment_name "CIFAR10_HOG_grid_components_kpca_lda" --classifier_type kpca_lda
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset 10000 --normalize_data --feature_extraction MobileNetV2 --feature_layer MobileNet_layer3 --train_experiment_name "CIFAR10_normalized_grid_components_MobileNetV2_layer3__kpca_lda" --classifier_type kpca_lda
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset 10000 --normalize_data --feature_extraction MobileNetV2 --feature_layer MobileNet_layer4 --train_experiment_name "CIFAR10_normalized_grid_components_MobileNetV2_layer4__kpca_lda" --classifier_type kpca_lda
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset 10000 --normalize_data --feature_extraction MobileNetV2 --feature_layer MobileNet_layer5 --train_experiment_name "CIFAR10_normalized_grid_components_MobileNetV2_layer5__kpca_lda" --classifier_type kpca_lda
#search kpca
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset 10000 --normalize_data --train_experiment_name "CIFAR10_normalized_kpca_lda_searchKPCA" --classifier_type kpca_lda

# nearest neighbor
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset 10000 --normalize_data --train_experiment_name "CIFAR10_normalized_nearestNeighbor" --classifier_type nearest_neighbor
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset 10000 --normalize_data --train_experiment_name "CIFAR10_normalized_nearestCentroid" --classifier_type nearest_centroid

# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset 10000 --normalize_data --train_experiment_name "CIFAR10_normalized_lda_pca" --classifier_type lda --


# imdb_wiki dataset:
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset 20000 --normalize_data --train_experiment_name "IMDB_WIKI_normalized"
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset 20000 --train_experiment_name "IMDB_WIKI_non_normalized"
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset 20000 --normalize_data --use_PCA --train_experiment_name "IMDB_WIKI_normalized_PCA"
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset 20000 --normalize_data --feature_extraction HOG --train_experiment_name "IMDB_WIKI_normalized_HOG"
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset 20000 --normalize_data --feature_extraction HOG --use_PCA --train_experiment_name "IMDB_WIKI_normalized_HOG_PCA"


# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset 20000 --normalize_data --feature_extraction DEX --feature_layer fc1 --train_experiment_name "IMDB_WIKI_DEX_fc1"
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset 20000 --normalize_data --feature_extraction DEX --feature_layer fc1 --use_PCA --train_experiment_name "IMDB_WIKI_DEX_fc1_PCA"
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset 20000 --normalize_data --feature_extraction DEX --feature_layer fc2 --train_experiment_name "IMDB_WIKI_DEX_fc2"
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset 20000 --normalize_data --feature_extraction DEX --feature_layer fc2 --use_PCA --train_experiment_name "IMDB_WIKI_DEX_fc2_PCA"


# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset 10000 --normalize_data --feature_extraction MobileNetV2 --feature_layer MobileNet_layer3 --train_experiment_name "CIFAR10_normalized_MobileNetV2_layer3"
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset 10000 --normalize_data --feature_extraction MobileNetV2 --feature_layer MobileNet_layer4 --train_experiment_name "CIFAR10_normalized_MobileNetV2_layer4"
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset 10000 --normalize_data --feature_extraction MobileNetV2 --feature_layer MobileNet_layer5 --train_experiment_name "CIFAR10_normalized_MobileNetV2_layer5"
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --grid_search --reduced_training_dataset 10000 --normalize_data --feature_extraction MobileNetV2 --feature_layer MobileNet_layer6 --train_experiment_name "CIFAR10_normalized_MobileNetV2_layer6"

# kpca
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset 20000 --normalize_data  --classifier_type kpca_lda --train_experiment_name "IMDB_WIKI_normalized_kpca_lda_gridSolver"
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset 20000 --normalize_data --classifier_type kpca_lda --feature_extraction HOG --train_experiment_name "IMDB_WIKI_normalized_HOG_kpca_lda_gridComponents"
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset 20000 --normalize_data --classifier_type kpca_lda --feature_extraction DEX --feature_layer fc1 --train_experiment_name "IMDB_WIKI_DEX_fc1_kpca_lda_gridComponents"
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset 20000 --normalize_data --classifier_type kpca_lda --feature_extraction DEX --feature_layer fc2 --train_experiment_name "IMDB_WIKI_DEX_fc2_kpca_lda_gridComponents"

# pca lda
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset 20000 --normalize_data --train_experiment_name "IMDB_WIKI_normalized_pca_lda" --classifier_type lda --pca_type PCA --bin_ages
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset 20000 --normalize_data --train_experiment_name "IMDB_WIKI_normalized_lda" --classifier_type lda --bin_ages

# nearest neighbor
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset 20000 --normalize_data  --classifier_type nearest_neighbor --train_experiment_name "IMDB_WIKI_normalized_nearestNeighbor" --bin_ages
# python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc" --dataset "IMDB_WIKI" --grid_search --reduced_training_dataset 20000 --normalize_data  --classifier_type nearest_centroid --train_experiment_name "IMDB_WIKI_normalized_nearestCentroid" --bin_ages



# MNIST spectral
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_Isomap6_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction Isomap --number_of_neighbors 6
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_Isomap7_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction Isomap --number_of_neighbors 7
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_Isomap8_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction Isomap --number_of_neighbors 8
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_Isomap9_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction Isomap --number_of_neighbors 9
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_Isomap10_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction Isomap --number_of_neighbors 10
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_Isomap11_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction Isomap --number_of_neighbors 11
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_Isomap12_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction Isomap --number_of_neighbors 12


#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_LLE6_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction LLE --number_of_neighbors 6
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_LLE7_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction LLE --number_of_neighbors 7
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_LLE8_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction LLE --number_of_neighbors 8
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_LLE9_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction LLE --number_of_neighbors 9
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_LLE10_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction LLE --number_of_neighbors 10
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_LLE11_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction LLE --number_of_neighbors 11
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_LLE12_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction LLE --number_of_neighbors 12


#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_modifiedLLE6_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction modified_LLE --number_of_neighbors 6
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_modifiedLLE7_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction modified_LLE --number_of_neighbors 7
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_modifiedLLE8_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction modified_LLE --number_of_neighbors 8
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_modifiedLLE9_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction modified_LLE --number_of_neighbors 9
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_modifiedLLE10_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction modified_LLE --number_of_neighbors 10
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_modifiedLLE11_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction modified_LLE --number_of_neighbors 11
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_modifiedLLE12_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction modified_LLE --number_of_neighbors 12


#python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_laplacian_eigenmaps_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction laplacian_eigenmaps
#python3 train.py  --dataset "MNIST" --reduced_training_dataset 1000 --normalize_data --train_experiment_name "MNIST_TSNE_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction TSNE


# CIFAR10 spectral
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_Isomap6_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction Isomap --number_of_neighbors 6
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_Isomap7_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction Isomap --number_of_neighbors 7
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_Isomap8_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction Isomap --number_of_neighbors 8
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_Isomap9_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction Isomap --number_of_neighbors 9
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_Isomap10_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction Isomap --number_of_neighbors 10
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_Isomap11_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction Isomap --number_of_neighbors 11
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_Isomap12_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction Isomap --number_of_neighbors 12


#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_LLE6_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction LLE --number_of_neighbors 6
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_LLE7_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction LLE --number_of_neighbors 7
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_LLE8_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction LLE --number_of_neighbors 8
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_LLE9_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction LLE --number_of_neighbors 9
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_LLE10_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction LLE --number_of_neighbors 10
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_LLE11_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction LLE --number_of_neighbors 11
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_LLE12_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction LLE --number_of_neighbors 12


#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_modifiedLLE6_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction modified_LLE --number_of_neighbors 6
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_modifiedLLE7_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction modified_LLE --number_of_neighbors 7
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_modifiedLLE8_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction modified_LLE --number_of_neighbors 8
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_modifiedLLE9_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction modified_LLE --number_of_neighbors 9
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_modifiedLLE10_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction modified_LLE --number_of_neighbors 10
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_modifiedLLE11_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction modified_LLE --number_of_neighbors 11
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_modifiedLLE12_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction modified_LLE --number_of_neighbors 12


#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "CIFAR10_laplacian_eigenmaps_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction laplacian_eigenmaps
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 1000 --normalize_data --train_experiment_name "CIFAR10_TSNE_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction TSNE


#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 1000 --normalize_data --feature_extraction MobileNetV2 --feature_layer MobileNet_layer3 --train_experiment_name "CIFAR10_MobilenetV2Layer3_TSNE_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction TSNE
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 1000 --normalize_data --feature_extraction MobileNetV2 --feature_layer MobileNet_layer4 --train_experiment_name "CIFAR10_MobilenetV2Layer4_TSNE_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction TSNE
#python3 train.py --train_path "/media/valia/TOSHIBA EXT/datasets/msc/cifar-10-python/cifar-10-batches-py" --dataset "CIFAR10" --reduced_training_dataset 1000 --normalize_data --feature_extraction MobileNetV2 --feature_layer MobileNet_layer5 --train_experiment_name "CIFAR10_MobilenetV2Layer5_TSNE_kmeans_spectral_5clusters" --classifier_type kmeans_spectral --dimentionality_reduction TSNE

# Final delete
python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_Isomap6_kmeans_spectral_10clusters" --classifier_type kmeans_spectral --dimentionality_reduction Isomap --number_of_neighbors 6


python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_LLE6_kmeans_spectral_10clusters" --classifier_type kmeans_spectral --dimentionality_reduction LLE --number_of_neighbors 6


python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_modifiedLLE6_kmeans_spectral_10clusters" --classifier_type kmeans_spectral --dimentionality_reduction modified_LLE --number_of_neighbors 6


python3 train.py  --dataset "MNIST" --reduced_training_dataset 2000 --normalize_data --train_experiment_name "MNIST_laplacian_eigenmaps_kmeans_spectral_10clusters" --classifier_type kmeans_spectral --dimentionality_reduction laplacian_eigenmaps
python3 train.py  --dataset "MNIST" --reduced_training_dataset 1000 --normalize_data --train_experiment_name "MNIST_TSNE_kmeans_spectral_10clusters" --classifier_type kmeans_spectral --dimentionality_reduction TSNE
