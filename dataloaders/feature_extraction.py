from skimage.feature import hog
from skimage.feature import local_binary_pattern
import numpy as np
import joblib
import os
import cv2 as cv

import torch

from dataloaders import utils


# HoG parameters
orientations = 9
pixels_per_cell = [8, 8]
cells_per_block = [2, 2]
visualize = False
normalize = True


def get_features(train, test, save_features=False, feature_type="HOG"):
    train_feature = []
    test_feature = []
    if feature_type == "HOG":
        for data in test["data"]:
            # image = np.reshape(data.T, (32, 32, 3))
            # data = cv.cvtColor(data, cv.COLOR_BGR2RGB)
            # cv.imshow("test", data)
            # cv.waitKey(0)
            gray = utils.rgb2gray(data)/255.0
            fd = local_binary_pattern(gray, orientations, pixels_per_cell, cells_per_block)
            test_feature.append(fd)
            #fd = np.concatenate((fd, data[1]))
            if save_features:
                filename = list(data[2])
                fd_name = str(filename[0], encoding="utf-8") .split('.')[0]+'.feat'
                fd_path = os.path.join('./data/features/test/', fd_name)
                joblib.dump(fd, fd_path)
        print("Test features are extracted and saved.")
        for data in train["data"]:
            # image = np.reshape(data[0].T, (32, 32, 3))
            gray = utils.rgb2gray(data)/255.0
            fd = local_binary_pattern(gray, orientations, pixels_per_cell, cells_per_block)
            train_feature.append(fd)
            if save_features:
                fd = np.concatenate((fd, data[1]))
                filename = list(data[2])
                fd_name = str(filename[0], encoding="utf-8") .split('.')[0]+'.feat'
                fd_path = os.path.join('./data/features/train/', fd_name)
                joblib.dump(fd, fd_path)
        print("Train features are extracted and saved.")
    # elif feature_type == "CNN":
    return train_feature, test_feature

