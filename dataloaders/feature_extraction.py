from skimage.feature import hog
from skimage.feature import local_binary_pattern
import numpy as np
import cv2 as cv
import joblib
import os
import torch

from models import DEX

from dataloaders import utils

DEBUG = True

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
            if DEBUG:
                data_bgr = cv.cvtColor(data, cv.COLOR_BGR2RGB)
                cv.imshow("test", data_bgr)
                cv.waitKey(0)
            gray = utils.rgb2gray(data)/255.0
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block)
            test_feature.append(fd)
            if save_features:
                filename = list(data[2])
                fd_name = str(filename[0], encoding="utf-8") .split('.')[0]+'.feat'
                fd_path = os.path.join('./data/features/test/', fd_name)
                joblib.dump(fd, fd_path)
        print("Test features are extracted.")
        for data in train["data"]:
            gray = utils.rgb2gray(data)/255.0
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block)
            train_feature.append(fd)
            if save_features:
                fd = np.concatenate((fd, data[1]))
                filename = list(data[2])
                fd_name = str(filename[0], encoding="utf-8") .split('.')[0]+'.feat'
                fd_path = os.path.join('./data/features/train/', fd_name)
                joblib.dump(fd, fd_path)
        print("Train features are extracted.")
    elif feature_type == "DEX":
        age_model = DEX.Age()
        DEX_weights_path = "data/age_sd.pth"
        age_model.load_state_dict(torch.load(DEX_weights_path))
        age_model.eval()
        print("DEX model is loaded")
        for img in train["data"]:
            if DEBUG:
                img_bgr = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                cv.imshow("test", img_bgr)
                cv.waitKey(0)
            img = cv.resize(img, (224, 224))
            img = np.transpose(img, (2, 0, 1))
            img = img[None, :, :, :]
            tensor = torch.from_numpy(img)
            tensor = tensor.type('torch.FloatTensor')
            with torch.no_grad():
                output = age_model(tensor)
            output = output.numpy().squeeze()
    return train_feature, test_feature

