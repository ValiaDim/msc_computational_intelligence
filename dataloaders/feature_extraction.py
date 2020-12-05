from skimage.feature import hog
from skimage.feature import local_binary_pattern
import numpy as np
import cv2 as cv
import joblib
import os
import torch

from models import DEX

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
        for data in test["data_non_processed"]:
            # data = cv.cvtColor(data, cv.COLOR_BGR2RGB)
            # cv.imshow("test", data)
            # cv.waitKey(0)
            gray = utils.rgb2gray(data)/255.0
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block)
            test_feature.append(fd)
            if save_features:
                filename = list(data[2])
                fd_name = str(filename[0], encoding="utf-8") .split('.')[0]+'.feat'
                fd_path = os.path.join('./data/features/test/', fd_name)
                joblib.dump(fd, fd_path)
        print("Test features are extracted and saved.")
        for data in train["data_non_processed"]:
            gray = utils.rgb2gray(data)/255.0
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block)
            train_feature.append(fd)
            if save_features:
                fd = np.concatenate((fd, data[1]))
                filename = list(data[2])
                fd_name = str(filename[0], encoding="utf-8") .split('.')[0]+'.feat'
                fd_path = os.path.join('./data/features/train/', fd_name)
                joblib.dump(fd, fd_path)
        print("Train features are extracted and saved.")
    elif feature_type == "DEX":
        age_model = DEX.Age()
        DEX_weights_path = "data/age_sh.pth"
        age_model.load_state_dict(torch.load(DEX_weights_path))
        age_model.eval()
        print("DEX model is loaded")
        for img in train["data_non_processed"]:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            cv.imshow("test", img)
            cv.waitKey(0)
            img = np.transpose(img, (2, 0, 1))
            img = img[None, :, :, :]
            tensor = torch.from_numpy(img)
            tensor = tensor.type('torch.FloatTensor')
            with torch.no_grad():
                output = age_model(tensor)
            output = output.numpy().squeeze()
    return train_feature, test_feature

