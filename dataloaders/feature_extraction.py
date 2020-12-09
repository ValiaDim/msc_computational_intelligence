from skimage.feature import hog
from skimage.feature import local_binary_pattern
import numpy as np
import cv2 as cv
import joblib
import os
import torch
from tqdm import tqdm
from models import DEX
from models.VGG19 import VGG19_batchnorm
from models.MobileNetV2 import MobileNetV2

from dataloaders import utils

DEBUG = False

# HoG parameters
orientations = 9
pixels_per_cell = [8, 8]
cells_per_block = [2, 2]
visualize = False
normalize = True


def get_features(train, test, save_features=False, feature_type="HOG", feature_layer=None):
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
        age_model = DEX.Age(which_features=feature_layer)
        DEX_weights_path = "data/age_sd.pth"
        age_model.load_state_dict(torch.load(DEX_weights_path))
        age_model.eval()
        print("DEX model is loaded")
        progress_bar = tqdm(total=len(train["data"]), desc='Extracting train features')
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
            train_feature.append(output)
            progress_bar.update(1)
        print("Train features are extracted.")
        progress_bar = tqdm(total=len(test["data"]), desc='Extracting test features')
        for img in test["data"]:
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
            test_feature.append(output)
            progress_bar.update(1)
        print("Train features are extracted.")
    elif feature_type=="VGG19":
        vgg19_model = VGG19_batchnorm(which_features=feature_layer)
        print("VGG model is loaded")
        progress_bar = tqdm(total=len(train["data"]), desc='Extracting train features')
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
                output = vgg19_model(tensor)
            output = output.numpy().squeeze()
            train_feature.append(output)
            progress_bar.update(1)
        print("Train features are extracted.")
        progress_bar = tqdm(total=len(test["data"]), desc='Extracting test features')
        for img in test["data"]:
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
                output = vgg19_model(tensor)
            output = output.numpy().squeeze()
            test_feature.append(output)
            progress_bar.update(1)
        print("Train features are extracted.")
    elif feature_type == "MobileNetV2":
        mobilenetV2_model = MobileNetV2(which_features=feature_layer)
        print("MobileNetV2 model is loaded")
        progress_bar = tqdm(total=len(train["data"]), desc='Extracting train features')
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
                output = mobilenetV2_model(tensor)
            output = output.numpy().squeeze()
            train_feature.append(output)
            progress_bar.update(1)
        print("Train features are extracted.")
        progress_bar = tqdm(total=len(test["data"]), desc='Extracting test features')
        for img in test["data"]:
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
                output = mobilenetV2_model(tensor)
            output = output.numpy().squeeze()
            test_feature.append(output)
            progress_bar.update(1)
        print("Train features are extracted.")
    return train_feature, test_feature

