import pandas as pd
import numpy as np
import cv2 as cv
from tqdm import tqdm
import os

DEBUG = False


class imdb_wiki_dataloader():
    def __init__(self, i_data_dir, i_percentage_validation, i_normalize=True, i_reduced_training_dataset=False,
                 i_raw_images=False, i_feature_type="off"):
        self.imdb_wiki_train = {}
        self.imdb_wiki_test = {}
        self.imdb_wiki_validation = {}
        self.data_dir = i_data_dir
        self.data_list = "dataloaders/wiki_imdb.csv"
        self.percentage_validation = i_percentage_validation
        self.percentage_test = 0.2
        self.normalize = i_normalize
        self.raw_images = i_raw_images
        self.feature_type = i_feature_type
        self.reduced_training_dataset = i_reduced_training_dataset
        self.face_cascade = cv.CascadeClassifier('dataloaders/haarcascade_frontalface_default.xml')

    def preprocess_data(self):
        self.imdb_wiki_train['data'] = self.imdb_wiki_train['data'].astype(np.float)
        self.imdb_wiki_test['data'] = self.imdb_wiki_test['data'].astype(np.float)
        self.imdb_wiki_validation['data'] = self.imdb_wiki_validation['data'].astype(np.float)
        print(self.imdb_wiki_train['data'].shape)
        print(self.imdb_wiki_test['data'].shape)
        print(self.imdb_wiki_validation['data'].shape)

        self.imdb_wiki_train['data'] = np.reshape(self.imdb_wiki_train['data'], (self.imdb_wiki_train['data'].shape[0], -1))
        self.imdb_wiki_test['data'] = np.reshape(self.imdb_wiki_test['data'], (self.imdb_wiki_test['data'].shape[0], -1))
        self.imdb_wiki_validation['data'] = np.reshape(self.imdb_wiki_validation['data'], (self.imdb_wiki_validation['data'].shape[0], -1))
        print(self.imdb_wiki_train['data'].shape)
        print(self.imdb_wiki_train['data'][0])

        # Normalize
        if self.normalize:
            self.imdb_wiki_train['data'] = (self.imdb_wiki_train['data'] / 255)
            self.imdb_wiki_test['data'] = (self.imdb_wiki_test['data'] / 255)
            self.imdb_wiki_validation['data'] = (self.imdb_wiki_validation['data'] / 255)

        print(self.imdb_wiki_train['data'].shape)
        print(self.imdb_wiki_train['data'][0])

    def create_validation_test_set(self):
        train_before_length = len(self.imdb_wiki_train['filenames'])
        validation_length = int(self.percentage_validation * train_before_length)
        test_length = int(self.percentage_test * train_before_length)
        training_length = train_before_length - validation_length - test_length
        # validation_length = 10000
        self.imdb_wiki_validation['data'] = self.imdb_wiki_train['data'][:validation_length, :, :, :]
        self.imdb_wiki_validation['filenames'] = self.imdb_wiki_train['filenames'][:validation_length]
        self.imdb_wiki_validation['labels'] = self.imdb_wiki_train['labels'][:validation_length]

        self.imdb_wiki_test['data'] = self.imdb_wiki_train['data'][validation_length:validation_length + test_length, :, :, :]
        self.imdb_wiki_test['filenames'] = self.imdb_wiki_train['filenames'][validation_length:validation_length + test_length]
        self.imdb_wiki_test['labels'] = self.imdb_wiki_train['labels'][validation_length:validation_length + test_length]

        self.imdb_wiki_train['data'] = self.imdb_wiki_train['data'][validation_length + test_length:, :, :, :]
        self.imdb_wiki_train['filenames'] = self.imdb_wiki_train['filenames'][validation_length + test_length:]
        self.imdb_wiki_train['labels'] = self.imdb_wiki_train['labels'][validation_length + test_length:]

    def get_imdb_wiki(self):
        data = pd.read_csv(self.data_list)
        images = []
        ages = []
        filenames = []
        iter = 0
        if self.reduced_training_dataset is not None:
            dataset_length = self.reduced_training_dataset
        else:
            dataset_length = len(data["path"])
            print(dataset_length)
        pbar = tqdm(total=dataset_length, desc='Cropping Face Images')
        for path in data["path"]:
            image_path = os.path.join(self.data_dir, path)
            image = cv.imread(os.path.join(self.data_dir, path))
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            faces_rect = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces_rect) == 1:
                if self.feature_type != "DEX":
                    [x, y, w, h] = faces_rect[0]
                    cropped_face = image[x:x+w, y:y+h]
                    if DEBUG:
                        cv.imshow('cropped_face', cropped_face)
                        cv.waitKey()
                    cropped_face = cv.resize(cropped_face, (50, 50))
                    images.append(cv.cvtColor(cropped_face, cv.COLOR_BGR2RGB))
                else:
                    image = cv.resize(image, (224, 224))
                    images.append(cv.cvtColor(image, cv.COLOR_BGR2RGB))
                ages.append(data["age"][iter])
                filenames.append(image_path)
                if self.reduced_training_dataset is not None:
                    pbar.update(1)
            if DEBUG:
                for (x, y, w, h) in faces_rect:
                    cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Display the output
                cv.imshow('img', image)
                cv.waitKey()
            if self.reduced_training_dataset is None:
                pbar.update(1)
            iter += 1
            if len(ages) == dataset_length:
                break
        self.imdb_wiki_train["data"] = np.stack(images, axis=0)
        self.imdb_wiki_train["labels"] = ages
        self.imdb_wiki_train["filenames"] = filenames
        if self.percentage_validation:
            assert self.percentage_validation <= 1, "Percentage validation needs to be less than 1"
            self.create_validation_test_set()
        if not self.raw_images:
            self.preprocess_data()
        return self.imdb_wiki_train, self.imdb_wiki_test, self.imdb_wiki_validation







