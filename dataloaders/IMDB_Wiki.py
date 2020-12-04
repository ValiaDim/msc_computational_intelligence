import pandas as pd
import numpy as np
import cv2 as cv
import os


class imdb_wiki_dataloader():
    def __init__(self, i_data_list, i_percentage_validation, i_preprocess=True, i_reduced_training_dataset=False):
        self.imdb_wiki_train = {}
        self.imdb_wiki_test = {}
        self.imdb_wiki_validation = {}
        self.data_list = i_data_list
        self.percentage_validation = i_percentage_validation
        self.percentage_test = 0.2
        self.preprocess = i_preprocess
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
        self.imdb_wiki_train['data'] = ((self.imdb_wiki_train['data'] / 255) * 2) - 1  # better to 0..1 think of it
        self.imdb_wiki_test['data'] = ((self.imdb_wiki_test['data'] / 255) * 2) - 1  # better to 0..1 think of it
        self.imdb_wiki_validation['data'] = ((self.imdb_wiki_validation['data'] / 255) * 2) - 1  # better to 0..1 think of it

        print(self.imdb_wiki_train['data'].shape)
        print(self.imdb_wiki_train['data'][0])

    def create_validation_test_set(self):
        train_before_length = len(self.imdb_wiki_train['filenames'])
        validation_length = int(self.percentage_validation * train_before_length)
        test_length = int(self.percentage_test * train_before_length)
        training_length = train_before_length - validation_length - test_length
        # validation_length = 10000
        self.imdb_wiki_validation['data'] = self.imdb_wiki_train['data'][:validation_length, :, :]
        self.imdb_wiki_validation['filenames'] = self.imdb_wiki_train['filenames'][:validation_length]
        self.imdb_wiki_validation['labels'] = self.imdb_wiki_train['labels'][:validation_length]

        self.imdb_wiki_test['data'] = self.imdb_wiki_train['data'][validation_length:validation_length + test_length, :, :]
        self.imdb_wiki_test['filenames'] = self.imdb_wiki_train['filenames'][validation_length:validation_length + test_length]
        self.imdb_wiki_test['labels'] = self.imdb_wiki_train['labels'][validation_length:validation_length + test_length]

        self.imdb_wiki_train['data'] = self.imdb_wiki_train['data'][validation_length + test_length:, :, :]
        self.imdb_wiki_train['filenames'] = self.imdb_wiki_train['filenames'][validation_length + test_length:]
        self.imdb_wiki_train['labels'] = self.imdb_wiki_train['labels'][validation_length + test_length:]

    def get_imdb_wiki(self):
        root_path = "/media/valia/TOSHIBA EXT/datasets/msc"
        data = pd.read_csv("dataloaders/wiki_imdb.csv")
        images = []
        ages = []
        filenames = []
        iter = 0

        for path in data["path"]:
            image_path = os.path.join(root_path, path)
            image = cv.imread(os.path.join(root_path, path))
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            faces_rect = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces_rect) == 1:
                [x, y, w, h] = faces_rect[0]
                cropped_face = gray[x:x+w, y:y+h]
                # cv.imshow('cropped_face', cropped_face)
                # cv.waitKey()
                cropped_face = cv.resize(cropped_face, (50, 50))
                images.append(cropped_face)
                ages.append(data["age"][iter])
                filenames.append(image_path)
            # for (x, y, w, h) in faces_rect:
            #     cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Display the output
            # cv.imshow('img', image)
            # cv.waitKey()
            iter += 1
            if len(ages) == 1000:
                break
        self.imdb_wiki_train["data"] = np.stack(images, axis=0 )
        self.imdb_wiki_train["labels"] = ages
        self.imdb_wiki_train["filenames"] = filenames
        self.create_validation_test_set()
        self.preprocess_data()
        return self.imdb_wiki_train, self.imdb_wiki_train, self.imdb_wiki_validation