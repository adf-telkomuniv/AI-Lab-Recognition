import numpy as np
import cv2
import glob

__author__ = 'ADF-AI'


def img_to_encoding(img, model):
    img = img[..., ::-1]
    img = np.around(np.transpose(img, (2, 0, 1)) / 255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding


def load_database(model, dir_path):
    database = {}
    for img_path in glob.glob(dir_path + '/*.jpg'):
        img = cv2.imread(img_path, 1)
        database[img_path[len(dir_path) + 1:-4]] = img_to_encoding(img, model)
    return database


def recognize(img, database, model):
    encoding = img_to_encoding(img, model)
    min_dist = 100
    identity = 'Unknown'
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding - db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name
    return min_dist, identity

# copyright (c) 2017 - Artificial Intelligence Laboratory, Telkom University #
