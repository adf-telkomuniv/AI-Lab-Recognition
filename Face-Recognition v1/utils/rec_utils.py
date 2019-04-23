##
# @license
# Copyright 2017 AI Lab - Telkom University. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =============================================================================

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


# @author ANDITYA ARIFIANTO
# copyright (c) 2017 - Artificial Intelligence Laboratory, Telkom University #