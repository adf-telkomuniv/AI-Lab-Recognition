##
# @license
# Copyright 2019 AI Lab - Telkom University. All Rights Reserved.
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

from statistics import mode
import keras.backend as K
from keras.models import load_model
from PyQt5 import QtCore, QtGui, uic, QtWidgets
from utils.eg_utils import *
import os
import pickle
import numpy as np

__author__ = 'ADF-AI'

form_class = uic.loadUiType("GUI.ui")[0]
running = False

class ImageWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ImageWindow, self).__init__(parent)
        self.image = None

    def set_image(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()


class ApplicationWindow(QtWidgets.QMainWindow,  form_class):
    def __init__(self, hand_model, verbose=0, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.verbose = verbose
        self.font = cv2.FONT_HERSHEY_TRIPLEX

        self.startButton.clicked.connect(self.start_clicked)
        self.stopButton.clicked.connect(self.stop_clicked)

        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = ImageWindow(self.ImgWidget)
		
        self.recognition_target_size = (96, 96)

        self.hand_model = load_model(hand_model, compile=False)

        self.stop_clicked()

    def stop_clicked(self):
        global running
        running = False
        cv2.destroyWindow('temp')
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.startButton.setText('Start Video')
        self.stopButton.setText('Camera off')
        self.update_frame(cv2.imread('./images/logo.jpg'))

    def start_clicked(self):
        global running
        running = True
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.startButton.setText('Starting...')
        self.stopButton.setText('Stop Video')

        # starting video streaming
        cv2.namedWindow('temp')
        cv2.imshow('temp', np.zeros((3, 3, 3), np.uint8))
        video_capture = cv2.VideoCapture(0)
        while running:
            recognition_text = 'Unknown'
            bgr_image = video_capture.read()[1]
            #bgr_image = bgr_image[115:355, 205:435]
            rgb_image = bgr_image
            #rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            #480 640
            rgb_image = rgb_image[160:310, 245:395]
            cv2.imshow("cropped", rgb_image)
            #print(rgb_image.shape)
            rgb_image = np.expand_dims(rgb_image, 0)
            rgb_image = preprocess_input(rgb_image, False)
            outp = self.hand_model.predict(rgb_image)
            num_arg = np.argmax(outp)
            print(num_arg)
            self.label.setText(str(num_arg))
            outp_str = ', '.join(map(str, list(np.around(outp,2))))
            self.label_3.setText(outp_str)
            self.update_frame(bgr_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def closeEvent(self, event):
        global running
        running = False
        cv2.destroyAllWindows()

    def update_frame(self, img):
        img_height, img_width, img_colors = img.shape
        scale_w = float(self.window_width) / float(img_width)
        scale_h = float(self.window_height) / float(img_height)
        scale = min([scale_w, scale_h])

        if scale == 0:
            scale = 1

        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bpl = channel * width
        image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
        self.ImgWidget.set_image(image)


# @author ANDITYA ARIFIANTO
# copyright (c) 2019 - Artificial Intelligence Laboratory, Telkom University #
