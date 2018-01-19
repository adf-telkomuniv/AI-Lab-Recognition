import cv2
import numpy as np
from statistics import mode

from keras.models import load_model
from PyQt5 import QtCore, QtGui, uic, QtWidgets
from utils.eg_utils import *

form_class = uic.loadUiType("GUI.ui")[0]
running = False


class ImageWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ImageWindow, self).__init__(parent)
        self.image = None

    def setImage(self, image):
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


class ApplicationWindow(QtWidgets.QMainWindow, form_class):
    def __init__(self, face_model, emotion_model, gender_model, parent=None, gender_labels=None, emotion_labels=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.startButton.clicked.connect(self.start_clicked)
        self.stopButton.clicked.connect(self.stop_clicked)

        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = ImageWindow(self.ImgWidget)

        # self.gender_labels = {1:'woman', 0:'man'}
        if gender_labels is None:
            self.gender_labels = {0: 'woman', 1: 'man'}
        else:
            self.gender_labels = gender_labels

        if emotion_labels is None:
            self.emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                                   4: 'sad', 5: 'surprise', 6: 'neutral'}
        else:
            self.emotion_labels = emotion_labels
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # hyper-parameters for bounding boxes shape
        self.frame_window = 10
        self.emotion_offsets = (20, 40)
        self.gender_offsets = (30, 60)

        # loading models
        self.face_detection = cv2.CascadeClassifier(face_model)
        self.emotion_classifier = load_model(emotion_model, compile=False)
        self.gender_classifier = load_model(gender_model, compile=False)

        # getting input model shapes for inference
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]
        self.gender_target_size = self.gender_classifier.input_shape[1:3]

        # starting lists for calculating modes
        self.emotion_window = []
        self.gender_window = []

        self.stop_clicked()

    def stop_clicked(self):
        global running
        running = False
        cv2.destroyWindow('window_frame')
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.startButton.setText('Start Video')
        self.stopButton.setText('Camera off')
        self.update_frame(cv2.imread('./images/logo.jpg'))

    def start_clicked(self):
        print('test')
        global running
        running = True
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.startButton.setText('Starting...')
        self.stopButton.setText('Stop Video')

        # starting video streaming
        cv2.namedWindow('temp')
        video_capture = cv2.VideoCapture(0)
        while running:
            bgr_image = video_capture.read()[1]
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            faces = self.face_detection.detectMultiScale(gray_image, 1.3, 5)

            for face_coordinates in faces:
                x1, x2, y1, y2 = apply_offsets(face_coordinates, self.gender_offsets)
                rgb_face = rgb_image[y1:y2, x1:x2]
                x1, x2, y1, y2 = apply_offsets(face_coordinates, self.emotion_offsets)
                gray_face = gray_image[y1:y2, x1:x2]
                try:
                    rgb_face = cv2.resize(rgb_face, self.gender_target_size)
                    gray_face = cv2.resize(gray_face, self.emotion_target_size)
                except:
                    continue

                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_prediction = self.emotion_classifier.predict(gray_face)
                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = self.emotion_labels[emotion_label_arg]
                self.emotion_window.append(emotion_text)

                rgb_face = np.expand_dims(rgb_face, 0)
                rgb_face = preprocess_input(rgb_face, False)
                gender_label_arg = np.argmax(self.gender_classifier.predict(rgb_face))
                gender_text = self.gender_labels[gender_label_arg]
                self.gender_window.append(gender_text)

                if len(self.emotion_window) > self.frame_window:
                    self.emotion_window.pop(0)
                    self.gender_window.pop(0)
                try:
                    emotion_mode = mode(self.emotion_window)
                    gender_mode = mode(self.gender_window)
                except:
                    continue

                if gender_text == self.gender_labels[0]:
                    color_g = (0, 0, 255)
                else:
                    color_g = (255, 0, 0)

                if emotion_text == 'angry':
                    color = emotion_probability * np.asarray((255, 0, 0))
                elif emotion_text == 'sad':
                    color = emotion_probability * np.asarray((0, 0, 255))
                elif emotion_text == 'happy':
                    color = emotion_probability * np.asarray((255, 255, 0))
                elif emotion_text == 'surprise':
                    color = emotion_probability * np.asarray((0, 255, 255))
                else:
                    color = emotion_probability * np.asarray((0, 255, 0))

                color = color.astype(int)
                color = color.tolist()

                draw_bounding_box(face_coordinates, rgb_image, color)
                draw_text(face_coordinates, rgb_image, emotion_mode,
                          color, 0, -45, 1, 1)
                draw_text(face_coordinates, rgb_image, gender_mode,
                          color_g, 0, -20, 1, 1)

            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
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
        height, width, bpc = img.shape
        bpl = bpc * width
        image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
        self.ImgWidget.setImage(image)
