from utils.ApplicationWindow import *
import sys

# ---------------------------------------------------------
face_model = './models/haarcascade_frontalface_default.xml'
emotion_model = './models/emotion_1.hdf5'
gender_model = './models/gender_1.hdf5'
recognition_model = './models/recognition_1.h5'
spoof_model = './models/spoof/modelCASIANB.pkl'
gender_labels = {0: 'woman', 1: 'man'}
verbose = 0
# ---------------------------------------------------------

app = QtWidgets.QApplication(sys.argv)
app.setWindowIcon(QtGui.QIcon('./images/badge.png'))
w = ApplicationWindow(face_model, emotion_model, gender_model, recognition_model,
                      spoof_model, gender_labels=gender_labels, verbose=verbose)
w.setWindowTitle('Face Emotion and Gender Recognition')
w.show()
app.exec_()


# copyright (c) 2017 - Artificial Intelligence Laboratory, Telkom University #
