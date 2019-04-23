##
# @license
# Copyright 2018 AI Lab - Telkom University. All Rights Reserved.
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

from utils.ApplicationWindow import *
import sys

# ---------------------------------------------------------
face_model = './models/haarcascade_frontalface_default.xml'
emotion_model = './models/emotion_1.hdf5'
gender_model = './models/gender_1.hdf5'
recognition_model = './models/recognition_2.h5'
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


# @author ANDITYA ARIFIANTO
# copyright (c) 2018 - Artificial Intelligence Laboratory, Telkom University #