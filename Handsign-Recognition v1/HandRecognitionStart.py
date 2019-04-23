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

from utils.ApplicationWindow import *
import sys

# ---------------------------------------------------------
#hand_model = './models/hand_recog.h5'
hand_model = './models/best.hdf5'
verbose = 1
# ---------------------------------------------------------

app = QtWidgets.QApplication(sys.argv)
app.setWindowIcon(QtGui.QIcon('./images/badge.png'))
w = ApplicationWindow(hand_model, verbose=verbose)
w.setWindowTitle('Handsign Number Recognition')
w.show()
app.exec_()

# @author ANDITYA ARIFIANTO
# copyright (c) 2019 - Artificial Intelligence Laboratory, Telkom University #