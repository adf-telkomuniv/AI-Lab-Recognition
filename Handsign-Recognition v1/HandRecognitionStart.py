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


# copyright (c) 2019 - Artificial Intelligence Laboratory and Computing Laboratory, Telkom University #
