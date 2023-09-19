from PyQt5 import QtCore, QtGui, QtWidgets
from Main_Colours import Colours 
from Aritmath1 import Ui_Aritmath



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(654, 364)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 40, 571, 111))
        font = QtGui.QFont()
        font.setFamily("MotterFemD")
        font.setPointSize(24)
        font.setItalic(True)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(110, 150, 421, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(110, 170, 421, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(110, 190, 421, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setEnabled(True)
        self.pushButton.setGeometry(QtCore.QRect(280, 270, 75, 23))
        self.pushButton.setAutoDefault(False)
        self.pushButton.setDefault(True)
        self.pushButton.setFlat(False)
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 654, 21))
        self.menubar.setObjectName("menubar")
        self.menuNext = QtWidgets.QMenu(self.menubar)
        self.menuNext.setObjectName("menuNext")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionColours = QtWidgets.QAction(MainWindow)
        self.actionColours.setObjectName("actionColours")
        self.actionAritmatic_Operations = QtWidgets.QAction(MainWindow)
        self.actionAritmatic_Operations.setObjectName("actionAritmatic_Operations")
        self.actionAritmatic_Operations.triggered.connect(self.Aritmath1)
        self.menuNext.addAction(self.actionColours)
        self.menuNext.addAction(self.actionAritmatic_Operations)
        self.menubar.addAction(self.menuNext.menuAction())

        self.retranslateUi(MainWindow)
        
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.Colours)
        
    def Colours(self):
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Colours()
        self.ui.setupUi(self.MainWindow)
        self.MainWindow.show()
        
    def Aritmath1(self):
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_Aritmath()
        self.ui.setupUi(self.MainWindow)
        self.MainWindow.show()
    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Processing Colours"))
        self.label.setText(_translate("MainWindow", "Image Processing Colours"))
        self.label.setText(_translate("MainWindow", 
                                      "<font color='#FF0000'>I</font><font color='#00FF00'>m</font><font color='#0000FF'>a</font><font color='#FF0000'>g</font><font color='#00FF00'>e</font><br>"
                                      "<font color='#0000FF'>P</font><font color='#FF0000'>r</font><font color='#00FF00'>o</font><font color='#0000FF'>c</font><font color='#FF0000'>e</font><font color='#00FF00'>s</font><font color='#0000FF'>s</font><br>"
                                      "<font color='#FF0000'>C</font><font color='#00FF00'>o</font><font color='#0000FF'>l</font><font color='#FF0000'>o</font><font color='#00FF00'>u</font><font color='#0000FF'>r</font><font color='#FF0000'>s</font>"))
        self.label_2.setText(_translate("MainWindow", "By:"))
        self.label_3.setText(_translate("MainWindow", "Ahmad Firdaus Tarmdzi"))
        self.label_4.setText(_translate("MainWindow", "E41211991"))
        self.pushButton.setText(_translate("MainWindow", "Lanjut"))
        self.menuNext.setTitle(_translate("MainWindow", "Next"))
        self.actionColours.setText(_translate("MainWindow", "Colours"))
        self.actionAritmatic_Operations.setText(_translate("MainWindow", "Aritmatic Operations"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
