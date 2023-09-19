from PyQt5.QtGui import QImage, QPixmap, QColor, QImageReader, QImageWriter
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QAction, QFileDialog, QMessageBox
from io import BytesIO


class Ui_Geometri(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(656, 365)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 281, 261))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(350, 10, 281, 261))
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setText("")
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 656, 21))
        self.menubar.setObjectName("menubar")
        self.menuTentang = QtWidgets.QMenu(self.menubar)
        self.menuTentang.setObjectName("menuTentang")
        self.menuScaling = QtWidgets.QMenu(self.menuTentang)
        self.menuScaling.setObjectName("menuScaling")
        self.menuFliping = QtWidgets.QMenu(self.menuTentang)
        self.menuFliping.setObjectName("menuFliping")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionBuka = QtWidgets.QAction(MainWindow)
        self.actionBuka.setObjectName("actionBuka")
        self.actionSimpan_Sebagai = QtWidgets.QAction(MainWindow)
        self.actionSimpan_Sebagai.setObjectName("actionSimpan_Sebagai")
        self.actionKeluar = QtWidgets.QAction(MainWindow)
        self.actionKeluar.setObjectName("actionKeluar")
        self.actionInput = QtWidgets.QAction(MainWindow)
        self.actionInput.setObjectName("actionInput")
        self.actionOutput = QtWidgets.QAction(MainWindow)
        self.actionOutput.setObjectName("actionOutput")
        self.actionInput_Output = QtWidgets.QAction(MainWindow)
        self.actionInput_Output.setObjectName("actionInput_Output")
        self.actionBrightness_Contras = QtWidgets.QAction(MainWindow)
        self.actionBrightness_Contras.setObjectName("actionBrightness_Contras")
        self.actionInvers = QtWidgets.QAction(MainWindow)
        self.actionInvers.setObjectName("actionInvers")
        self.actionLog_Brightness = QtWidgets.QAction(MainWindow)
        self.actionLog_Brightness.setObjectName("actionLog_Brightness")
        self.actionGamma_Corrections = QtWidgets.QAction(MainWindow)
        self.actionGamma_Corrections.setObjectName("actionGamma_Corrections")
        self.actionKuning = QtWidgets.QAction(MainWindow)
        self.actionKuning.setObjectName("actionKuning")
        self.actionCyan = QtWidgets.QAction(MainWindow)
        self.actionCyan.setObjectName("actionCyan")
        self.actionUngu = QtWidgets.QAction(MainWindow)
        self.actionUngu.setObjectName("actionUngu")
        self.actionAbu_abu = QtWidgets.QAction(MainWindow)
        self.actionAbu_abu.setObjectName("actionAbu_abu")
        self.actionAvverage = QtWidgets.QAction(MainWindow)
        self.actionAvverage.setObjectName("actionAvverage")
        self.actionLightness = QtWidgets.QAction(MainWindow)
        self.actionLightness.setObjectName("actionLightness")
        self.actionLuminance = QtWidgets.QAction(MainWindow)
        self.actionLuminance.setObjectName("actionLuminance")
        self.actionContrast = QtWidgets.QAction(MainWindow)
        self.actionContrast.setObjectName("actionContrast")
        self.action1_Bit = QtWidgets.QAction(MainWindow)
        self.action1_Bit.setObjectName("action1_Bit")
        self.action2_BIt = QtWidgets.QAction(MainWindow)
        self.action2_BIt.setObjectName("action2_BIt")
        self.action3_bit = QtWidgets.QAction(MainWindow)
        self.action3_bit.setObjectName("action3_bit")
        self.action4_bit = QtWidgets.QAction(MainWindow)
        self.action4_bit.setObjectName("action4_bit")
        self.action5_bit = QtWidgets.QAction(MainWindow)
        self.action5_bit.setObjectName("action5_bit")
        self.action6_bit = QtWidgets.QAction(MainWindow)
        self.action6_bit.setObjectName("action6_bit")
        self.action7_bit = QtWidgets.QAction(MainWindow)
        self.action7_bit.setObjectName("action7_bit")
        self.actionHistogram_Equalization = QtWidgets.QAction(MainWindow)
        self.actionHistogram_Equalization.setObjectName("actionHistogram_Equalization")
        self.actionFuzzy_HE_RGB = QtWidgets.QAction(MainWindow)
        self.actionFuzzy_HE_RGB.setObjectName("actionFuzzy_HE_RGB")
        self.actionFuzzy_Grayscale = QtWidgets.QAction(MainWindow)
        self.actionFuzzy_Grayscale.setObjectName("actionFuzzy_Grayscale")
        self.actionIdentity = QtWidgets.QAction(MainWindow)
        self.actionIdentity.setObjectName("actionIdentity")
        self.actionSharpen = QtWidgets.QAction(MainWindow)
        self.actionSharpen.setObjectName("actionSharpen")
        self.actionUnshape_Masking = QtWidgets.QAction(MainWindow)
        self.actionUnshape_Masking.setObjectName("actionUnshape_Masking")
        self.actionAverage_Filter = QtWidgets.QAction(MainWindow)
        self.actionAverage_Filter.setObjectName("actionAverage_Filter")
        self.actionLow_Pass_Filter = QtWidgets.QAction(MainWindow)
        self.actionLow_Pass_Filter.setObjectName("actionLow_Pass_Filter")
        self.actionHigh_Pass_Filter = QtWidgets.QAction(MainWindow)
        self.actionHigh_Pass_Filter.setObjectName("actionHigh_Pass_Filter")
        self.actionBandstop_Filter = QtWidgets.QAction(MainWindow)
        self.actionBandstop_Filter.setObjectName("actionBandstop_Filter")
        self.actionEdge_Detection_1 = QtWidgets.QAction(MainWindow)
        self.actionEdge_Detection_1.setObjectName("actionEdge_Detection_1")
        self.actionEdge_Detection_2 = QtWidgets.QAction(MainWindow)
        self.actionEdge_Detection_2.setObjectName("actionEdge_Detection_2")
        self.actionEdge_Detection_3 = QtWidgets.QAction(MainWindow)
        self.actionEdge_Detection_3.setObjectName("actionEdge_Detection_3")
        self.actionGaussian_Blur_3x3 = QtWidgets.QAction(MainWindow)
        self.actionGaussian_Blur_3x3.setObjectName("actionGaussian_Blur_3x3")
        self.actionGaussian_Blur_5x5 = QtWidgets.QAction(MainWindow)
        self.actionGaussian_Blur_5x5.setObjectName("actionGaussian_Blur_5x5")
        self.actionPrewitt = QtWidgets.QAction(MainWindow)
        self.actionPrewitt.setObjectName("actionPrewitt")
        self.actionSobel = QtWidgets.QAction(MainWindow)
        self.actionSobel.setObjectName("actionSobel")
        self.actionSquare_3 = QtWidgets.QAction(MainWindow)
        self.actionSquare_3.setObjectName("actionSquare_3")
        self.actionSquare_5 = QtWidgets.QAction(MainWindow)
        self.actionSquare_5.setObjectName("actionSquare_5")
        self.actionCross_3 = QtWidgets.QAction(MainWindow)
        self.actionCross_3.setObjectName("actionCross_3")
        self.actionSquare_4 = QtWidgets.QAction(MainWindow)
        self.actionSquare_4.setObjectName("actionSquare_4")
        self.actionSquare_6 = QtWidgets.QAction(MainWindow)
        self.actionSquare_6.setObjectName("actionSquare_6")
        self.actionCross_4 = QtWidgets.QAction(MainWindow)
        self.actionCross_4.setObjectName("actionCross_4")
        self.actionSquare_9 = QtWidgets.QAction(MainWindow)
        self.actionSquare_9.setObjectName("actionSquare_9")
        self.actionSquare_10 = QtWidgets.QAction(MainWindow)
        self.actionSquare_10.setObjectName("actionSquare_10")
        self.actionOperasi_Aitmatika = QtWidgets.QAction(MainWindow)
        self.actionOperasi_Aitmatika.setObjectName("actionOperasi_Aitmatika")
        self.actionCroping = QtWidgets.QAction(MainWindow)
        self.actionCroping.setObjectName("actionCroping")
        self.actionTranslasi = QtWidgets.QAction(MainWindow)
        self.actionTranslasi.setObjectName("actionTranslasi")
        self.actionRotasi = QtWidgets.QAction(MainWindow)
        self.actionRotasi.setObjectName("actionRotasi")
        self.actionUniform_Scaling = QtWidgets.QAction(MainWindow)
        self.actionUniform_Scaling.setObjectName("actionUniform_Scaling")
        self.actionNon_Uniform_Scaling = QtWidgets.QAction(MainWindow)
        self.actionNon_Uniform_Scaling.setObjectName("actionNon_Uniform_Scaling")
        self.actionFlip_Horizontal = QtWidgets.QAction(MainWindow)
        self.actionFlip_Horizontal.setObjectName("actionFlip_Horizontal")
        self.actionFlip_Vertical = QtWidgets.QAction(MainWindow)
        self.actionFlip_Vertical.setObjectName("actionFlip_Vertical")
        self.menuScaling.addAction(self.actionUniform_Scaling)
        self.menuScaling.addAction(self.actionNon_Uniform_Scaling)
        self.menuFliping.addAction(self.actionFlip_Horizontal)
        self.menuFliping.addAction(self.actionFlip_Vertical)
        self.menuTentang.addAction(self.menuScaling.menuAction())
        self.menuTentang.addAction(self.menuFliping.menuAction())
        self.menuTentang.addAction(self.actionCroping)
        self.menuTentang.addAction(self.actionTranslasi)
        self.menuTentang.addAction(self.actionRotasi)
        self.menubar.addAction(self.menuTentang.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Geometri Operation"))
        self.menuTentang.setTitle(_translate("MainWindow", "Geometri"))
        self.menuScaling.setTitle(_translate("MainWindow", "Scaling"))
        self.menuFliping.setTitle(_translate("MainWindow", "Fliping"))
        self.actionBuka.setText(_translate("MainWindow", "Buka"))
        self.actionSimpan_Sebagai.setText(_translate("MainWindow", "Simpan Sebagai"))
        self.actionKeluar.setText(_translate("MainWindow", "Keluar"))
        self.actionInput.setText(_translate("MainWindow", "Input"))
        self.actionOutput.setText(_translate("MainWindow", "Output"))
        self.actionInput_Output.setText(_translate("MainWindow", "Input Output"))
        self.actionBrightness_Contras.setText(_translate("MainWindow", "Brightness - Contras"))
        self.actionInvers.setText(_translate("MainWindow", "Invers"))
        self.actionLog_Brightness.setText(_translate("MainWindow", "Log Brightness"))
        self.actionGamma_Corrections.setText(_translate("MainWindow", "Gamma Corrections"))
        self.actionKuning.setText(_translate("MainWindow", "Kuning"))
        self.actionCyan.setText(_translate("MainWindow", "Cyan"))
        self.actionUngu.setText(_translate("MainWindow", "Ungu"))
        self.actionAbu_abu.setText(_translate("MainWindow", "Abu-abu"))
        self.actionAvverage.setText(_translate("MainWindow", "Avverage"))
        self.actionLightness.setText(_translate("MainWindow", "Lightness"))
        self.actionLuminance.setText(_translate("MainWindow", "Luminance"))
        self.actionContrast.setText(_translate("MainWindow", "Contrast"))
        self.action1_Bit.setText(_translate("MainWindow", "1 bit"))
        self.action2_BIt.setText(_translate("MainWindow", "2 bit"))
        self.action3_bit.setText(_translate("MainWindow", "3 bit"))
        self.action4_bit.setText(_translate("MainWindow", "4 bit"))
        self.action5_bit.setText(_translate("MainWindow", "5 bit"))
        self.action6_bit.setText(_translate("MainWindow", "6 bit"))
        self.action7_bit.setText(_translate("MainWindow", "7 bit"))
        self.actionHistogram_Equalization.setText(_translate("MainWindow", "Histogram Equalization"))
        self.actionFuzzy_HE_RGB.setText(_translate("MainWindow", "Fuzzy HE RGB"))
        self.actionFuzzy_Grayscale.setText(_translate("MainWindow", "Fuzzy Grayscale"))
        self.actionIdentity.setText(_translate("MainWindow", "Identity"))
        self.actionSharpen.setText(_translate("MainWindow", "Sharpen"))
        self.actionUnshape_Masking.setText(_translate("MainWindow", "Unshape Masking"))
        self.actionAverage_Filter.setText(_translate("MainWindow", "Average Filter"))
        self.actionLow_Pass_Filter.setText(_translate("MainWindow", "Low Pass Filter"))
        self.actionHigh_Pass_Filter.setText(_translate("MainWindow", "High Pass Filter"))
        self.actionBandstop_Filter.setText(_translate("MainWindow", "Bandstop Filter"))
        self.actionEdge_Detection_1.setText(_translate("MainWindow", "Edge Detection 1"))
        self.actionEdge_Detection_2.setText(_translate("MainWindow", "Edge Detection 2"))
        self.actionEdge_Detection_3.setText(_translate("MainWindow", "Edge Detection 3"))
        self.actionGaussian_Blur_3x3.setText(_translate("MainWindow", "Gaussian Blur 3x3"))
        self.actionGaussian_Blur_5x5.setText(_translate("MainWindow", "Gaussian Blur 5x5"))
        self.actionPrewitt.setText(_translate("MainWindow", "Prewitt"))
        self.actionSobel.setText(_translate("MainWindow", "Sobel"))
        self.actionSquare_3.setText(_translate("MainWindow", "Square 3"))
        self.actionSquare_5.setText(_translate("MainWindow", "Square 5"))
        self.actionCross_3.setText(_translate("MainWindow", "Cross 3"))
        self.actionSquare_4.setText(_translate("MainWindow", "Square 3"))
        self.actionSquare_6.setText(_translate("MainWindow", "Square 5"))
        self.actionCross_4.setText(_translate("MainWindow", "Cross 3"))
        self.actionSquare_9.setText(_translate("MainWindow", "Square 9"))
        self.actionSquare_10.setText(_translate("MainWindow", "Square 9"))
        self.actionOperasi_Aitmatika.setText(_translate("MainWindow", "Operasi Aitmatika"))
        self.actionCroping.setText(_translate("MainWindow", "Croping"))
        self.actionTranslasi.setText(_translate("MainWindow", "Translasi"))
        self.actionRotasi.setText(_translate("MainWindow", "Rotasi"))
        self.actionUniform_Scaling.setText(_translate("MainWindow", "Uniform Scaling"))
        self.actionNon_Uniform_Scaling.setText(_translate("MainWindow", "Non Uniform Scaling"))
        self.actionFlip_Horizontal.setText(_translate("MainWindow", "Flip Horizontal"))
        self.actionFlip_Vertical.setText(_translate("MainWindow", "Flip Vertical"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Geometri()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
