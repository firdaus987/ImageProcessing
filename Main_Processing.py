from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QScrollArea, QPushButton, QAction
from Main_Aritmath import Ui_Aritmath 
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt


class Ui_Process(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 366)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 381, 281))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(410, 10, 381, 281))
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setText("")
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuHistogram = QtWidgets.QMenu(self.menuView)
        self.menuHistogram.setObjectName("menuHistogram")
        self.menuColors = QtWidgets.QMenu(self.menubar)
        self.menuColors.setObjectName("menuColors")
        self.menuRGB = QtWidgets.QMenu(self.menuColors)
        self.menuRGB.setObjectName("menuRGB")
        self.menuRGB_to_Grayscale = QtWidgets.QMenu(self.menuColors)
        self.menuRGB_to_Grayscale.setObjectName("menuRGB_to_Grayscale")
        self.menuBrightness = QtWidgets.QMenu(self.menuColors)
        self.menuBrightness.setObjectName("menuBrightness")
        self.menuBit_Depth = QtWidgets.QMenu(self.menuColors)
        self.menuBit_Depth.setObjectName("menuBit_Depth")
        self.menuTentang = QtWidgets.QMenu(self.menubar)
        self.menuTentang.setObjectName("menuTentang")
        self.menuImage_Processing = QtWidgets.QMenu(self.menubar)
        self.menuImage_Processing.setObjectName("menuImage_Processing")
        self.menuAritmatical_Operation = QtWidgets.QMenu(self.menubar)
        self.menuAritmatical_Operation.setObjectName("menuAritmatical_Operation")
        arithmetic_action = self.menuAritmatical_Operation.addAction("Aritmatika")
        arithmetic_action.triggered.connect(self.frameArimatika)
        self.menuIler = QtWidgets.QMenu(self.menubar)
        self.menuIler.setObjectName("menuIler")
        self.menuEdge_Detection_2 = QtWidgets.QMenu(self.menuIler)
        self.menuEdge_Detection_2.setObjectName("menuEdge_Detection_2")
        self.menuGaussian_Blur = QtWidgets.QMenu(self.menuIler)
        self.menuGaussian_Blur.setObjectName("menuGaussian_Blur")
        self.menuEdge_Detection = QtWidgets.QMenu(self.menubar)
        self.menuEdge_Detection.setObjectName("menuEdge_Detection")
        self.menuMorfologi = QtWidgets.QMenu(self.menubar)
        self.menuMorfologi.setObjectName("menuMorfologi")
        self.menuErosion = QtWidgets.QMenu(self.menuMorfologi)
        self.menuErosion.setObjectName("menuErosion")
        self.menuDilation = QtWidgets.QMenu(self.menuMorfologi)
        self.menuDilation.setObjectName("menuDilation")
        self.menuOpening = QtWidgets.QMenu(self.menuMorfologi)
        self.menuOpening.setObjectName("menuOpening")
        self.menuClosing = QtWidgets.QMenu(self.menuMorfologi)
        self.menuClosing.setObjectName("menuClosing")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEkstraksiFitur = QtWidgets.QMenu(self.menubar)
        self.menuEkstraksiFitur.setObjectName("menuEkstraksiFitur")
        MainWindow.setMenuBar(self.menubar)
        self.actionInput = QtWidgets.QAction(MainWindow)
        self.actionInput.setObjectName("actionInput")
        # self.actionInput.triggered.connect(self.histogram)
        self.actionInput.triggered.connect(self.input_histogram)
        self.actionfliphorizontal = QtWidgets.QAction(MainWindow)
        self.actionfliphorizontal.setObjectName("actionFlipHorizontal")
        self.actionfliphorizontal.triggered.connect(self.flip_horizontal)
        self.actionflipvertical = QtWidgets.QAction(MainWindow)
        self.actionflipvertical.setObjectName("actionFlipVertical")
        self.actionflipvertical.triggered.connect(self.flip_vertical)
        self.actionRotate = QtWidgets.QAction(MainWindow)
        self.actionRotate.setObjectName("ActionRotate")
        self.actionRotate.triggered.connect(self.rotasiGambare)
        self.actionCroping = QtWidgets.QAction(MainWindow)
        self.actionCroping.setObjectName("ActionCroping")
        # self.actionCroping.triggered.connect(self.cropImage)
        self.actionOutput = QtWidgets.QAction(MainWindow)
        self.actionOutput.setObjectName("actionOutput")
        self.actionOutput.triggered.connect(self.Output_histogram)
        self.actionInput_Output = QtWidgets.QAction(MainWindow)
        self.actionInput_Output.setObjectName("actionInput_Output")
        self.actionInput_Output.triggered.connect(self.input_histogram)
        self.actionInput_Output.triggered.connect(self.Output_histogram)
        self.actionBrightness_Contrast = QtWidgets.QAction(MainWindow)
        self.actionBrightness_Contrast.setObjectName("actionBrightness_Contrast")
        self.actionBrightness_Contrast.triggered.connect(self.open_brightness_contrast_dialog)
        self.actionInvers = QtWidgets.QAction(MainWindow)
        self.actionInvers.setObjectName("actionInvers")
        self.actionInvers.triggered.connect(self.Invers)
        self.actionTreshold = QtWidgets.QAction(MainWindow)
        self.actionTreshold.setObjectName("actionTreshold")
        self.actionTreshold.triggered.connect(self.Thresholding)
        self.actionLog_Brightness = QtWidgets.QAction(MainWindow)
        self.actionLog_Brightness.setObjectName("actionLog_Brightness")
        self.actionLog_Brightness.triggered.connect(self.segmentImage)
        self.actionGamma_Corretion = QtWidgets.QAction(MainWindow)
        self.actionGamma_Corretion.setObjectName("actionGamma_Corretion")
        # self.actionGamma_Corretion.triggered.connect(self.removeBackground)
        self.actionKuning = QtWidgets.QAction(MainWindow)
        self.actionKuning.setObjectName("actionKuning")
        self.actionKuning.triggered.connect(self.Kuning)
        self.actionOrange = QtWidgets.QAction(MainWindow)
        self.actionOrange.setObjectName("actionOrange")
        self.actionOrange.triggered.connect(self.Orange)
        self.actionCyan = QtWidgets.QAction(MainWindow)
        self.actionCyan.setObjectName("actionCyan")
        self.actionCyan.triggered.connect(self.Cyan)
        self.actionPurple = QtWidgets.QAction(MainWindow)
        self.actionPurple.setObjectName("actionPurple")
        self.actionPurple.triggered.connect(self.Purple)
        self.actionGray = QtWidgets.QAction(MainWindow)
        self.actionGray.setObjectName("actionGray")
        self.actionGray.triggered.connect(self.Gray)
        self.actionCoklat = QtWidgets.QAction(MainWindow)
        self.actionCoklat.setObjectName("actionCoklat")
        self.actionCoklat.triggered.connect(self.Coklat)
        self.actionMerah = QtWidgets.QAction(MainWindow)
        self.actionMerah.setObjectName("actionMerah")
        self.actionMerah.triggered.connect(self.Merah)
        self.actionAverage = QtWidgets.QAction(MainWindow)
        self.actionAverage.setObjectName("actionAverage")
        self.actionAverage.triggered.connect(self.Average)
        self.actionLightness = QtWidgets.QAction(MainWindow)
        self.actionLightness.setObjectName("actionLightness")
        self.actionLightness.triggered.connect(self.Lightness)
        self.actionLuminance = QtWidgets.QAction(MainWindow)
        self.actionLuminance.setObjectName("actionLuminance")
        self.actionLuminance.triggered.connect(self.Luminosity)
        self.actionContrast = QtWidgets.QAction(MainWindow)
        self.actionContrast.setObjectName("actionContrast")
        self.actionContrast.triggered.connect(self.ContrastEffect)
        self.action1_bit = QtWidgets.QAction(MainWindow)
        self.action1_bit.setObjectName("action1_bit")
        self.action1_bit.triggered.connect(self.Bit1)
        self.action2_bit = QtWidgets.QAction(MainWindow)
        self.action2_bit.setObjectName("action2_bit")
        self.action2_bit.triggered.connect(self.Bit2)
        self.action3_bit = QtWidgets.QAction(MainWindow)
        self.action3_bit.setObjectName("action3_bit")
        self.action3_bit.triggered.connect(self.Bit3)
        self.action4_bit = QtWidgets.QAction(MainWindow)
        self.action4_bit.setObjectName("action4_bit")
        self.action4_bit.triggered.connect(self.Bit4)
        self.action5_bit = QtWidgets.QAction(MainWindow)
        self.action5_bit.setObjectName("action5_bit")
        self.action5_bit.triggered.connect(self.Bit5)
        self.action6_bit = QtWidgets.QAction(MainWindow)
        self.action6_bit.setObjectName("action6_bit")
        self.action6_bit.triggered.connect(self.Bit6)
        self.action7_bit = QtWidgets.QAction(MainWindow)
        self.action7_bit.setObjectName("action7_bit")
        self.action7_bit.triggered.connect(self.Bit7)
        self.action8_bit = QtWidgets.QAction(MainWindow)
        self.action8_bit.setObjectName("action7_bit")
        self.action8_bit.triggered.connect(self.Bit8)
        self.actionHistogram_Equalization = QtWidgets.QAction(MainWindow)
        self.actionHistogram_Equalization.setObjectName("actionHistogram_Equalization")
        self.actionHistogram_Equalization.triggered.connect(self.HistogramEqualization)
        self.actionFuzzy_HE_RGB = QtWidgets.QAction(MainWindow)
        self.actionFuzzy_HE_RGB.setObjectName("actionFuzzy_HE_RGB")
        self.actionFuzzy_HE_RGB.triggered.connect(self.Fuzzy_HE_RGB)
        self.actionFuzzy_Grayscale = QtWidgets.QAction(MainWindow)
        self.actionFuzzy_Grayscale.setObjectName("actionFuzzy_Grayscale")
        self.actionFuzzy_Grayscale.triggered.connect(self.fuzzy_greyscale)
        self.actionIdentity = QtWidgets.QAction(MainWindow)
        self.actionIdentity.setObjectName("actionIdentity")
        self.actionIdentity.triggered.connect(self.Identity)
        self.actionSharpen = QtWidgets.QAction(MainWindow)
        self.actionSharpen.setObjectName("actionSharpen")
        self.actionSharpen.triggered.connect(self.sharpenImage)
        self.actionUnsharp_Masking = QtWidgets.QAction(MainWindow)
        self.actionUnsharp_Masking.setObjectName("actionUnsharp_Masking")
        self.actionUnsharp_Masking.triggered.connect(self.applyUnsharpMasking)
        self.actionAvarage_Filter = QtWidgets.QAction(MainWindow)
        self.actionAvarage_Filter.setObjectName("actionAvarage_Filter")
        self.actionLow_Pass_Filler = QtWidgets.QAction(MainWindow)
        self.actionLow_Pass_Filler.setObjectName("actionLow_Pass_Filler")
        self.actionLow_Pass_Filler.triggered.connect(self.applyLowpassFilter)
        self.actionHight_Pass_Filter = QtWidgets.QAction(MainWindow)
        self.actionHight_Pass_Filter.setObjectName("actionHight_Pass_Filter")
        self.actionHight_Pass_Filter.triggered.connect(self.applyHighPassFilter)
        self.actionBandstop_Filter = QtWidgets.QAction(MainWindow)
        self.actionBandstop_Filter.setObjectName("actionBandstop_Filter")
        self.actionEdge_Detection_1 = QtWidgets.QAction(MainWindow)
        self.actionEdge_Detection_1.setObjectName("actionEdge_Detection_1")
        self.actionEdge_Detection_2 = QtWidgets.QAction(MainWindow)
        self.actionEdge_Detection_2.setObjectName("actionEdge_Detection_2")
        self.actionEdge_Detection_3 = QtWidgets.QAction(MainWindow)
        self.actionEdge_Detection_3.setObjectName("actionEdge_Detection_3")
        self.actionGaussian_Blur_3_x_3 = QtWidgets.QAction(MainWindow)
        self.actionGaussian_Blur_3_x_3.setObjectName("actionGaussian_Blur_3_x_3")
        self.actionGaussian_Blur_3_x_3.triggered.connect(self.applyGaussianBlur3)
        self.actionGaussian_Blur_5_X_5 = QtWidgets.QAction(MainWindow)
        self.actionGaussian_Blur_5_X_5.setObjectName("actionGaussian_Blur_5_X_5")
        self.actionGaussian_Blur_5_X_5.triggered.connect(self.applyGaussianBlur5)
        self.actionPrewitt = QtWidgets.QAction(MainWindow)
        self.actionPrewitt.setObjectName("actionPrewitt")
        self.actionPrewitt.triggered.connect(self.prewittEdgeDetection)
        self.actionSobel = QtWidgets.QAction(MainWindow)
        self.actionSobel.setObjectName("actionSobel")
        self.actionSobel.triggered.connect(self.sobelEdgeDetection)
        self.actionRobert = QtWidgets.QAction(MainWindow)
        self.actionRobert.setObjectName("actionRobert")
        self.actionRobert.triggered.connect(self.applyRobertFilter)
        self.actionSquare_3 = QtWidgets.QAction(MainWindow)
        self.actionSquare_3.setObjectName("actionSquare_3")
        self.actionSquare_3.triggered.connect(self.erodeSquare3)
        self.actionSquare_5 = QtWidgets.QAction(MainWindow)
        self.actionSquare_5.setObjectName("actionSquare_5")
        self.actionSquare_5.triggered.connect(self.erodeSquare5)
        self.actionCross_3 = QtWidgets.QAction(MainWindow)
        self.actionCross_3.setObjectName("actionCross_3")
        self.actionCross_3.triggered.connect(self.erodeCross3)
        self.actionSquare_4 = QtWidgets.QAction(MainWindow)
        self.actionSquare_4.setObjectName("actionSquare_4")
        self.actionSquare_4.triggered.connect(self.dilateSquare3)
        self.actionSquare_6 = QtWidgets.QAction(MainWindow)
        self.actionSquare_6.setObjectName("actionSquare_6")
        self.actionSquare_6.triggered.connect(self.dilateSquare5)
        self.actionCross = QtWidgets.QAction(MainWindow)
        self.actionCross.setObjectName("actionCross")
        self.actionCross.triggered.connect(self.dilateCross3)
        self.actionSquare_9 = QtWidgets.QAction(MainWindow)
        self.actionSquare_9.setObjectName("actionSquare_9")
        self.actionSquare_9.triggered.connect(self.applyOpening)
        self.actionSquare_10 = QtWidgets.QAction(MainWindow)
        self.actionSquare_10.setObjectName("actionSquare_10")
        self.actionSquare_10.triggered.connect(self.applyClosingOperation)
        self.actionWarnaRGB = QtWidgets.QAction(MainWindow)
        self.actionWarnaRGB.setObjectName("actionWarnaRGB")
        self.actionWarnaRGB.triggered.connect(self.extractfeaturesRGB)
        self.actionWarnaRGBtoHSV= QtWidgets.QAction(MainWindow)
        self.actionWarnaRGBtoHSV.setObjectName("actionWarnaRGBtoHSV")
        self.actionWarnaRGBtoYCrCb= QtWidgets.QAction(MainWindow)
        self.actionWarnaRGBtoYCrCb.setObjectName("actionWarnaRGBtoYCrCb")
        self.actionOpen_File = QtWidgets.QAction(MainWindow)
        self.actionOpen_File.setObjectName("actionOpen_File")
        self.actionOpen_File.triggered.connect(self.openFile)
        self.actionSave_As = QtWidgets.QAction(MainWindow)
        self.actionSave_As.setObjectName("actionSave_As")
        self.actionSave_As.triggered.connect(self.saveImage)
        self.actionKeluar = QtWidgets.QAction(MainWindow)
        self.actionKeluar.setObjectName("actionKeluar")
        self.actionKeluar.triggered.connect(self.exitApplication)
        self.menuHistogram.addAction(self.actionInput)
        self.menuHistogram.addAction(self.actionOutput)
        self.menuHistogram.addAction(self.actionInput_Output)
        self.menuView.addAction(self.menuHistogram.menuAction())
        self.menuRGB.addAction(self.actionKuning)
        self.menuRGB.addAction(self.actionOrange)
        self.menuRGB.addAction(self.actionCyan)
        self.menuRGB.addAction(self.actionPurple)
        self.menuRGB.addAction(self.actionGray)
        self.menuRGB.addAction(self.actionCoklat)
        self.menuRGB.addAction(self.actionMerah)
        self.menuRGB_to_Grayscale.addAction(self.actionAverage)
        self.menuRGB_to_Grayscale.addAction(self.actionLightness)
        self.menuRGB_to_Grayscale.addAction(self.actionLuminance)
        self.menuBrightness.addAction(self.actionContrast)
        self.menuBit_Depth.addAction(self.action1_bit)
        self.menuBit_Depth.addAction(self.action2_bit)
        self.menuBit_Depth.addAction(self.action3_bit)
        self.menuBit_Depth.addAction(self.action4_bit)
        self.menuBit_Depth.addAction(self.action5_bit)
        self.menuBit_Depth.addAction(self.action6_bit)
        self.menuBit_Depth.addAction(self.action7_bit)
        self.menuBit_Depth.addAction(self.action8_bit)
        self.menuColors.addAction(self.menuRGB.menuAction())
        self.menuColors.addAction(self.menuRGB_to_Grayscale.menuAction())
        self.menuColors.addAction(self.menuBrightness.menuAction())
        self.menuColors.addAction(self.actionBrightness_Contrast)
        self.menuColors.addAction(self.actionInvers)
        self.menuColors.addAction(self.actionTreshold)
        self.menuColors.addAction(self.actionLog_Brightness)
        self.menuColors.addAction(self.menuBit_Depth.menuAction())
        self.menuColors.addAction(self.actionGamma_Corretion)
        self.menuImage_Processing.addAction(self.actionHistogram_Equalization)
        self.menuImage_Processing.addAction(self.actionFuzzy_HE_RGB)
        self.menuImage_Processing.addAction(self.actionFuzzy_Grayscale)
        self.menuEdge_Detection_2.addAction(self.actionEdge_Detection_1)
        self.menuEdge_Detection_2.addAction(self.actionEdge_Detection_2)
        self.menuEdge_Detection_2.addAction(self.actionEdge_Detection_3)
        self.menuGaussian_Blur.addAction(self.actionGaussian_Blur_3_x_3)
        self.menuGaussian_Blur.addAction(self.actionGaussian_Blur_5_X_5)
        self.menuIler.addAction(self.actionIdentity)
        self.menuIler.addAction(self.menuEdge_Detection_2.menuAction())
        self.menuIler.addAction(self.actionSharpen)
        self.menuIler.addAction(self.menuGaussian_Blur.menuAction())
        self.menuIler.addAction(self.actionUnsharp_Masking)        
        self.menuIler.addAction(self.actionLow_Pass_Filler)
        self.menuIler.addAction(self.actionHight_Pass_Filter)
        self.menuIler.addAction(self.actionAvarage_Filter)
        self.menuIler.addAction(self.actionBandstop_Filter)
        self.menuEdge_Detection.addAction(self.actionPrewitt)
        self.menuEdge_Detection.addAction(self.actionSobel)
        self.menuEdge_Detection.addAction(self.actionRobert)
        self.menuErosion.addAction(self.actionSquare_3)
        self.menuErosion.addAction(self.actionSquare_5)
        self.menuErosion.addAction(self.actionCross_3)
        self.menuDilation.addAction(self.actionSquare_4)
        self.menuDilation.addAction(self.actionSquare_6)
        self.menuDilation.addAction(self.actionCross)
        self.menuOpening.addAction(self.actionSquare_9)
        self.menuClosing.addAction(self.actionSquare_10)
        self.menuMorfologi.addAction(self.menuErosion.menuAction())
        self.menuMorfologi.addAction(self.menuDilation.menuAction())
        self.menuMorfologi.addAction(self.menuOpening.menuAction())
        self.menuMorfologi.addAction(self.menuClosing.menuAction())
        self.menuFile.addAction(self.actionOpen_File)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addAction(self.actionKeluar)
        self.menuTentang.addAction(self.actionfliphorizontal)
        self.menuTentang.addAction(self.actionflipvertical)
        self.menuTentang.addAction(self.actionRotate)
        self.menuTentang.addAction(self.actionCroping)
        self.menuEkstraksiFitur.addAction(self.actionWarnaRGB)
        self.menuEkstraksiFitur.addAction(self.actionWarnaRGBtoHSV)
        self.menuEkstraksiFitur.addAction(self.actionWarnaRGBtoYCrCb)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuColors.menuAction())
        self.menubar.addAction(self.menuTentang.menuAction())
        self.menubar.addAction(self.menuImage_Processing.menuAction())
        self.menubar.addAction(self.menuAritmatical_Operation.menuAction())
        self.menubar.addAction(self.menuIler.menuAction())
        self.menubar.addAction(self.menuEdge_Detection.menuAction())
        self.menubar.addAction(self.menuMorfologi.menuAction())
        self.menubar.addAction(self.menuEkstraksiFitur.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.menuHistogram.setTitle(_translate("MainWindow", "Histogram"))
        self.menuColors.setTitle(_translate("MainWindow", "Colors"))
        self.menuRGB.setTitle(_translate("MainWindow", "RGB"))
        self.menuRGB_to_Grayscale.setTitle(_translate("MainWindow", "RGB to Grayscale"))
        self.menuBrightness.setTitle(_translate("MainWindow", "Brightness"))
        self.menuBit_Depth.setTitle(_translate("MainWindow", "Bit Depth"))
        self.menuTentang.setTitle(_translate("MainWindow", "Geometri"))
        self.menuImage_Processing.setTitle(_translate("MainWindow", "Image Processing"))
        self.menuAritmatical_Operation.setTitle(_translate("MainWindow", "Aritmatical Operation"))
        self.menuIler.setTitle(_translate("MainWindow", "Filter"))
        self.menuEdge_Detection_2.setTitle(_translate("MainWindow", "Edge Detection"))
        self.menuGaussian_Blur.setTitle(_translate("MainWindow", "Gaussian Blur"))
        self.menuEdge_Detection.setTitle(_translate("MainWindow", "Edge Detection"))
        self.menuMorfologi.setTitle(_translate("MainWindow", "Morfologi"))
        self.menuErosion.setTitle(_translate("MainWindow", "Erosion"))
        self.menuDilation.setTitle(_translate("MainWindow", "Dilation"))
        self.menuOpening.setTitle(_translate("MainWindow", "Opening"))
        self.menuClosing.setTitle(_translate("MainWindow", "Closing"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEkstraksiFitur.setTitle(_translate("MainWindow", "Ekstraksi"))
        self.actionInput.setText(_translate("MainWindow", "Input"))
        self.actionOutput.setText(_translate("MainWindow", "Output"))
        self.actionInput_Output.setText(_translate("MainWindow", "Input Output"))
        self.actionBrightness_Contrast.setText(_translate("MainWindow", "Brightness - Contrast"))
        self.actionInvers.setText(_translate("MainWindow", "Invers"))
        self.actionTreshold.setText(_translate("MainWindow", "Treshold"))
        self.actionLog_Brightness.setText(_translate("MainWindow", "Segmentasi Citra"))
        self.actionGamma_Corretion.setText(_translate("MainWindow", "ROI"))
        self.actionKuning.setText(_translate("MainWindow", "Kuning"))
        self.actionOrange.setText(_translate("MainWindow", "Orange"))
        self.actionCyan.setText(_translate("MainWindow", "Cyan"))
        self.actionPurple.setText(_translate("MainWindow", "Ungu"))
        self.actionGray.setText(_translate("MainWindow", "Gray"))
        self.actionCoklat.setText(_translate("MainWindow", "Coklat"))
        self.actionMerah.setText(_translate("MainWindow", "Merah"))
        self.actionAverage.setText(_translate("MainWindow", "Average"))
        self.actionLightness.setText(_translate("MainWindow", "Lightness"))
        self.actionLuminance.setText(_translate("MainWindow", "Luminance"))
        self.actionContrast.setText(_translate("MainWindow", "Contrast"))
        self.action1_bit.setText(_translate("MainWindow", "1 Bit"))
        self.action2_bit.setText(_translate("MainWindow", "2 Bit"))
        self.action3_bit.setText(_translate("MainWindow", "3 Bit"))
        self.action4_bit.setText(_translate("MainWindow", "4 Bit"))
        self.action5_bit.setText(_translate("MainWindow", "5 Bit"))
        self.action6_bit.setText(_translate("MainWindow", "6 Bit"))
        self.action7_bit.setText(_translate("MainWindow", "7 Bit"))
        self.action8_bit.setText(_translate("MainWindow", "8 Bit"))
        self.actionHistogram_Equalization.setText(_translate("MainWindow", "Histogram Equalization"))
        self.actionFuzzy_HE_RGB.setText(_translate("MainWindow", "Fuzzy HE RGB"))
        self.actionFuzzy_Grayscale.setText(_translate("MainWindow", "Fuzzy Grayscale"))
        self.actionIdentity.setText(_translate("MainWindow", "Identity"))
        self.actionSharpen.setText(_translate("MainWindow", "Sharpen"))
        self.actionUnsharp_Masking.setText(_translate("MainWindow", "Unsharp Masking"))
        self.actionLow_Pass_Filler.setText(_translate("MainWindow", "Low Pass Filter"))
        self.actionHight_Pass_Filter.setText(_translate("MainWindow", "Hight Pass Filter"))
        self.actionAvarage_Filter.setText(_translate("MainWindow", "Uniform Scaling"))
        self.actionBandstop_Filter.setText(_translate("MainWindow", "Non-Uniform Scaling"))
        self.actionEdge_Detection_1.setText(_translate("MainWindow", "Edge Detection 1"))
        self.actionEdge_Detection_2.setText(_translate("MainWindow", "Edge Detection 2"))
        self.actionEdge_Detection_3.setText(_translate("MainWindow", "Edge Detection 3"))
        self.actionGaussian_Blur_3_x_3.setText(_translate("MainWindow", "Gaussian Blur 3 x 3"))
        self.actionGaussian_Blur_5_X_5.setText(_translate("MainWindow", "Gaussian Blur 5 X 5 "))
        self.actionPrewitt.setText(_translate("MainWindow", "Prewitt"))
        self.actionSobel.setText(_translate("MainWindow", "Sobel"))
        self.actionRobert.setText(_translate("MainWindow", "Robert"))
        self.actionSquare_3.setText(_translate("MainWindow", "Square 3"))
        self.actionSquare_5.setText(_translate("MainWindow", "Square 5"))
        self.actionCross_3.setText(_translate("MainWindow", "Cross 3"))
        self.actionSquare_4.setText(_translate("MainWindow", "Square 3"))
        self.actionSquare_6.setText(_translate("MainWindow", "Square 5"))
        self.actionCross.setText(_translate("MainWindow", "Cross 3"))
        self.actionSquare_9.setText(_translate("MainWindow", "Square 9"))
        self.actionSquare_10.setText(_translate("MainWindow", "Square 9"))
        self.actionOpen_File.setText(_translate("MainWindow", "Buka"))
        self.actionSave_As.setText(_translate("MainWindow", "Simpan"))
        self.actionKeluar.setText(_translate("MainWindow", "Keluar"))
        self.actionfliphorizontal.setText(_translate("mainWindow" , "Flip Horizontal"))
        self.actionflipvertical.setText(_translate("MainWindow","Flip Vertical"))
        self.actionRotate.setText(_translate("MainWindow","Rotate"))
        self.actionWarnaRGB.setText(_translate("MainWindow","WarnaRGB"))
        self.actionWarnaRGBtoHSV.setText(_translate("MainWindow","WarnaRGBtoHSV"))
        self.actionWarnaRGBtoYCrCb.setText(_translate("MainWindow","WarnaRGBtoYCrCb"))
        self.actionCroping.setText(_translate("MainWindow","Croping"))


        


    

    def frameArimatika(self): 
        # Definisi fungsi frameArimatika yang merupakan metode dari suatu kelas. Fungsi ini mungkin digunakan untuk membuat dan menampilkan sebuah jendela Qt.
        self.window = QtWidgets.QMainWindow() # Membuat sebuah objek jendela utama dari Qt.
        self.ui = Ui_Aritmath() # Membuat sebuah objek dari kelas latihan.
        self.ui.setupUi(self.window) # Memanggil metode setupUi dari objek ui untuk menginisialisasi tampilan antarmuka.
        self.window.show()  # Menampilkan jendela utama. 
        
    def saveImage(self):
        # Inisialisasi opsi untuk dialog pemilihan berkas
        options = QFileDialog.Options()
        # Menambahkan opsi mode baca saja ke dalam opsi dialog
        options |= QFileDialog.ReadOnly 
        # menampung file path dari dialog open file dan difilter hanya format png , jpg , bmp
        file_name, _ = QFileDialog.getSaveFileName(None, "Save Image File", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)
        # check apakah terdapat path file
        if file_name:
            #Simpan gambar yang telah diformat
            pixmap = self.label_2.pixmap()
            pixmap.save(file_name)
            # self.label_4.setText(file_name)
            
    def rotasiGambare(self): # Definisi fungsi rotasiGambare yang mungkin merupakan metode dari suatu kelas.

        # Meminta pengguna untuk memasukkan sudut rotasi dalam derajat.
        rotation , ok = QtWidgets.QInputDialog.getInt(None , "Rotate Image","Enter rotation angle (degress):",0,-360,360) 
        if ok: # Memeriksa apakah input dari pengguna valid (tombol "OK" ditekan).
            # Mengambil gambar saat ini dari label pertama (self.label) dan label kedua (self.label_2).
            current_pixmap = self.label.pixmap()
            second_pixmap = self.label_2.pixmap()
            
            # Mengecek apakah gambar pada label kedua (self.label_2) sudah ada atau belum.
            if self.label_2.pixmap() is None:
                    # Jika belum ada, maka rotasi dilakukan pada gambar dari label pertama.
                    rotated_image = current_pixmap.transformed(QtGui.QTransform().rotate(rotation))
                    # rotated_pixmap = QtGui.QPixmap.fromImage(rotated_image)
                    self.label_2.setPixmap(rotated_image)  # Menetapkan gambar yang sudah dirotasi ke label kedua.
                    self.label_2.setAlignment(QtCore.Qt.AlignCenter)
                    self.label_2.setScaledContents(True)
                    self.image = rotated_image.toImage() # Mengambil objek QImage dari gambar yang sudah dirotasi.
                    
            else:   
                    # Jika sudah ada, maka rotasi dilakukan pada gambar dari label kedua. 
                    rotated_image = second_pixmap.transformed(QtGui.QTransform().rotate(rotation))
                    # rotated_pixmap = QtGui.QPixmap.fromImage(rotated_image)
                    self.label_2.setPixmap(rotated_image) # Menetapkan gambar yang sudah dirotasi ke label kedua.
                    self.label_2.setAlignment(QtCore.Qt.AlignCenter)
                    self.label_2.setScaledContents(True)
                    self.image = rotated_image.toImage() # Mengambil objek QImage dari gambar yang sudah dirotasi.
                                   
                
    def Average(self): #
        width = self.image.width()
        height = self.image.height()
        
        average = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)
        
        # Iterasi melalui semua piksel gambar asli.
        for y in range(height):
            for x in range(width):
                # Mendapatkan warna piksel (r, g, b) dari gambar asli.
                pixel_color = QtGui.QColor(self.image.pixel(x, y))
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                rumusAverage = int((r + g + b)/3) # Menghitung nilai rata-rata dari komponen warna (grayscale).
                # Membuat objek QColor baru dengan komponen warna grayscale.
                grayscale_color = QtGui.QColor(rumusAverage, rumusAverage, rumusAverage)
                average.setPixelColor(x, y, grayscale_color) # Menetapkan warna grayscale ke piksel yang sesuai di QImage baru.
        
        p = QtGui.QPixmap.fromImage(average) # Mengonversi QImage hasil rata-rata ke QPixmap.
        self.label_2.setPixmap(p) # Menampilkan QPixmap di label_2 dalam antarmuka pengguna.
        self.image = average # Mengganti gambar asli dengan gambar hasil rata-rata.
          
    def Luminosity(self): 
        width = self.image.width() 
        height = self.image.height()
 
        grayscale_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)
        # Melakukan iterasi melalui setiap piksel dalam gambar asli.
        for y in range(height):
            for x in range(width):
                # Mendapatkan warna piksel dari gambar asli.
                pixel_color = QtGui.QColor(self.image.pixel(x, y))
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                # Menghitung nilai luminance menggunakan rumus yang diberikan.
                rumusLuminance = int(0.299 * r + 0.587 * g + 0.114 * b)
                grayscale_color = QtGui.QColor(rumusLuminance, rumusLuminance, rumusLuminance) # Membuat objek QColor baru untuk warna grayscale.
                grayscale_image.setPixelColor(x, y, grayscale_color) # Mengatur warna piksel di gambar grayscale dengan warna grayscale yang dihitung.
        
        p = QtGui.QPixmap.fromImage(grayscale_image)  # Mengonversi gambar grayscale menjadi QPixmap.
        self.label_2.setPixmap(p)  # Menampilkan gambar grayscale di label_2 (mungkin adalah elemen GUI).
        self.image = grayscale_image # Mengganti gambar asli dengan gambar grayscale yang baru dibuat.

    #INVERS
    def Invers(self):
        # Ambil pixmap dari label 1
        pixmap = self.label.pixmap()
        if pixmap:
            img = pixmap.toImage()
            width, height = img.width(), img.height()

            for y in range(height):
                for x in range(width):
                    pixel = img.pixel(x, y)
                    # Dapatkan nilai merah (red), hijau (green), dan biru (blue) dari pixel
                    red = QtGui.qRed(pixel)
                    green = QtGui.qGreen(pixel)
                    blue = QtGui.qBlue(pixel)
                    # Inversi warna
                    inverted_color = QtGui.QColor(255 - red, 255 - green, 255 - blue)
                    # Set pixel ke warna invers pada gambar
                    img.setPixel(x, y, inverted_color.rgb())

            # Terapkan gambar invers pada label 2
            pixmap = QtGui.QPixmap.fromImage(img)
            self.label_2.setPixmap(pixmap)
            self.label_2.setScaledContents(True)

    #BRIGHTNESS
    def ContrastEffect(self):
        if self.label:
 
            width = self.image.width()
            height = self.image.height()
 
            output_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Faktor kontras (misalnya, 1.5 untuk meningkatkan kontras)
            contrast_factor = 1.5
            # Loop melalui semua pixel dalam gambar dan mengubah kontrasnya.
            for x in range(width):
                for y in range(height):
                    pixel_color = QtGui.QColor(*self.image.pixelColor(x, y).getRgb())

                    # Mengubah nilai warna pixel dengan faktor kontras
                    new_red = self.ContrastToColor(pixel_color.red(), contrast_factor)
                    new_green = self.ContrastToColor(pixel_color.green(), contrast_factor)
                    new_blue = self.ContrastToColor(pixel_color.blue(), contrast_factor)

                    # Menetapkan warna baru untuk pixel
                    output_image.setPixelColor(x, y, QtGui.QColor(new_red, new_green, new_blue))

            # Mengubah QPixmap hasil ke dalam QLabel
            output_pixmap = QtGui.QPixmap.fromImage(output_image)
            self.label_2.setPixmap(output_pixmap)

            # Mengganti gambar asli (self.image) dengan gambar yang telah diubah kontrasnya.
            self.image = output_image
        
    def ContrastToColor(self, color_value, contrast_factor):
        # Mengaplikasikan kontras pada nilai warna individual
        new_color_value = (color_value - 128) * contrast_factor + 128
        # Memastikan nilai warna berada dalam rentang 0 hingga 255.
        return max(0, min(255, int(new_color_value)))    
    
    def openFile(self):
        options = QFileDialog.Options() # Inisialisasi opsi untuk dialog pemilihan berkas
        options |= QFileDialog.ReadOnly # Menambahkan opsi mode baca saja ke dalam opsi dialog
        # menghapus gambar dari label kedua
        self.label_2.clear()
 
        # menampung file path dari dialog open file dan difilter hanya format png , jpg , bmp
        file_name, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)
        # check apakah terdapat path file
        if file_name:
            # untuk membuat objek QImage dari suatu berkas gambar dengan nama file_name
            image = QtGui.QImage(file_name)
            # check varibale apakah tidak kosong
            if not image.isNull():
                # simpan gambar pada variable pixmap
                pixmap = QtGui.QPixmap.fromImage(image)
                self.label.setPixmap(pixmap)  # Set gambar pada label
                self.label.setScaledContents(True) # set  kontennya agar sesuai dengan ukuran label.
                self.image = image # menetapkan objek QImage yang sudah dibuat sebelumnya (dalam contoh kode sebelumnya) ke atribut self.image dari kelas atau objek saat ini
                self.checkHisto = file_name

                         
    def Lightness(self):
        width = self.image.width()
        height = self.image.height()

        # Membuat objek QImage dengan ukuran yang sama dengan gambar awal dan format RGB32.
        lightness = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)
        
        # Looping melalui setiap pixel dalam gambar.
        for y in range(height):
            for x in range(width):
                pixel_color = QtGui.QColor(self.image.pixel(x, y))
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                rumusLightness = int((max(r,g,b)+ min(r,g,b))/2) # Menghitung nilai kecerahan (lightness) berdasarkan rumus.
                # Membuat objek QColor untuk warna skala abu-abu berdasarkan nilai kecerahan.
                grayscale_color = QtGui.QColor(rumusLightness, rumusLightness, rumusLightness)
                lightness.setPixelColor(x, y, grayscale_color) # Menetapkan warna skala abu-abu ke pixel dalam objek QImage yang baru.
        
        p = QtGui.QPixmap.fromImage(lightness) # Membuat objek QPixmap dari objek QImage yang baru.
        self.label_2.setPixmap(p) # Menetapkan QPixmap ke label_2 (mungkin adalah elemen GUI).
        self.image = lightness # Mengganti gambar asli dengan gambar yang telah diubah menjadi skala abu-abu.
        
    def histogram(self):
        img = cv2.imread(self.checkHisto , 1)
        # alternative way to find histogram of an image
        plt.hist(img.ravel(),256,[0,256])
        plt.show()
           
    def apply_brightness_contrast(self, brightness, contrast): 
        # Mendapatkan lebar dan tinggi gambar.
        width = self.image.width()
        height = self.image.height()
        # Membuat array numpy untuk menyimpan gambar yang sudah disesuaikan.
        adjusted_image = np.zeros((height, width, 4), dtype=np.uint8)

        # Melakukan perulangan untuk setiap piksel gambar.
        for y in range(height):
            for x in range(width):
                # Mendapatkan nilai piksel RGB.
                r, g, b, a = QtGui.QColor(self.image.pixel(x, y)).getRgb()
                # Mengatur kecerahan (brightness)
                adjusted_r = min(max(r + brightness, 0), 255)
                adjusted_g = min(max(g + brightness, 0), 255)
                adjusted_b = min(max(b + brightness, 0), 255)
                # Mengatur kontras (contrast)
                adjusted_r = min(max(((adjusted_r - 127) * contrast) + 127, 0), 255)
                adjusted_g = min(max(((adjusted_g - 127) * contrast) + 127, 0), 255)
                adjusted_b = min(max(((adjusted_b - 127) * contrast) + 127, 0), 255)

                # Set the adjusted color values in the numpy array
                adjusted_image[y][x] = [adjusted_r, adjusted_g, adjusted_b, a]

        # Membuat QImage dari array numpy yang sudah disesuaikan.
        adjusted_qimage = QtGui.QImage(adjusted_image.data, width, height, width * 4, QtGui.QImage.Format_RGBA8888)

        # Membuat QPixmap dari QImage dan menetapkannya ke self.label_2.
        adjusted_pixmap = QtGui.QPixmap.fromImage(adjusted_qimage)
        self.label_2.setPixmap(adjusted_pixmap)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.image = adjusted_qimage

    def open_brightness_contrast_dialog(self): 
        # Membuka dialog untuk mendapatkan input kecerahan dan kontras dari pengguna.
        brightness, ok1 = QtWidgets.QInputDialog.getInt(None, "Brightness", "Enter brightness (-255 to 255):", 0, -255, 255)
        contrast, ok2 = QtWidgets.QInputDialog.getDouble(None, "Contrast", "Enter contrast (0.01 to 4.0):", 1.0, 0.01, 4.0)

        if ok1 and ok2:
            # Mengaplikasikan penyesuaian kecerahan dan kontras
            self.apply_brightness_contrast(brightness, contrast)
            
    def flip_horizontal(self):
            # mendapatkan ukuran gambar
            width = self.image.width()
            height = self.image.height()

            # Membuat sebuah QImage baru untuk menyimpan gambar yang sudah di-flip horizontal
            flipped_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGBA8888)

            # Loop melalui setiap piksel di gambar asli
            for y in range(height):
                for x in range(width):
                     # Mendapatkan warna piksel dari gambar asli
                    pixel_color = QtGui.QColor(self.image.pixel(x, y))

                    # Menyimpan warna piksel di gambar baru, tetapi di posisi yang sudah di-flip horizontal
                    flipped_image.setPixelColor(width - 1 - x, y, pixel_color)       
            
             # Membuat QPixmap dari QImage yang sudah di-flip dan menampilkannya di label_2
            flipped_pixmap = QtGui.QPixmap.fromImage(flipped_image)
            self.label_2.setPixmap(flipped_pixmap)
            # Mengatur alignment dari label_2 untuk menampilkan gambar di tengah
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
             # Menyimpan gambar yang sudah di-flip ke dalam atribut image
            self.image = flipped_image
            
    def flip_vertical(self):
            # mendapatkan ukuran gambar
            width = self.image.width()
            height = self.image.height()

            # Membuat sebuah QImage baru untuk menyimpan gambar yang sudah di-flip vertikal
            flipped_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGBA8888)

            # Loop melalui setiap piksel di gambar asli
            for y in range(height):
                for x in range(width):
                    # Mendapatkan warna piksel dari gambar asli
                    pixel_color = QtGui.QColor(self.image.pixel(x, y))

                     # Menyimpan warna piksel di gambar baru, tetapi di posisi yang sudah di-flip vertikal
                    flipped_image.setPixelColor(x, height - 1 - y, pixel_color) 
            # Membuat QPixmap dari QImage yang sudah di-flip dan menampilkannya di label_2
            flipped_pixmap = QtGui.QPixmap.fromImage(flipped_image)
            self.label_2.setPixmap(flipped_pixmap)
            # Mengatur alignment dari label_2 untuk menampilkan gambar di tengah    
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
            # Menyimpan gambar yang sudah di-flip ke dalam atribut image
            self.image = flipped_image     

    def HistogramEqualization(self):
        # Memeriksa apakah self.label ada (terdapat gambar yang akan diolah).
        if self.label:
            # mendapatkan ukuran gambar
            width = self.image.width()
            height = self.image.height()

            # Membuat objek QImage untuk gambar hasil dengan format RGB32.
            equalized_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Menghitung histogram untuk gambar asli
            histogram = [0] * 256
            total_pixels = width * height

            for x in range(width):
                for y in range(height):
                    pixel_color = QtGui.QColor(*self.image.pixelColor(x, y).getRgb())
                    intensity = pixel_color.red()  # Asumsi bahwa gambar adalah grayscale.

                    histogram[intensity] += 1

            # Menghitung distribusi kumulatif histogram
            cumulative_distribution = [0] * 256
            cumulative_distribution[0] = histogram[0] / total_pixels

            for i in range(1, 256):
                cumulative_distribution[i] = cumulative_distribution[i - 1] + histogram[i] / total_pixels

            # Menyesuaikan nilai pixel pada gambar hasil berdasarkan distribusi kumulatif.
            for x in range(width):
                for y in range(height):
                    pixel_color = QtGui.QColor(*self.image.pixelColor(x, y).getRgb())
                    intensity = pixel_color.red()  # Asumsi bahwa gambar adalah greyscale
                    
                    # Menggunakan distribusi kumulatif untuk menghitung nilai pixel baru.
                    new_intensity = int(255 * cumulative_distribution[intensity])
                    new_color = QtGui.QColor(new_intensity, new_intensity, new_intensity)
                    equalized_image.setPixelColor(x, y, new_color)
            

            # Mengubah gambar hasil (equalized_image) menjadi QPixmap.
            output_pixmap = QtGui.QPixmap.fromImage(equalized_image)
            # Menampilkan gambar hasil di label self.label_2.
            self.label_2.setPixmap(output_pixmap)  
            # Menyimpan gambar hasil dalam variabel self.image.
            self.image = equalized_image

    def Fuzzy_HE_RGB(self):
        # Memeriksa apakah label (mungkin digunakan untuk menampilkan gambar) tidak bernilai None.
        if self.label is not None:
            # mendapatkan ukuran gambar
            width = self.image.width()
            height = self.image.height()

            # Mengambil data piksel dari gambar input
            input_data = np.zeros((height, width, 3), dtype=np.uint8)
            # Mengambil data piksel dari gambar input dan menyimpannya dalam input_data.
            for y in range(height):
                for x in range(width):
                    color = QtGui.QColor(self.image.pixel(x, y))
                    input_data[y, x, 0] = color.red()
                    input_data[y, x, 1] = color.green()
                    input_data[y, x, 2] = color.blue()

            # Menerapkan rumus Fuzzy HE RGB
            output_data = np.zeros_like(input_data)
            for i in range(3):  # Loop untuk saluran warna (R, G, B)
                for y in range(height):
                    for x in range(width):
                        val = input_data[y, x, i]
                        if val < 128:
                            output_data[y, x, i] = int(2 * val ** 2 / 255.0)
                        else:
                            output_data[y, x, i] = int(255 - 2 * (255 - val) ** 2 / 255.0)

            # Membuat gambar output dan menampilkannya di pbOutput
            output_image = QtGui.QImage(output_data.data, width, height, width * 3, QtGui.QImage.Format_RGB888)
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(output_image))
            self.image = output_image
    
    def fuzzy_greyscale(self):
        # Memeriksa apakah label_2 sudah terdefinisi atau tidak.
        if self.label_2 is not None:
            # mendapatkan ukuran gambar
            width = self.image.width()
            height = self.image.height()

            # Mengambil data piksel dari gambar input
            input_data = np.zeros((height, width), dtype=np.uint8)
            # Loop melalui setiap piksel gambar input untuk menghitung nilai grayscale menggunakan rumus Fuzzy.
            for y in range(height):
                for x in range(width):
                    color = QtGui.QColor(self.image.pixel(x, y))
                    # Menghitung nilai greyscale menggunakan rumus Fuzzy
                    val = int(0.3 * color.red() + 0.59 * color.green() + 0.11 * color.blue())
                    input_data[y, x] = val

            # Membuat gambar output dengan format Grayscale8 menggunakan data yang telah dihitung.
            output_image = QtGui.QImage(input_data.data, width, height, width, QtGui.QImage.Format_Grayscale8)
            # Menyetel pixmap label_2 dengan gambar output yang telah dibuat.
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(output_image))
            # Menyimpan gambar output sebagai atribut self.image untuk penggunaan selanjutnya.
            self.image = output_image

    def input_histogram(self):
            # Mendapatkan citra dari label dan mengonversinya ke objek QImage.
            inputan = self.label.pixmap()
            input_image = inputan.toImage()
            # Mengambil lebar dan tinggi gambar input
            width = input_image.width()
            height = input_image.height()

            # Membuat matriks kosong dengan ukuran yang sesuai untuk data piksel gambar input.
            input_data = np.zeros((height, width), dtype=np.uint8)
             # Loop melalui setiap piksel gambar input dan mengubahnya menjadi citra grayscale.
            for y in range(height):
                for x in range(width):
                    color = QtGui.QColor(input_image.pixel(x, y))
                    # Menghitung nilai grayscale menggunakan formula luminance.
                    val = int(0.3 * color.red() + 0.59 * color.green() + 0.11 * color.blue())
                    input_data[y, x] = val

            # Menghitung histogram dari citra grayscale.
            histogram, bins = np.histogram(input_data, bins=256, range=(0, 256))

            # Menampilkan grafik histogram
            plt.figure(figsize=(8, 6))
            plt.bar(bins[:-1], histogram, width=1, color='blue')
            plt.title('Histogram Input')
            plt.xlabel('Intensitas Piksel')
            plt.ylabel('Frekuensi')
            plt.show()

    def Output_histogram(self):
            # Mendapatkan citra dari label dan mengonversinya ke objek QImage.
         if self.label_2 is not None:
            inputan = self.label_2.pixmap()
            input_image = inputan.toImage()
            # Mengambil lebar dan tinggi gambar input
            width = input_image.width()
            height = input_image.height()

            # Membuat matriks kosong dengan ukuran yang sesuai untuk data piksel gambar input.
            input_data = np.zeros((height, width), dtype=np.uint8)
             # Loop melalui setiap piksel gambar input dan mengubahnya menjadi citra grayscale.
            for y in range(height):
                for x in range(width):
                    color = QtGui.QColor(input_image.pixel(x, y))
                    # Menghitung nilai grayscale menggunakan formula luminance.
                    val = int(0.3 * color.red() + 0.59 * color.green() + 0.11 * color.blue())
                    input_data[y, x] = val

            # Menghitung histogram dari citra grayscale.
            histogram, bins = np.histogram(input_data, bins=256, range=(0, 256))

            # Menampilkan grafik histogram
            plt.figure(figsize=(8, 6))
            plt.bar(bins[:-1], histogram, width=1, color='blue')
            plt.title('Histogram Output')
            plt.xlabel('Intensitas Piksel')
            plt.ylabel('Frekuensi')
            plt.show()

    def Kuning(self):
        if self.label is not None:
            image = self.label.pixmap()
            inputan_image = image.toImage()

            for x in range(image.width()):
                for y in range(image.height()):
                    pixel = QtGui.QColor(inputan_image.pixel(x, y))

                    # Add yellow color to the pixel
                    r = min(pixel.red() + 100, 255)
                    g = min(pixel.green() + 100, 255)

                    # Set the new pixel color
                    inputan_image.setPixelColor(x, y, QtGui.QColor(r, g, pixel.blue()))

            self.output_image = QtGui.QPixmap.fromImage(inputan_image)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def Orange(self):
        if self.label is not None:
            image = self.label.pixmap()
            inputan_image = image.toImage()

            for x in range(image.width()):
                for y in range(image.height()):
                    pixel = QtGui.QColor(inputan_image.pixel(x, y))
                    

                    # Sample orange filter: increase red, decrease blue
                    r = min(pixel.red() + 100, 255)
                    g = min(pixel.green() + 50, 255)
                    b = min(pixel.blue() - 50, 255)

                    # Set the new pixel color
                    inputan_image.setPixelColor(x, y, QtGui.QColor(r, g, b))

            self.output_image = QtGui.QPixmap.fromImage(inputan_image)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def Cyan(self):
        if self.label is not None:
            image = self.label.pixmap()
            inputan_image = image.toImage()

            for x in range(image.width()):
                for y in range(image.height()):
                    pixel = QtGui.QColor(inputan_image.pixel(x, y))

                # Set red and green components to 0, leave blue component unchanged
                    r = 0
                    g = pixel.green()
                    b = pixel.blue()

                # Set the new pixel color
                    inputan_image.setPixelColor(x, y, QtGui.QColor(r, g, b))

            self.output_image = QtGui.QPixmap.fromImage(inputan_image)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def Purple(self):
        if self.label is not None:
            image = self.label.pixmap()
            inputan_image = image.toImage()

            for x in range(image.width()):
                for y in range(image.height()):
                    pixel = QtGui.QColor(inputan_image.pixel(x, y))

                # Add purple color to the pixel
                    r = min(pixel.red() + 100, 255)  # Ubah komponen merah (R)
                    g = 0  # Atur komponen hijau (G) menjadi 0 untuk membuat warna ungu
                    b = min(pixel.blue() + 100, 255)  # Ubah komponen biru (B)

                # Setel warna piksel yang baru
                    inputan_image.setPixelColor(x, y, QtGui.QColor(r, g, b))

            self.output_image = QtGui.QPixmap.fromImage(inputan_image)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def Gray(self):
        if self.label is not None:
            image = self.label.pixmap()
            inputan_image = image.toImage()

            for x in range(image.width()):
                for y in range(image.height()):
                    pixel = QtGui.QColor(inputan_image.pixel(x, y))

                    # Calculate grayscale value using the average of R, G, and B values
                    grayscale_value = (pixel.red() + pixel.green() + pixel.blue()) // 3

                    # Set the new pixel color as grayscale
                    inputan_image.setPixelColor(x, y, QtGui.QColor(grayscale_value, grayscale_value, grayscale_value))

            self.output_image = QtGui.QPixmap.fromImage(inputan_image)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)
    def Coklat(self):
        if self.label is not None:
            image = self.label.pixmap()
            inputan_image = image.toImage()

            for x in range(image.width()):
                for y in range(image.height()):
                    pixel = QtGui.QColor(inputan_image.pixel(x, y))

                    # Ubah ke warna coklat (R lebih tinggi, G lebih rendah, B rendah)
                    r = min(pixel.red() + 50, 255)  # Ubah sesuai kebutuhan
                    g = max(pixel.green() - 50, 0)   # Ubah sesuai kebutuhan
                    b = max(pixel.blue() - 100, 0)  # Ubah sesuai kebutuhan

                    # Setel warna piksel baru
                    inputan_image.setPixelColor(x, y, QtGui.QColor(r, g, b))

            self.output_image = QtGui.QPixmap.fromImage(inputan_image)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def Merah(self):
        if self.label is not None:
            image = self.label.pixmap()
            inputan_image = image.toImage()

            for x in range(image.width()):
                for y in range(image.height()):
                    pixel = QtGui.QColor(inputan_image.pixel(x, y))

                    # Set red color to the pixel (R=255, G=0, B=0)
                    r = pixel.red()
                    g = 0
                    b = 0

                    # Set the new pixel color
                    inputan_image.setPixelColor(x, y, QtGui.QColor(r, g, b))

            self.output_image = QtGui.QPixmap.fromImage(inputan_image)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def prewittEdgeDetection(self):
        if self.label:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.label.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            for x in range(1, width - 1):
                for y in range(1, height - 1):
                    gx = 0
                    gy = 0

                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            pixel_color = QtGui.QColor(input_qimage.pixel(x + dx, y + dy))
                            intensity = pixel_color.red()  # Using red channel for grayscale

                            # Prewitt masks
                            gx += dx * intensity
                            gy += dy * intensity

                    edge_intensity = min(int(abs(gx) + abs(gy)), 255)
                    output_qimage.setPixelColor(x, y, QtGui.QColor(edge_intensity, edge_intensity, edge_intensity))

            # Convert QImage to QPixmap for display
            self.output_image = QtGui.QPixmap.fromImage(output_qimage)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def sobelEdgeDetection(self):
        if self.label is not None:
            image = self.label.pixmap().toImage()
            width = image.width()
            height = image.height()

            kernel = [[0, -1, -2],
                    [1, 0, -1],
                    [2, 1, 0]]

            for i in range(width):
                for j in range(height):
                    cRed, cGreen, cBlue = 0, 0, 0
                    colorP = [cRed, cGreen, cBlue]

                    for k in range(3):
                        for l in range(3):
                            a = i + k + 1
                            b = j + l + 1

                            if 0 <= a < width and 0 <= b < height:
                                pixel_value = image.pixelColor(a, b)
                                colorP[0] += ((kernel[k][l]) * pixel_value.red())
                                colorP[1] += ((kernel[k][l]) * pixel_value.green())
                                colorP[2] += ((kernel[k][l]) * pixel_value.blue())

                    colorP = [max(0, min(int(val), 255)) for val in colorP]

                    new_color = QtGui.QColor(colorP[0], colorP[1], colorP[2])
                    image.setPixelColor(i, j, new_color)

            outputSobel = QtGui.QPixmap.fromImage(image)
            self.label_2.setPixmap(outputSobel)
            self.label_2.setScaledContents(True)

    def applyRobertFilter(self):
        if self.label:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.label.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Define Robert Filter kernels for two directions
            kernel_x = np.array([[-1, 0], [0, 1]])
            kernel_y = np.array([[0, -1], [1, 0]])

            for x in range(1, width - 1):
                for y in range(1, height - 1):
                    sum_x = sum_y = 0

                    for i in range(2):
                        for j in range(2):
                            pixel_color = QtGui.QColor(input_qimage.pixel(x - 1 + i, y - 1 + j))
                            gray = pixel_color.red()  # Assuming it's a grayscale image

                            sum_x += gray * kernel_x[i][j]
                            sum_y += gray * kernel_y[i][j]

                    # Calculate the gradient magnitude
                    magnitude = int(np.sqrt(sum_x**2 + sum_y**2))
                    output_qimage.setPixelColor(x, y, QtGui.QColor(magnitude, magnitude, magnitude))

            # Convert QImage to QPixmap for display
            self.output_image = QtGui.QPixmap.fromImage(output_qimage)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def applyLowpassFilter(self):
        if self.label:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.label.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            for x in range(width):
                for y in range(height):
                    # Apply the lowpass filter by averaging pixel values in the neighborhood
                    r_sum = 0
                    g_sum = 0
                    b_sum = 0
                    num_neighbors = 0

                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx = x + dx
                            ny = y + dy

                            if 0 <= nx < width and 0 <= ny < height:
                                pixel_color = QtGui.QColor(input_qimage.pixel(nx, ny))
                                r, g, b, _ = pixel_color.getRgb()
                                r_sum += r
                                g_sum += g
                                b_sum += b
                                num_neighbors += 1

                    # Calculate the average value
                    if num_neighbors > 0:
                        r_avg = r_sum // num_neighbors
                        g_avg = g_sum // num_neighbors
                        b_avg = b_sum // num_neighbors
                        output_qimage.setPixel(x, y, QtGui.qRgb(r_avg, g_avg, b_avg))

            # Convert QImage to QPixmap for display
            self.output_image = QtGui.QPixmap.fromImage(output_qimage)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def applyHighPassFilter(self):
        if self.label:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.label.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            for x in range(1, width - 1):
                for y in range(1, height - 1):
                    # Get the colors of the surrounding pixels
                    pixel_center = QtGui.QColor(input_qimage.pixel(x, y))
                    pixel_left = QtGui.QColor(input_qimage.pixel(x - 1, y))
                    pixel_right = QtGui.QColor(input_qimage.pixel(x + 1, y))
                    pixel_up = QtGui.QColor(input_qimage.pixel(x, y - 1))
                    pixel_down = QtGui.QColor(input_qimage.pixel(x, y + 1))

                    # Calculate the high-pass filtered pixel value
                    r = 5 * pixel_center.red() - pixel_left.red() - pixel_right.red() - pixel_up.red() - pixel_down.red()
                    g = 5 * pixel_center.green() - pixel_left.green() - pixel_right.green() - pixel_up.green() - pixel_down.green()
                    b = 5 * pixel_center.blue() - pixel_left.blue() - pixel_right.blue() - pixel_up.blue() - pixel_down.blue()

                    # Clip the values to the valid color range (0-255)
                    r = max(0, min(255, r))
                    g = max(0, min(255, g))
                    b = max(0, min(255, b))

                    # Set the new pixel color
                    output_qimage.setPixel(x, y, QtGui.qRgb(r, g, b))

            # Convert QImage to QPixmap for display
            self.output_image = QtGui.QPixmap.fromImage(output_qimage)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def sharpenImage(self):
        if self.label:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.label.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Create a sharpening kernel (you can adjust the kernel values)
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])

            # Apply the convolution operation to sharpen the image
            for x in range(1, width - 1):
                for y in range(1, height - 1):
                    new_color = [0, 0, 0]

                    for dx in range(3):
                        for dy in range(3):
                            pixel_color = QtGui.QColor(input_qimage.pixel(x + dx - 1, y + dy - 1))
                            new_color[0] += pixel_color.red() * kernel[dx][dy]
                            new_color[1] += pixel_color.green() * kernel[dx][dy]
                            new_color[2] += pixel_color.blue() * kernel[dx][dy]

                    for i in range(3):
                        new_color[i] = min(max(new_color[i], 0), 255)

                    output_qimage.setPixelColor(x, y, QtGui.QColor(new_color[0], new_color[1], new_color[2]))

            # Convert QImage to QPixmap for display
            self.output_image = QtGui.QPixmap.fromImage(output_qimage)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def erodeSquare(self, size):
        if self.label:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.label.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Define the erosion element (square or cross)
            element = np.ones((size, size), dtype=np.uint8)

            for x in range(width):
                for y in range(height):
                    pixel_color = QtGui.QColor(input_qimage.pixel(x, y))
                    r, g, b, _ = pixel_color.getRgb()

                    # Apply erosion operation
                    if r == 0 and g == 0 and b == 0:
                        neighborhood = np.zeros((size, size), dtype=np.uint8)

                        # Check if the neighborhood matches the element
                        if np.array_equal(neighborhood, element):
                            r = 255
                            g = 255
                            b = 255

                    # Set the new pixel color
                    output_qimage.setPixel(x, y, QtGui.qRgb(r, g, b))

            # Convert QImage to QPixmap for display
            self.output_image = QtGui.QPixmap.fromImage(output_qimage)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def erodeSquare3(self):
        self.erodeSquare(3)

    def erodeSquare5(self):
        self.erodeSquare(5)

    def erodeCross3(self):
        if self.label:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.label.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Define the 3x3 cross element
            element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

            for x in range(width):
                for y in range(height):
                    pixel_color = QtGui.QColor(input_qimage.pixel(x, y))
                    r, g, b, _ = pixel_color.getRgb()

                    # Apply erosion operation
                    if r == 0 and g == 0 and b == 0:
                        neighborhood = np.zeros((3, 3), dtype=np.uint8)

                        for i in range(-1, 2):
                            for j in range(-1, 2):
                                if 0 <= x + i < width and 0 <= y + j < height:
                                    if element[i + 1][j + 1] == 1:
                                        neighbor_color = QtGui.QColor(input_qimage.pixel(x + i, y + j))
                                        neighbor_r, neighbor_g, neighbor_b, _ = neighbor_color.getRgb()

                                        if neighbor_r == 0 and neighbor_g == 0 and neighbor_b == 0:
                                            neighborhood[i + 1][j + 1] = 1

                        # Check if the neighborhood matches the element
                        if np.array_equal(neighborhood, element):
                            r = 255
                            g = 255
                            b = 255

                    # Set the new pixel color
                    output_qimage.setPixel(x, y, QtGui.qRgb(r, g, b))

            # Convert QImage to QPixmap for display
            self.output_image = QtGui.QPixmap.fromImage(output_qimage)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def dilateSquare(self, size):
        if self.label:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.label.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Define the dilation element (square or cross)
            element = np.ones((size, size), dtype=np.uint8)

            for x in range(width):
                for y in range(height):
                    pixel_color = QtGui.QColor(input_qimage.pixel(x, y))
                    r, g, b, _ = pixel_color.getRgb()

                    # Apply dilation operation
                    if r == 255 and g == 255 and b == 255:
                        neighborhood = np.ones((size, size), dtype=np.uint8)

                        # Set the new pixel color if any element in the neighborhood matches
                        if np.any(np.logical_and(neighborhood, element)):
                            r = 0
                            g = 0
                            b = 0

                    # Set the new pixel color
                    output_qimage.setPixel(x, y, QtGui.qRgb(r, g, b))

            # Convert QImage to QPixmap for display
            self.output_image = QtGui.QPixmap.fromImage(output_qimage)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def dilateSquare3(self):
        self.dilateSquare(3)

    def dilateSquare5(self):
        self.dilateSquare(5)

    def dilateCross3(self):
        if self.label:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.label.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Define the 3x3 cross element
            element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

            for x in range(width):
                for y in range(height):
                    pixel_color = QtGui.QColor(input_qimage.pixel(x, y))
                    r, g, b, _ = pixel_color.getRgb()

                    # Apply dilation operation
                    if r == 255 and g == 255 and b == 255:
                        neighborhood = np.zeros((3, 3), dtype=np.uint8)

                        for i in range(-1, 2):
                            for j in range(-1, 2):
                                if 0 <= x + i < width and 0 <= y + j < height:
                                    if element[i + 1][j + 1] == 1:
                                        neighbor_color = QtGui.QColor(input_qimage.pixel(x + i, y + j))
                                        neighbor_r, neighbor_g, neighbor_b, _ = neighbor_color.getRgb()

                                        if neighbor_r == 255 and neighbor_g == 255 and neighbor_b == 255:
                                            neighborhood[i + 1][j + 1] = 1

                        # Set the new pixel color if any element in the neighborhood matches
                        if np.any(np.logical_and(neighborhood, element)):
                            r = 0
                            g = 0
                            b = 0

                    # Set the new pixel color
                    output_qimage.setPixel(x, y, QtGui.qRgb(r, g, b))

            # Convert QImage to QPixmap for display
            self.output_image = QtGui.QPixmap.fromImage(output_qimage)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def applyOpening(self):
        if self.label:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.label.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Define the square kernel (9x9)
            kernel_size = 9
            kernel_radius = kernel_size // 2

            # Apply Opening Operation
            for x in range(kernel_radius, width - kernel_radius):
                for y in range(kernel_radius, height - kernel_radius):
                    min_r, min_g, min_b = 255, 255, 255
                    max_r, max_g, max_b = 0, 0, 0

                    for i in range(-kernel_radius, kernel_radius + 1):
                        for j in range(-kernel_radius, kernel_radius + 1):
                            pixel_color = QtGui.QColor(input_qimage.pixel(x + i, y + j))
                            min_r = min(min_r, pixel_color.red())
                            min_g = min(min_g, pixel_color.green())
                            min_b = min(min_b, pixel_color.blue())
                            max_r = max(max_r, pixel_color.red())
                            max_g = max(max_g, pixel_color.green())
                            max_b = max(max_b, pixel_color.blue())

                    # Set the new pixel color
                    output_qimage.setPixelColor(x, y, QtGui.QColor(max_r, max_g, max_b))

            # Convert QImage to QPixmap for display
            self.output_image = QtGui.QPixmap.fromImage(output_qimage)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def applyClosingOperation(self):
        if self.label:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.label.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Define a 9x9 square kernel for Closing Operation
            kernel = [[1] * 9 for _ in range(9)]

            # Apply Closing Operation
            for x in range(width):
                for y in range(height):
                    r_sum, g_sum, b_sum = 0, 0, 0
                    for i in range(9):
                        for j in range(9):
                            xi, yj = x + i - 4, y + j - 4
                            if 0 <= xi < width and 0 <= yj < height:
                                pixel_color = QtGui.QColor(input_qimage.pixel(xi, yj))
                                r_sum += pixel_color.red() * kernel[i][j]
                                g_sum += pixel_color.green() * kernel[i][j]
                                b_sum += pixel_color.blue() * kernel[i][j]

                    # Set the new pixel color
                    output_qimage.setPixelColor(x, y, QtGui.QColor(min(r_sum, 255), min(g_sum, 255), min(b_sum, 255)))

            # Convert QImage to QPixmap for display
            self.output_image = QtGui.QPixmap.fromImage(output_qimage)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def applyGaussianBlur3(self):
        if self.label:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.label.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Gaussian kernel
            kernel = np.array([[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]])

            kernel = kernel / 16.0  # Normalize the kernel

            for x in range(1, width - 1):
                for y in range(1, height - 1):
                    r, g, b, _ = input_qimage.pixelColor(x, y).getRgb()

                    r_sum = 0
                    g_sum = 0
                    b_sum = 0

                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            neighbor_color = input_qimage.pixelColor(x + i, y + j)
                            neighbor_r, neighbor_g, neighbor_b, _ = neighbor_color.getRgb()

                            r_sum += neighbor_r * kernel[i + 1][j + 1]
                            g_sum += neighbor_g * kernel[i + 1][j + 1]
                            b_sum += neighbor_b * kernel[i + 1][j + 1]

                    # Set the new pixel color
                    output_qimage.setPixelColor(x, y, QtGui.QColor(int(r_sum), int(g_sum), int(b_sum)))

            self.output_image = QtGui.QPixmap.fromImage(output_qimage)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def applyGaussianBlur5(self):
        if self.label:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.label.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Define Gaussian Kernel (5x5)
            kernel = np.array([
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1]
            ]) / 256.0  # Normalize the kernel

            # Apply Gaussian Blur
            for x in range(2, width - 2):
                for y in range(2, height - 2):
                    r, g, b = 0, 0, 0
                    for i in range(-2, 3):
                        for j in range(-2, 3):
                            pixel_color = QtGui.QColor(input_qimage.pixel(x + i, y + j))
                            r += pixel_color.red() * kernel[i + 2][j + 2]
                            g += pixel_color.green() * kernel[i + 2][j + 2]
                            b += pixel_color.blue() * kernel[i + 2][j + 2]

                    # Set the new pixel color
                    output_qimage.setPixelColor(x, y, QtGui.QColor(int(r), int(g), int(b)))

            # Convert QImage to QPixmap for display
            self.output_image = QtGui.QPixmap.fromImage(output_qimage)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def applyUnsharpMasking(self):
        if self.label:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.label.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Convert input image to grayscale
            grayscale_image = input_qimage.convertToFormat(QtGui.QImage.Format_Grayscale8)

            # Create a copy of the grayscale image
            sharpened_image = grayscale_image.copy()

            # Define Unsharp Masking parameters
            alpha = 1.5  # Adjust this value as needed
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])

            # Apply Unsharp Masking
            for x in range(1, width - 1):
                for y in range(1, height - 1):
                    value = 0
                    for i in range(3):
                        for j in range(3):
                            pixel_value = grayscale_image.pixelColor(x + i - 1, y + j - 1).value()
                            kernel_value = kernel[i][j]
                            value += pixel_value * kernel_value
                    value = min(255, max(0, value))
                    value = int(value)
                    sharpened_image.setPixel(x,y,value)

            # Combine sharpened image with original
            for x in range(width):
                for y in range(height):
                    orig_value = input_qimage.pixelColor(x, y).value()
                    sharpened_value = sharpened_image.pixelColor(x, y).value()
                    new_value = min(255, max(0, orig_value + alpha * (orig_value - sharpened_value)))
                    new_value = int(new_value)
                    output_qimage.setPixelColor(x, y, QtGui.QColor(new_value, new_value, new_value))

            # Convert QImage to QPixmap for display
            self.output_image = QtGui.QPixmap.fromImage(output_qimage)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)  

    def extractfeaturesRGB(self):
        if self.label:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.label.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Initialize color feature counters
            red_pixels = 0
            green_pixels = 0
            blue_pixels = 0

            # Iterate through the image pixels to extract color features
            for x in range(width):
                for y in range(height):
                    pixel_color = QtGui.QColor(input_qimage.pixel(x, y))
                    r, g, b, _ = pixel_color.getRgb()

                    # Count pixels in each color channel
                    red_pixels += r
                    green_pixels += g
                    blue_pixels += b

            # Calculate the average color values
            total_pixels = width * height
            avg_red = red_pixels / total_pixels
            avg_green = green_pixels / total_pixels
            avg_blue = blue_pixels / total_pixels

            # Display color feature extraction result
            result_text = f'Average Red: {avg_red:.2f}\nAverage Green: {avg_green:.2f}\nAverage Blue: {avg_blue:.2f}'
            self.label_2.setText(result_text)
            self.label_2.setScaledContents(True)

    def removeBackground(self):
        if self.label:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.label.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Define the background color to be removed (e.g., green)
            bg_color = QtGui.QColor(0, 255, 0)  # Green

            # Threshold for color similarity
            color_threshold = 100

            for x in range(width):
                for y in range(height):
                    pixel_color = QtGui.QColor(input_qimage.pixel(x, y))

                    # Calculate color difference
                    color_diff = abs(pixel_color.red() - bg_color.red()) + \
                                 abs(pixel_color.green() - bg_color.green()) + \
                                 abs(pixel_color.blue() - bg_color.blue())

                    # Check if pixel is similar to the background color
                    if color_diff < color_threshold:
                        # Set the pixel to white (background removal)
                        output_qimage.setPixelColor(x, y, QtGui.QColor(QtGui.Qt.white))
                    else:
                        # Keep the original pixel color
                        output_qimage.setPixelColor(x, y, pixel_color)

            # Convert QImage to QPixmap for display
            self.output_image = QtGui.QPixmap.fromImage(output_qimage)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def segmentImage(self):
        if self.label:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.label.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            for x in range(1, width - 1):
                for y in range(1, height - 1):
                    gx = 0
                    gy = 0

                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            pixel_color = QtGui.QColor(input_qimage.pixel(x + dx, y + dy))
                            intensity = pixel_color.red()  # Using red channel for grayscale

                            # Sobel masks
                            if dx == -1:
                                gx -= intensity
                            elif dx == 1:
                                gx += intensity
                            if dy == -1:
                                gy -= intensity
                            elif dy == 1:
                                gy += intensity

                    edge_intensity = min(int(abs(gx) + abs(gy)), 255)
                    output_qimage.setPixelColor(x, y, QtGui.QColor(edge_intensity, edge_intensity, edge_intensity))

            # Convert QImage to QPixmap for display
            self.output_image = QtGui.QPixmap.fromImage(output_qimage)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    # def cropImage(self):
    #     if self.label and self.crop_start and self.crop_end:
    #         # Create a QRect for the cropping region
    #         crop_rect = QtGui.QRect(self.crop_start, self.crop_end)

    #         # Convert QPixmap to QImage for image processing
    #         input_qimage = self.label.pixmap().toImage()
    #         width = input_qimage.width()
    #         height = input_qimage.height()

    #         # Create a new QImage for the cropped image
    #         output_qimage = input_qimage.copy(crop_rect)

    #         # Convert QImage to QPixmap for display
    #         self.output_image = QtGui.QPixmap.fromImage(output_qimage)
    #         self.label_2.setPixmap(self.output_image)
    #         self.label_2.setScaledContents(True)

    # def mousePressEvent(self, event):
    #     if self.label:
    #         self.crop_start = event.pos()
    #         self.cropping = True

    # def mouseMoveEvent(self, event):
    #     if self.cropping:
    #         self.crop_end = event.pos()
    #         self.update()

    # def mouseReleaseEvent(self, event):
    #     if self.cropping:
    #         self.crop_end = event.pos()
    #         self.cropping = False
    #         self.cropImage()
    #         self.update()
    
    def Thresholding(self):
        if self.label:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.label.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Set threshold value (adjust as needed)
            threshold_value = 128

            # Apply Thresholding
            for x in range(width):
                for y in range(height):
                    pixel_color = QtGui.QColor(input_qimage.pixel(x, y))
                    # Calculate grayscale value (average of R, G, B)
                    grayscale_value = (pixel_color.red() + pixel_color.green() + pixel_color.blue()) // 3

                    # Apply thresholding
                    if grayscale_value < threshold_value:
                        output_qimage.setPixelColor(x, y, QtGui.QColor(0, 0, 0))  # Set pixel to black
                    else:
                        output_qimage.setPixelColor(x, y, QtGui.QColor(255, 255, 255))  # Set pixel to white

            # Convert QImage to QPixmap for display
            self.output_image = QtGui.QPixmap.fromImage(output_qimage)
            self.label_2.setPixmap(self.output_image)
            self.label_2.setScaledContents(True)

    def Identity(self):
        if self.label:
        # Ambil gambar dari QLabel (self.label)
            input_image = self.label.pixmap().toImage()
        
        # Buat salinan dari gambar input
            output_image = QtGui.QPixmap.fromImage(input_image)
        
        # Tampilkan gambar hasil pada QLabel lainnya (self.label_2)
            self.label_2.setPixmap(output_image)
            self.label_2.setScaledContents(True)



    def bitdepth(self, bit):
            image = self.label.pixmap().toImage()
            level = 255 / (2 ** bit - 1)
            
            for i in range(image.width()):
                for j in range(image.height()):
                    color = QtGui.QColor(image.pixel(i, j))
                    R = int(round(color.red() / level) * level)
                    G = int(round(color.green() / level) * level)
                    B = int(round(color.blue() / level) * level)
                    image.setPixel(i, j, QtGui.QColor(R, G, B).rgb())

            siap = QtGui.QPixmap.fromImage(image)
            self.label_2.setPixmap(siap)
            self.label_2.setScaledContents(True)

   

   

    def Bit1(self):
        jumlah_bit = 1
        self.bitdepth(jumlah_bit)

    def Bit2(self):
        jumlah_bit = 2
        self.bitdepth(jumlah_bit)

    def Bit3(self):
        jumlah_bit = 3
        self.bitdepth(jumlah_bit)
    
    def Bit4(self):
        jumlah_bit = 4
        self.bitdepth(jumlah_bit)
    
    def Bit5(self):
        jumlah_bit = 5
        self.bitdepth(jumlah_bit)
    
    def Bit6(self):
        jumlah_bit = 6
        self.bitdepth(jumlah_bit)

    def Bit7(self):
        jumlah_bit = 7
        self.bitdepth(jumlah_bit)
    
    def Bit8(self):
        jumlah_bit = 8
        self.bitdepth(jumlah_bit)

             
    def exitApplication(self):
        # untuk keluar dari aplikasi
        QtWidgets.qApp.quit()    
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Process()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
