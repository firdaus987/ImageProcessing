from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QScrollArea, QPushButton, QAction
from aritmatika import Ui_MainWindow as latihan # Mengimpor kelas Ui_MainWindow dari modul aritmatika dan memberinya alias latihan.
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt


class Ui_MainWindow(object):
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
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 300, 241, 16))
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(260, 300, 241, 16))
        self.label_4.setFrameShape(QtWidgets.QFrame.Box)
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
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
        self.actionOutput = QtWidgets.QAction(MainWindow)
        self.actionOutput.setObjectName("actionOutput")
        # self.actionOutput.triggered.connect(self.histogram)
        self.actionInput_Output = QtWidgets.QAction(MainWindow)
        self.actionInput_Output.setObjectName("actionInput_Output")
        self.actionBrightness_Contrast = QtWidgets.QAction(MainWindow)
        self.actionBrightness_Contrast.setObjectName("actionBrightness_Contrast")
        self.actionBrightness_Contrast.triggered.connect(self.open_brightness_contrast_dialog)
        self.actionInvers = QtWidgets.QAction(MainWindow)
        self.actionInvers.setObjectName("actionInvers")
        self.actionInvers.triggered.connect(self.Invers)
        self.actionLog_Brightness = QtWidgets.QAction(MainWindow)
        self.actionLog_Brightness.setObjectName("actionLog_Brightness")
        self.actionGamma_Corretion = QtWidgets.QAction(MainWindow)
        self.actionGamma_Corretion.setObjectName("actionGamma_Corretion")
        self.actionKuning = QtWidgets.QAction(MainWindow)
        self.actionKuning.setObjectName("actionKuning")
        self.actionOrange = QtWidgets.QAction(MainWindow)
        self.actionOrange.setObjectName("actionOrange")
        self.actionCyan = QtWidgets.QAction(MainWindow)
        self.actionCyan.setObjectName("actionCyan")
        self.actionPurple = QtWidgets.QAction(MainWindow)
        self.actionPurple.setObjectName("actionPurple")
        self.actionGray = QtWidgets.QAction(MainWindow)
        self.actionGray.setObjectName("actionGray")
        self.actionCoklat = QtWidgets.QAction(MainWindow)
        self.actionCoklat.setObjectName("actionCoklat")
        self.actionMerah = QtWidgets.QAction(MainWindow)
        self.actionMerah.setObjectName("actionMerah")
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
        self.action2_bit = QtWidgets.QAction(MainWindow)
        self.action2_bit.setObjectName("action2_bit")
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
        self.actionHistogram_Equalization.triggered.connect(self.HistogramEqualization)
        self.actionFuzzy_HE_RGB = QtWidgets.QAction(MainWindow)
        self.actionFuzzy_HE_RGB.setObjectName("actionFuzzy_HE_RGB")
        self.actionFuzzy_HE_RGB.triggered.connect(self.Fuzzy_HE_RGB)
        self.actionFuzzy_Grayscale = QtWidgets.QAction(MainWindow)
        self.actionFuzzy_Grayscale.setObjectName("actionFuzzy_Grayscale")
        self.actionFuzzy_Grayscale.triggered.connect(self.fuzzy_greyscale)
        self.actionIdentity = QtWidgets.QAction(MainWindow)
        self.actionIdentity.setObjectName("actionIdentity")
        self.actionSharpen = QtWidgets.QAction(MainWindow)
        self.actionSharpen.setObjectName("actionSharpen")
        self.actionUnsharp_Masking = QtWidgets.QAction(MainWindow)
        self.actionUnsharp_Masking.setObjectName("actionUnsharp_Masking")
        self.actionAvarage_Filter = QtWidgets.QAction(MainWindow)
        self.actionAvarage_Filter.setObjectName("actionAvarage_Filter")
        self.actionLow_Pass_Filler = QtWidgets.QAction(MainWindow)
        self.actionLow_Pass_Filler.setObjectName("actionLow_Pass_Filler")
        self.actionHight_Pass_Filter = QtWidgets.QAction(MainWindow)
        self.actionHight_Pass_Filter.setObjectName("actionHight_Pass_Filter")
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
        self.actionGaussian_Blur_5_X_5 = QtWidgets.QAction(MainWindow)
        self.actionGaussian_Blur_5_X_5.setObjectName("actionGaussian_Blur_5_X_5")
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
        self.actionCross = QtWidgets.QAction(MainWindow)
        self.actionCross.setObjectName("actionCross")
        self.actionSquare_9 = QtWidgets.QAction(MainWindow)
        self.actionSquare_9.setObjectName("actionSquare_9")
        self.actionSquare_10 = QtWidgets.QAction(MainWindow)
        self.actionSquare_10.setObjectName("actionSquare_10")
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
        self.menuColors.addAction(self.menuRGB.menuAction())
        self.menuColors.addAction(self.menuRGB_to_Grayscale.menuAction())
        self.menuColors.addAction(self.menuBrightness.menuAction())
        self.menuColors.addAction(self.actionBrightness_Contrast)
        self.menuColors.addAction(self.actionInvers)
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
        self.menuIler.addAction(self.actionAvarage_Filter)
        self.menuIler.addAction(self.actionLow_Pass_Filler)
        self.menuIler.addAction(self.actionHight_Pass_Filter)
        self.menuIler.addAction(self.actionBandstop_Filter)
        self.menuEdge_Detection.addAction(self.actionPrewitt)
        self.menuEdge_Detection.addAction(self.actionSobel)
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
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuColors.menuAction())
        self.menubar.addAction(self.menuTentang.menuAction())
        self.menubar.addAction(self.menuImage_Processing.menuAction())
        self.menubar.addAction(self.menuAritmatical_Operation.menuAction())
        self.menubar.addAction(self.menuIler.menuAction())
        self.menubar.addAction(self.menuEdge_Detection.menuAction())
        self.menubar.addAction(self.menuMorfologi.menuAction())

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
        self.actionInput.setText(_translate("MainWindow", "Input"))
        self.actionOutput.setText(_translate("MainWindow", "Output"))
        self.actionInput_Output.setText(_translate("MainWindow", "Input Output"))
        self.actionBrightness_Contrast.setText(_translate("MainWindow", "Brightness - Contrast"))
        self.actionInvers.setText(_translate("MainWindow", "Invers"))
        self.actionLog_Brightness.setText(_translate("MainWindow", "Log Brightness"))
        self.actionGamma_Corretion.setText(_translate("MainWindow", "Gamma Corretion"))
        self.actionKuning.setText(_translate("MainWindow", "Kuning"))
        self.actionOrange.setText(_translate("MainWindow", "Orange"))
        self.actionCyan.setText(_translate("MainWindow", "Cyan"))
        self.actionPurple.setText(_translate("MainWindow", "Purple"))
        self.actionGray.setText(_translate("MainWindow", "Gray"))
        self.actionCoklat.setText(_translate("MainWindow", "Coklat"))
        self.actionMerah.setText(_translate("MainWindow", "Merah"))
        self.actionAverage.setText(_translate("MainWindow", "Average"))
        self.actionLightness.setText(_translate("MainWindow", "Lightness"))
        self.actionLuminance.setText(_translate("MainWindow", "Luminance"))
        self.actionContrast.setText(_translate("MainWindow", "Contrast"))
        self.action1_bit.setText(_translate("MainWindow", "1 bit"))
        self.action2_bit.setText(_translate("MainWindow", "2 bit"))
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
        self.actionUnsharp_Masking.setText(_translate("MainWindow", "Unsharp Masking"))
        self.actionAvarage_Filter.setText(_translate("MainWindow", "Avarage Filter"))
        self.actionLow_Pass_Filler.setText(_translate("MainWindow", "Low Pass Filter"))
        self.actionHight_Pass_Filter.setText(_translate("MainWindow", "Hight Pass Filter"))
        self.actionBandstop_Filter.setText(_translate("MainWindow", "Bandstop Filter"))
        self.actionEdge_Detection_1.setText(_translate("MainWindow", "Edge Detection 1"))
        self.actionEdge_Detection_2.setText(_translate("MainWindow", "Edge Detection 2"))
        self.actionEdge_Detection_3.setText(_translate("MainWindow", "Edge Detection 3"))
        self.actionGaussian_Blur_3_x_3.setText(_translate("MainWindow", "Gaussian Blur 3 x 3"))
        self.actionGaussian_Blur_5_X_5.setText(_translate("MainWindow", "Gaussian Blur 5 X 5 "))
        self.actionPrewitt.setText(_translate("MainWindow", "Prewitt"))
        self.actionSobel.setText(_translate("MainWindow", "Sobel"))
        self.actionSquare_3.setText(_translate("MainWindow", "Square 3"))
        self.actionSquare_5.setText(_translate("MainWindow", "Square 5"))
        self.actionCross_3.setText(_translate("MainWindow", "Cross 3"))
        self.actionSquare_4.setText(_translate("MainWindow", "Square 3"))
        self.actionSquare_6.setText(_translate("MainWindow", "Square 5"))
        self.actionCross.setText(_translate("MainWindow", "Cross 3"))
        self.actionSquare_9.setText(_translate("MainWindow", "Square 9"))
        self.actionSquare_10.setText(_translate("MainWindow", "Square 9"))
        self.actionOpen_File.setText(_translate("MainWindow", "Open File"))
        self.actionSave_As.setText(_translate("MainWindow", "Save As"))
        self.actionKeluar.setText(_translate("MainWindow", "Keluar"))
        self.actionfliphorizontal.setText(_translate("mainWindow" , "Flip Horizontal"))
        self.actionflipvertical.setText(_translate("MainWindow","Flip Vertical"))
        self.actionRotate.setText(_translate("MainWindow","Rotate"))

    

    def frameArimatika(self): 
        # Definisi fungsi frameArimatika yang merupakan metode dari suatu kelas. Fungsi ini mungkin digunakan untuk membuat dan menampilkan sebuah jendela Qt.
        self.window = QtWidgets.QMainWindow() # Membuat sebuah objek jendela utama dari Qt.
        self.ui = latihan() # Membuat sebuah objek dari kelas latihan.
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
            self.label_4.setText(file_name)
            
    def rotasiGambare(self): # Definisi fungsi rotasiGambare yang mungkin merupakan metode dari suatu kelas.
        rotation , ok = QtWidgets.QInputDialog.getInt(None , "Rotate Image","Enter rotation angle (degress):",0,-360,360) # Meminta pengguna untuk memasukkan sudut rotasi dalam derajat.
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
                                   
                
    def Average(self): # Definisi fungsi Average.
        # Mendapatkan lebar (width) dan tinggi (height) gambar yang sedang diproses.
        width = self.image.width()
        height = self.image.height()
         # Membuat objek QImage baru dengan ukuran yang sama dengan gambar asli
         # dengan format QImage.Format_RGB32.
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
          
    def Luminosity(self): # Fungsi Luminosity untuk mengubah gambar berwarna menjadi gambar grayscale menggunakan rumus luminance.
        # Mendapatkan lebar dan tinggi gambar asli.
        width = self.image.width() 
        height = self.image.height()

        # Membuat objek QImage baru dengan ukuran yang sama dengan gambar asli,
        # dengan format QImage.Format_RGB32 untuk gambar RGBA.
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

        # Dapatkan ukuran gambar
            width = self.image.width()
            height = self.image.height()

            # Membuat salinan gambar untuk diubah
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
        self.label_3.setText("")
        self.label_4.setText("")
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
                self.label_3.setText(file_name)
                         
    def Lightness(self): # Fungsi Lightness untuk mengubah gambar berwarna menjadi gambar grayscale menggunakan rumus lightness.
        # Mendapatkan lebar dan tinggi gambar asli.
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
           
    def apply_brightness_contrast(self, brightness, contrast): # Fungsi apply_brightness_contrast digunakan 
        # untuk mengatur kecerahan dan kontras gambar.
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

    def open_brightness_contrast_dialog(self): # Fungsi open_brightness_contrast_dialog digunakan untuk membuka dialog dan mendapatkan input 
        # dari pengguna untuk kecerahan dan kontras.
        
        # Membuka dialog untuk mendapatkan input kecerahan dan kontras dari pengguna.
        brightness, ok1 = QtWidgets.QInputDialog.getInt(None, "Brightness", "Enter brightness (-255 to 255):", 0, -255, 255)
        contrast, ok2 = QtWidgets.QInputDialog.getDouble(None, "Contrast", "Enter contrast (0.01 to 4.0):", 1.0, 0.01, 4.0)

        if ok1 and ok2:
            # Mengaplikasikan penyesuaian kecerahan dan kontras
            self.apply_brightness_contrast(brightness, contrast)
            
    def flip_horizontal(self):
            width = self.image.width()
            height = self.image.height()

            # Create a numpy array to store the flipped image
            flipped_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGBA8888)

            for y in range(height):
                for x in range(width):
                    pixel_color = QtGui.QColor(self.image.pixel(x, y))
                    flipped_image.setPixelColor(width - 1 - x, y, pixel_color)       

            flipped_pixmap = QtGui.QPixmap.fromImage(flipped_image)
            self.label_2.setPixmap(flipped_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
            self.image = flipped_image
            
    def flip_vertical(self):
            width = self.image.width()
            height = self.image.height()

            # Create a numpy array to store the flipped image
            flipped_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGBA8888)

            for y in range(height):
                for x in range(width):
                    pixel_color = QtGui.QColor(self.image.pixel(x, y))
                    flipped_image.setPixelColor(x, height - 1 - y, pixel_color) 

            flipped_pixmap = QtGui.QPixmap.fromImage(flipped_image)
            self.label_2.setPixmap(flipped_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
            self.image = flipped_image     

    def HistogramEqualization(self):
        if self.label:
            width = self.image.width()
            height = self.image.height()

            equalized_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Menghitung histogram
            histogram = [0] * 256
            total_pixels = width * height

            for x in range(width):
                for y in range(height):
                    pixel_color = QtGui.QColor(*self.image.pixelColor(x, y).getRgb())
                    intensity = pixel_color.red()  # Kami asumsikan gambar grayscale

                    histogram[intensity] += 1

            # Menghitung distribusi kumulatif
            cumulative_distribution = [0] * 256
            cumulative_distribution[0] = histogram[0] / total_pixels

            for i in range(1, 256):
                cumulative_distribution[i] = cumulative_distribution[i - 1] + histogram[i] / total_pixels

            # Menyesuaikan nilai pixel pada gambar hasil
            for x in range(width):
                for y in range(height):
                    pixel_color = QtGui.QColor(*self.image.pixelColor(x, y).getRgb())
                    intensity = pixel_color.red()  # Kami asumsikan gambar grayscale

                    new_intensity = int(255 * cumulative_distribution[intensity])
                    new_color = QtGui.QColor(new_intensity, new_intensity, new_intensity)
                    equalized_image.setPixelColor(x, y, new_color)
            

            output_pixmap = QtGui.QPixmap.fromImage(equalized_image)
            self.label_2.setPixmap(output_pixmap)   
            self.image = equalized_image

    def Fuzzy_HE_RGB(self):
        if self.label is not None:
            width = self.image.width()
            height = self.image.height()

            # Mengambil data piksel dari gambar input
            input_data = np.zeros((height, width, 3), dtype=np.uint8)
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
        if self.label_2 is not None:
            width = self.image.width()
            height = self.image.height()

            # Mengambil data piksel dari gambar input
            input_data = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    color = QtGui.QColor(self.image.pixel(x, y))
                    # Menghitung nilai greyscale menggunakan rumus Fuzzy
                    val = int(0.3 * color.red() + 0.59 * color.green() + 0.11 * color.blue())
                    input_data[y, x] = val

            # Membuat gambar output dan menampilkannya di pbOutput
            output_image = QtGui.QImage(input_data.data, width, height, width, QtGui.QImage.Format_Grayscale8)
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(output_image))
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

             
    def exitApplication(self):
        # untuk keluar dari aplikasi
        QtWidgets.qApp.quit()    
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
