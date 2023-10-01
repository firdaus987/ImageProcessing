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
from Aritmath1 import Ui_Aritmath
from Geometri import Ui_Geometri


class Colours(object):
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
        self.menubar.setGeometry(QtCore.QRect(0, 0, 646, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuHistogram = QtWidgets.QMenu(self.menuView)
        self.menuHistogram.setObjectName("menuHistogram")
        self.menuWarna = QtWidgets.QMenu(self.menubar)
        self.menuWarna.setObjectName("menuWarna")
        self.menuRGB = QtWidgets.QMenu(self.menuWarna)
        self.menuRGB.setObjectName("menuRGB")
        self.menuRGB_ke_Grayscale = QtWidgets.QMenu(self.menuWarna)
        self.menuRGB_ke_Grayscale.setObjectName("menuRGB_ke_Grayscale")
        self.menuBrightness = QtWidgets.QMenu(self.menuWarna)
        self.menuBrightness.setObjectName("menuBrightness")
        self.menuBit_Depth = QtWidgets.QMenu(self.menuWarna)
        self.menuBit_Depth.setObjectName("menuBit_Depth")
        self.menuImage_Processing = QtWidgets.QMenu(self.menubar)
        self.menuImage_Processing.setObjectName("menuImage_Processing")
        self.menuAritmatic_Operations = QtWidgets.QMenu(self.menubar)
        self.menuAritmatic_Operations.setObjectName("menuAritmatic_Operations")
        self.menuFilter = QtWidgets.QMenu(self.menubar)
        self.menuFilter.setObjectName("menuFilter")
        self.menuEdge_Detection = QtWidgets.QMenu(self.menuFilter)
        self.menuEdge_Detection.setObjectName("menuEdge_Detection")
        self.menuGaussian_Blur = QtWidgets.QMenu(self.menuFilter)
        self.menuGaussian_Blur.setObjectName("menuGaussian_Blur")
        self.menuEdge_Detection_2 = QtWidgets.QMenu(self.menubar)
        self.menuEdge_Detection_2.setObjectName("menuEdge_Detection_2")
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
        self.menuGeometri = QtWidgets.QMenu(self.menubar)
        self.menuGeometri.setObjectName("menuGeometri")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionBuka = QtWidgets.QAction(MainWindow)
        self.actionBuka.setObjectName("actionBuka")
        self.actionBuka.triggered.connect(self.open_image)
        self.actionSimpan_Sebagai = QtWidgets.QAction(MainWindow)
        self.actionSimpan_Sebagai.setObjectName("actionSimpan_Sebagai")
        self.actionKeluar = QtWidgets.QAction(MainWindow)
        self.actionKeluar.setObjectName("actionKeluar")
        self.actionKeluar.triggered.connect(self.show_exit_confirmation) 
        self.actionInput = QtWidgets.QAction(MainWindow)
        self.actionInput.setObjectName("actionInput")
        self.actionInput.triggered.connect(self.show_input_histogram)
        self.actionOutput = QtWidgets.QAction(MainWindow)
        self.actionOutput.setObjectName("actionOutput")
        self.actionOutput.triggered.connect(self.show_output_histogram)
        self.actionInput_Output = QtWidgets.QAction(MainWindow)
        self.actionInput_Output.setObjectName("actionInput_Output")
        self.actionInput_Output.triggered.connect(self.show_input_and_output)
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
        self.actionKuning.triggered.connect(self.apply_yellow_filter)
        self.actionCyan = QtWidgets.QAction(MainWindow)
        self.actionCyan.setObjectName("actionCyan")
        self.actionUngu = QtWidgets.QAction(MainWindow)
        self.actionUngu.setObjectName("actionUngu")
        self.actionAbu_abu = QtWidgets.QAction(MainWindow)
        self.actionAbu_abu.setObjectName("actionAbu_abu")
        self.actionAverage = QtWidgets.QAction(MainWindow)
        self.actionAverage.setObjectName("actionAverage")
        self.actionAverage.triggered.connect(self.average_filter)
        self.actionLightness = QtWidgets.QAction(MainWindow)
        self.actionLightness.setObjectName("actionLightness")
        self.actionLightness.triggered.connect(self.lightness_filter)
        self.actionLuminance = QtWidgets.QAction(MainWindow)
        self.actionLuminance.setObjectName("actionLuminance")
        self.actionLuminance.triggered.connect(self.luminance_filter)
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
        self.actionHistogram_Equalization.triggered.connect(self.histogram_equalization_process)
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
        self.actionSquare_4.triggered.connect(self.dilate_image)
        self.actionSquare_6 = QtWidgets.QAction(MainWindow)
        self.actionSquare_6.triggered.connect(self.dilate_image)
        self.actionSquare_6.setObjectName("actionSquare_6")
        self.actionCross_4 = QtWidgets.QAction(MainWindow)
        self.actionCross_4.setObjectName("actionCross_4")
        self.actionCross_4.triggered.connect(self.dilate_image)
        self.actionSquare_9 = QtWidgets.QAction(MainWindow)
        self.actionSquare_9.setObjectName("actionSquare_9")
        self.actionSquare_10 = QtWidgets.QAction(MainWindow)
        self.actionSquare_10.setObjectName("actionSquare_10")
        self.actionOperasi_Aritmatika = QtWidgets.QAction(MainWindow)
        self.actionOperasi_Aritmatika.setObjectName("actionOperasi_Aritmatika")
        self.actionOperasi_Aritmatika.triggered.connect(self.Aritmath1)
        self.actionOperasi_Geometri = QtWidgets.QAction(MainWindow)
        self.actionOperasi_Geometri.setObjectName("actionOperasi_Geometri")
        self.actionOperasi_Geometri.triggered.connect(self.Geometri)
        self.menuFile.addAction(self.actionBuka)
        self.menuFile.addAction(self.actionSimpan_Sebagai)
        self.menuFile.addAction(self.actionKeluar)
        self.menuHistogram.addAction(self.actionInput)
        self.menuHistogram.addAction(self.actionOutput)
        self.menuHistogram.addAction(self.actionInput_Output)
        self.menuView.addAction(self.menuHistogram.menuAction())
        self.menuRGB.addAction(self.actionKuning)
        self.menuRGB.addAction(self.actionCyan)
        self.menuRGB.addAction(self.actionUngu)
        self.menuRGB.addAction(self.actionAbu_abu)
        self.menuRGB_ke_Grayscale.addAction(self.actionAverage)
        self.menuRGB_ke_Grayscale.addAction(self.actionLightness)
        self.menuRGB_ke_Grayscale.addAction(self.actionLuminance)
        self.menuBrightness.addAction(self.actionContrast)
        self.menuBit_Depth.addAction(self.action1_Bit)
        self.menuBit_Depth.addAction(self.action2_BIt)
        self.menuBit_Depth.addAction(self.action3_bit)
        self.menuBit_Depth.addAction(self.action4_bit)
        self.menuBit_Depth.addAction(self.action5_bit)
        self.menuBit_Depth.addAction(self.action6_bit)
        self.menuBit_Depth.addAction(self.action7_bit)
        self.menuWarna.addAction(self.menuRGB.menuAction())
        self.menuWarna.addAction(self.menuRGB_ke_Grayscale.menuAction())
        self.menuWarna.addAction(self.menuBrightness.menuAction())
        self.menuWarna.addAction(self.actionBrightness_Contras)
        self.menuWarna.addAction(self.actionInvers)
        self.menuWarna.addAction(self.actionLog_Brightness)
        self.menuWarna.addAction(self.menuBit_Depth.menuAction())
        self.menuWarna.addAction(self.actionGamma_Corrections)
        self.menuImage_Processing.addAction(self.actionHistogram_Equalization)
        self.menuImage_Processing.addAction(self.actionFuzzy_HE_RGB)
        self.menuImage_Processing.addAction(self.actionFuzzy_Grayscale)
        self.menuAritmatic_Operations.addAction(self.actionOperasi_Aritmatika)
        self.menuGeometri.addAction(self.actionOperasi_Geometri)
        self.menuEdge_Detection.addAction(self.actionEdge_Detection_1)
        self.menuEdge_Detection.addAction(self.actionEdge_Detection_2)
        self.menuEdge_Detection.addAction(self.actionEdge_Detection_3)
        self.menuGaussian_Blur.addAction(self.actionGaussian_Blur_3x3)
        self.menuGaussian_Blur.addAction(self.actionGaussian_Blur_5x5)
        self.menuFilter.addAction(self.actionIdentity)
        self.menuFilter.addAction(self.menuEdge_Detection.menuAction())
        self.menuFilter.addAction(self.actionSharpen)
        self.menuFilter.addAction(self.menuGaussian_Blur.menuAction())
        self.menuFilter.addAction(self.actionUnshape_Masking)
        self.menuFilter.addAction(self.actionAverage_Filter)
        self.menuFilter.addAction(self.actionLow_Pass_Filter)
        self.menuFilter.addAction(self.actionHigh_Pass_Filter)
        self.menuFilter.addAction(self.actionBandstop_Filter)
        self.menuEdge_Detection_2.addAction(self.actionPrewitt)
        self.menuEdge_Detection_2.addAction(self.actionSobel)
        self.menuErosion.addAction(self.actionSquare_3)
        self.menuErosion.addAction(self.actionSquare_5)
        self.menuErosion.addAction(self.actionCross_3)
        self.menuDilation.addAction(self.actionSquare_4)
        self.menuDilation.addAction(self.actionSquare_6)
        self.menuDilation.addAction(self.actionCross_4)
        self.menuOpening.addAction(self.actionSquare_9)
        self.menuClosing.addAction(self.actionSquare_10)
        self.menuMorfologi.addAction(self.menuErosion.menuAction())
        self.menuMorfologi.addAction(self.menuDilation.menuAction())
        self.menuMorfologi.addAction(self.menuOpening.menuAction())
        self.menuMorfologi.addAction(self.menuClosing.menuAction())
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuWarna.menuAction())
        self.menubar.addAction(self.menuImage_Processing.menuAction())
        self.menubar.addAction(self.menuAritmatic_Operations.menuAction())
        self.menubar.addAction(self.menuFilter.menuAction())
        self.menubar.addAction(self.menuEdge_Detection_2.menuAction())
        self.menubar.addAction(self.menuMorfologi.menuAction())
        self.menubar.addAction(self.menuGeometri.menuAction())
        

        self.retranslateUi(MainWindow)
        
        # Inisialisasi jendela "Colours"
        self.colours_window = Colours()
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def Geometri(self):
     
        
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_Geometri()
        self.ui.setupUi(self.MainWindow)
        self.MainWindow.show()

    def Aritmath1(self):

        
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_Aritmath()
        self.ui.setupUi(self.MainWindow)
        self.MainWindow.show()



    
    def Main_Colours(self):
        self.colours_window.show()

    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Colours"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.menuHistogram.setTitle(_translate("MainWindow", "Histogram"))
        self.menuWarna.setTitle(_translate("MainWindow", "Warna"))
        self.menuRGB.setTitle(_translate("MainWindow", "RGB"))
        self.menuRGB_ke_Grayscale.setTitle(_translate("MainWindow", "RGB ke Grayscale"))
        self.menuBrightness.setTitle(_translate("MainWindow", "Brightness"))
        self.menuBit_Depth.setTitle(_translate("MainWindow", "Bit Depth"))
        self.menuImage_Processing.setTitle(_translate("MainWindow", "Image Processing"))
        self.menuAritmatic_Operations.setTitle(_translate("MainWindow", "Aritmatic Operations"))
        self.menuFilter.setTitle(_translate("MainWindow", "Filter"))
        self.menuEdge_Detection.setTitle(_translate("MainWindow", "Edge Detection"))
        self.menuGaussian_Blur.setTitle(_translate("MainWindow", "Gaussian Blur"))
        self.menuEdge_Detection_2.setTitle(_translate("MainWindow", "Edge Detection"))
        self.menuMorfologi.setTitle(_translate("MainWindow", "Morfologi"))
        self.menuErosion.setTitle(_translate("MainWindow", "Erosion"))
        self.menuDilation.setTitle(_translate("MainWindow", "Dilation"))
        self.menuOpening.setTitle(_translate("MainWindow", "Opening"))
        self.menuClosing.setTitle(_translate("MainWindow", "Closing"))
        self.menuGeometri.setTitle(_translate("MainWindow", "Geometri"))
        self.actionBuka.setText(_translate("MainWindow", "Buka"))
        self.actionSimpan_Sebagai.setText(_translate("MainWindow", "Simpan Sebagai"))
        self.actionSimpan_Sebagai.triggered.connect(self.save_image)
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
        self.actionAverage.setText(_translate("MainWindow", "Average"))
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
        self.actionOperasi_Aritmatika.setText(_translate("MainWindow", "Operasi Aritmatika"))
        self.actionOperasi_Geometri.setText(_translate("MainWindow", "Operasi Geometri"))
    
    def Aritmath(self):
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_Aritmath()
        self.ui.setupUi(MainWindow)
        self.MainWindow.show()

    def Geometri(self):
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_Geometri()
        self.ui.setupUi(MainWindow)
        self.MainWindow.show()
        
    def open_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly 
        self.label_2.clear()
        file_name, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)
        if file_name:
            image = QtGui.QImage(file_name)
            if not image.isNull():
                pixmap = QtGui.QPixmap.fromImage(image)
                self.label.setPixmap(pixmap) 
                self.label.setScaledContents(True) 
                self.image = image  # Simpan gambar dalam atribut self.image
    
    
    def save_image(self):
        if hasattr(self, 'gambar'):
            # Inisialisasi opsi untuk dialog pemilihan berkas
            options = QFileDialog.Options()
            # Menambahkan opsi mode baca saja ke dalam opsi dialog
            options |= QFileDialog.ReadOnly 
            # menampung file path dari dialog open file dan difilter hanya format png , jpg , bmp
            file_name, _ = QFileDialog.getSaveFileName(None, "Save Image File", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)
            # check apakah terdapat path file
            if file_name:
                #Simpan gambar yang telah diformat
                self.gambar.save(file_name)  
            
    def exit_application(self):
        QtWidgets.qApp.quit()
    
    def show_exit_confirmation(self):
        reply = QMessageBox.question(None, 'Konfirmasi Keluar', 'Apakah Anda yakin ingin keluar?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
        if reply == QMessageBox.Yes:
            self.exit_application()
            
    def apply_yellow_filter(self):
        # options = QFileDialog.Options()
        # options |= QFileDialog.ReadOnly
        # file_name, _ = QFileDialog.getOpenFileName(None, 'Buka Gambar', '', 'Images (*.png *.jpg *.bmp);;All Files (*)', options=options)
        
        if hasattr(self , 'pathh'):
            image = Image.open(self.pathh)
            width, height = image.size

            for i in range(width):
                for j in range(height):
                    r, g, b = image.getpixel((i, j))
                    r1 = min(r + 255, 255)  # Ensure that the value doesn't exceed 255
                    g1 = min(g + 255, 255)  # Ensure that the value doesn't exceed 255
                    b1 = b                  # No change needed for blue
                    image.putpixel((i, j), (r1, g1, b1))

            output_file_name = self.pathh.replace(".", "_yellow.")
            image.save(output_file_name)  # Save the modified image with "_yellow" suffix
            self.show_image(output_file_name)
            
    def dilate_image(self):
        if hasattr(self, 'image'):
            # Gunakan filter berukuran 3x3 untuk dilasi
            kernel = np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]], dtype=np.uint8)
            
            # Lakukan dilasi pada citra menggunakan OpenCV
            dilated_image = cv2.dilate(self.image, kernel, iterations=1)
            
            # Konversi hasil dilasi ke tipe QImage
            h, w, c = dilated_image.shape
            bytes_per_line = 3 * w
            q_image = QImage(dilated_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Tampilkan hasil dilasi pada label
            pixmap = QPixmap.fromImage(q_image)
            self.label_2.setPixmap(pixmap)
            self.label_2.setScaledContents(True)
            
    def average_filter(self):
        width = self.image.width()
        height = self.image.height()
        average = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)
        
        for y in range(height):
            for x in range(width):
                pixel_color = QtGui.QColor(self.image.pixel(x, y))
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                rumusAverage = int((r + g + b)/3)
                grayscale_color = QtGui.QColor(rumusAverage, rumusAverage, rumusAverage)
                average.setPixelColor(x, y, grayscale_color)
        
        p = QtGui.QPixmap.fromImage(average) 
        
        self.label_2.setPixmap(p)
        self.gambar = average  
    
    def lightness_filter(self):
        width = self.image.width()
        height = self.image.height()
        lightness = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)
        
        for y in range(height):
            for x in range(width):
                pixel_color = QtGui.QColor(self.image.pixel(x, y))
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                rumusLightness = int((max(r,g,b)+ min(r,g,b))/2)
                grayscale_color = QtGui.QColor(rumusLightness, rumusLightness, rumusLightness)
                lightness.setPixelColor(x, y, grayscale_color)
        
        p = QtGui.QPixmap.fromImage(lightness) 
        self.label_2.setPixmap(p)
        self.gambar = lightness 

    def luminance_filter(self):
        width = self.image.width()
        height = self.image.height()
        grayscale_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

        for y in range(height):
            for x in range(width):
                pixel_color = QtGui.QColor(self.image.pixel(x, y))
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                rumusLuminance = int(0.299 * r + 0.587 * g + 0.114 * b)
                grayscale_color = QtGui.QColor(rumusLuminance, rumusLuminance, rumusLuminance)
                grayscale_image.setPixelColor(x, y, grayscale_color)
        
        p = QtGui.QPixmap.fromImage(grayscale_image) 
        self.label_2.setPixmap(p)
        self.gambar = grayscale_image       
    
    def show_image(self, file_name):
        pixmap = QtGui.QPixmap(file_name)
        self.label_2.setPixmap(pixmap)
        self.label_2.setScaledContents(True)
    
    def histogram_equalization_process(self):
        if hasattr(self, 'image'):
            gray_image = self.image.convertToFormat(QtGui.QImage.Format_Grayscale8)

        # Hitung histogram gambar
        histogram = [0] * 256
        for y in range(gray_image.height()):
            for x in range(gray_image.width()):
                pixel_value = gray_image.pixelColor(x, y).red()
                histogram[pixel_value] += 1

        # Normalisasi histogram
        total_pixels = gray_image.width() * gray_image.height()
        normalized_histogram = [value / total_pixels for value in histogram]

        # Hitung distribusi kumulatif
        cumulative_distribution = [sum(normalized_histogram[:i + 1]) for i in range(256)]

        # Ubah intensitas piksel berdasarkan distribusi kumulatif
        for y in range(gray_image.height()):
            for x in range(gray_image.width()):
                pixel_value = gray_image.pixelColor(x, y).red()
                new_pixel_value = int(cumulative_distribution[pixel_value] * 255)
                gray_image.setPixelColor(x, y, QtGui.QColor(new_pixel_value, new_pixel_value, new_pixel_value))


        # Tampilkan gambar hasil
        self.show_image(gray_image)
        
    def show_input_histogram(self):
        if hasattr(self, 'image'):
            gray_image = self.image.convertToFormat(QtGui.QImage.Format_Grayscale8)

            # Hitung histogram gambar
            histogram = [0] * 256
            for y in range(gray_image.height()):
                for x in range(gray_image.width()):
                    pixel_value = gray_image.pixelColor(x, y).red()
                    histogram[pixel_value] += 1

            # Tampilkan histogram menggunakan Matplotlib
            plt.figure()
            plt.bar(np.arange(256), histogram, color='gray', alpha=0.7)
            plt.xlabel('Nilai Piksel')
            plt.ylabel('Frekuensi')
            plt.title('Histogram Gambar Input')
            plt.show()
            
    def show_output_histogram(self):
        if hasattr(self, 'label_2'):
            pixmap = self.label_2.pixmap()
            if pixmap:
                gray_image = pixmap.toImage().convertToFormat(QtGui.QImage.Format_Grayscale8)

                # Hitung histogram gambar hasil equalisasi
                histogram = [0] * 256
                for y in range(gray_image.height()):
                    for x in range(gray_image.width()):
                        pixel_value = gray_image.pixelColor(x, y).red()
                        histogram[pixel_value] += 1

                # Tampilkan histogram menggunakan Matplotlib
                plt.figure()
                plt.bar(np.arange(256), histogram, color='gray', alpha=0.7)
                plt.xlabel('Nilai Piksel')
                plt.ylabel('Frekuensi')
                plt.title('Histogram Gambar Output')
                plt.show()

    def show_input_and_output(self):
        self.show_input_histogram()
        self.show_output_histogram()
        



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Colours()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
