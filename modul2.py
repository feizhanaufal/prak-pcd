#Import Library
import math
import sys
import cv2
import numpy as np
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import QDialog,QApplication,QMainWindow
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('gui.ui', self)
        self.image = None
        self.pushButton.clicked.connect(self.loadClicked)
        self.proses.clicked.connect(self.grays)
        # operasi titik
        self.actionpencerahan.triggered.connect(self.brightness)
        self.actionsimple_contras.triggered.connect(self.contrast)
        self.actioncontrast_stretching.triggered.connect(self.constret)
        self.actionNegative_Image.triggered.connect(self.negative)
        self.actionBiner_Image.triggered.connect(self.threshold)
        # histogram
        self.actionhistogram_graysclae.triggered.connect(self.grayHistogram)
        self.actionRGB_Histogram.triggered.connect(self.RGBhistogram)
        self.actionEqual_Histogram.triggered.connect(self.EqualHistogram)
        # operasi geometri
        self.actiontranslasi.triggered.connect(self.translasi)
        self.action_45_derajat.triggered.connect(self.rotasimin45)
        self.action45_derajat.triggered.connect(self.rotasi45)
        self.actionmin90_derajat.triggered.connect(self.rotasimin90)
        self.action90_derajat.triggered.connect(self.rotasi90)
        self.action180_derajat.triggered.connect(self.rotasi180)
        self.actiontranspose.triggered.connect(self.transpose)
        self.actionzoom_in.triggered.connect(self.zoomin)
        self.actionzoom_out.triggered.connect(self.zoomout)
        self.actionskewed_image.triggered.connect(self.skewedimage)
        self.actioncrop.triggered.connect(self.crop)
        # operasi aritmatika
        self.actiontambah.triggered.connect(self.tambah)
        self.actionkurang.triggered.connect(self.kurang)
        self.actionkali.triggered.connect(self.kali)
        self.actionbagi.triggered.connect(self.bagi)
        # operasi boolean
        self.actionAND.triggered.connect(self.AND)
        self.actionOR.triggered.connect(self.OR)
        self.actionXOR.triggered.connect(self.XOR)
        # operasi konvolusi
        self.actionkonfila.triggered.connect(self.kona)
        self.actionkonfilb.triggered.connect(self.konb)
        self.actionmeana.triggered.connect(self.mena)
        self.actionmeanb.triggered.connect(self.menb)
        self.actiongaussian.triggered.connect(self.gaus)
        self.actionsharp1.triggered.connect(self.shrp1)
        self.actionsharp2.triggered.connect(self.shrp2)
        self.actionsharp3.triggered.connect(self.shrp3)
        self.actionsharp4.triggered.connect(self.shrp4)
        self.actionsharp5_2.triggered.connect(self.shrp5)
        self.actionsharp6_2.triggered.connect(self.shrp6)
        self.actionlaplace.triggered.connect(self.lape)
        self.actionmedian_2.triggered.connect(self.median)
        self.actionmedmax_2.triggered.connect(self.medimax)
        self.actionmedmin_2.triggered.connect(self.medimin)
        # transformasi fourier
        self.actionDFT_Smooting.triggered.connect(self.smooth)
        self.actionDFT_HPF.triggered.connect(self.HPF)
        # deteksi tepi
        self.actionsobel.triggered.connect(self.sobell)
        self.actionprewitt.triggered.connect(self.prewit)
        self.actionroberts.triggered.connect(self.robert)
        # canny
        self.actioncanny.triggered.connect(self.cany)
        # morfologi
        self.actiondilasi.triggered.connect(self.lasi)
        self.actionerosi.triggered.connect(self.eros)
        self.actionopening.triggered.connect(self.open)
        self.actionclosing.triggered.connect(self.losing)
        self.actionskeletonizing.triggered.connect(self.skelet)
        # local biner
        self.actionbinary.triggered.connect(self.binar)
        self.actionbinaryinvers.triggered.connect(self.binarinvers)
        self.actiontrunch.triggered.connect(self.trunc)
        self.actionto_zero.triggered.connect(self.actionzero)
        self.actionto_zero_invers.triggered.connect(self.zeroinvers)
        # adaptive threshold
        self.actionMean_Thresholding.triggered.connect(self.meanthres)
        self.action_Gaussian_Thresholding.triggered.connect(self.gaussthress)
        self.actionotsu_threshld.triggered.connect(self.otsuthres)
        self.actioncountour.triggered.connect(self.countur)

        # color tracking
        self.actioncolor_tracking.triggered.connect(self.c_tracking)
        self.actioncolor_picker.triggered.connect(self.c_picker)
    @pyqtSlot()
    def loadClicked(self):
        self.loadImage('yamaha.jpeg')

    def loadImage(self,flname):
        self.image = cv2.imread(flname)
        self.displayImage()

    def grays(self):
        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i,j] = np.clip(0.299 * self.image[i, j, 0] + 0.587 * self.image[i, j, 1] + 0.114 * self.image[i, j, 2], 0, 255)
        self.image = gray
        self.displayImage(2)

    def brightness(self):
        try:
            self.image = cv2.cvtcolor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        h, w = self.image.shape[:2]
        brightness = 50
        for i in np.arange(h):
            for j in np.arange(w):
                a = self.image.item(i, j)
                b = np.clip(a + brightness, 0, 255)
                self.image.itemset((i, j), b)

        self.displayImage(1)

    def contrast(self):
        try:
            self.image = cv2.cvtcolor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        h, w = self.image.shape[:2]
        contrast = 1.6
        for i in np.arange(h):
            for j in np.arange(w):
                a = self.image.item(i, j)
                b = np.clip(a * contrast, 0, 255)
                self.image.itemset((i, j), b)

        self.displayImage(1)

    def constret(self):
        try:
            self.image = cv2.cvtcolor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        h, w = self.image.shape[:2]
        minv = np.min(self.image)
        maxv = np.max(self.image)

        for i in np.arange(h):
            for j in np.arange(w):
                a = self.image.item(i, j)
                b = float(a - minv) / (maxv - minv) * 255
                self.image.itemset((i, j), b)

        self.displayImage(1)

    def negative(self):
        try:
            self.image = cv2.cvtcolor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        h, w = self.image.shape[:2]
        for i in np.arange(h):
            for j in np.arange(w):
                a = self.image.item(i, j)
                b = np.clip(225 - a, 0, 225)
                self.image.itemset((i, j), b)

        self.displayImage(1)

    def threshold(self):
        try:
            self.image = cv2.cvtcolor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        h, w = self.image.shape[:2]
        for i in np.arange(h):
            for j in np.arange(w):
                a = self.image.item(i, j)
                #b = np.clip (a == 180 == 0 , a < 180 == 1 , a > 180 == 225)
                if a == 180:
                    b = 0
                elif a < 180:
                    b = 1
                elif a > 180:
                    b = 255
                self.image.itemset((i, j), b)

        self.displayImage(1)

    def grayHistogram(self):
        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i,j] = np.clip(0.299 * self.image[i, j, 0] + 0.587 * self.image[i, j, 1] + 0.114 * self.image[i, j, 2], 0, 255)
        self.image = gray
        self.displayImage(2)
        plt.hist(self.image.ravel(), 255, [0, 255])
        plt.show()

    def RGBhistogram(self):
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histo = cv2.calcHist([self.image], [i], None, [255], [0, 255])
            plt.plot(histo, color=col)
            plt.xlim([0, 256])
        self.displayImage(2)
        plt.show()

    def EqualHistogram(self):
        hist, bins = np.histogram(self.image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        self.image = cdf[self.image]
        self.displayImage(2)

        plt.plot(cdf_normalized, color='b')
        plt.hist(self.image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()

    def translasi(self):
        h, w = self.image.shape[:2]
        quarter_h, quarter_w = h / 4, w / 4
        T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
        image = cv2.warpAffine(self.image, T, (w, h))

        self.image = image
        self.displayImage(2)

    def rotasi(self, degree):
        h, w = self.image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 0.7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2

        rot_image = cv2.warpAffine(self.image, rotationMatrix, (h, w))
        self.image = rot_image
        self.displayImage(2)

    def rotasimin45(self):
        self.rotasi(-45)
    def rotasi45(self):
        self.rotasi(45)
    def rotasimin90(self):
        self.rotasi(-90)
    def rotasi90(self):
        self.rotasi(90)
    def rotasi180(self):
        self.rotasi(180)

    def transpose(self):
        image = cv2.imread('anatomy.jfif')
        window_nama = 'hasil'
        image = cv2.transpose(image)
        cv2.imshow(window_nama, image)
        cv2.waitKey(0)

    def zoomin(self):
        resize_image = cv2.resize(self.image, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
        cv2.imshow('original', self.image)
        cv2.imshow('zoom in', resize_image)

    def zoomout(self):
        resize_image = cv2.resize(self.image, None, fx=0.50, fy=0.50)
        cv2.imshow('original', self.image)
        cv2.imshow('zoom out', resize_image)

    def skewedimage(self):
        resize_image = cv2.resize(self.image, (900, 400), interpolation=cv2.INTER_AREA)
        cv2.imshow('original', self.image)
        cv2.imshow('skewed', resize_image)

    def crop(self):
        start_x = 20
        start_y = 20
        end_x = 300
        end_y = 300
        crop_img = self.image[start_x:end_x, start_y:end_y]
        cv2.imshow("cropped", crop_img)
        cv2.waitKey(0)

    def tambah(self):
        Image1 = cv2.imread('anatomy.jfif', 0)
        Image2 = cv2.imread('anatomy.jfif', 0)
        Image_tambah = Image1 + Image2
        cv2.imshow('image 1 original', Image1)
        cv2.imshow('image Tambah', Image_tambah)
        cv2.waitKey(0)
    def kurang(self):
        Image1 = cv2.imread('anatomy.jfif', 0)
        Image2 = cv2.imread('anatomy.jfif', 0)
        Image_kurang = Image1 - Image2
        cv2.imshow('image  original', Image1)
        cv2.imshow('image kurang', Image_kurang)
        cv2.waitKey(0)
    def kali(self):
        Image1 = cv2.imread('anatomy.jfif', 0)
        Image2 = cv2.imread('anatomy.jfif', 0)
        Image_kali = Image1 * Image2
        cv2.imshow('image  original', Image1)
        cv2.imshow('image kali', Image_kali)
        cv2.waitKey(0)
    def bagi(self):
        Image1 = cv2.imread('anatomy.jfif', 0)
        Image2 = cv2.imread('anatomy.jfif', 0)
        Image_bagi = Image1 / Image2
        cv2.imshow('image  original', Image1)
        cv2.imshow('image bagi', Image_bagi)
        cv2.waitKey(0)

    def AND(self):
        Image1 = cv2.imread('anatomy.jfif', 1)
        Image2 = cv2.imread('jeroan.jfif', 1)
        Image1 = cv2.cvtColor(Image1, cv2.COLOR_BGR2RGB)
        Image2 = cv2.cvtColor(Image2, cv2.COLOR_BGR2RGB)
        resize_image1=cv2.resize(Image1,(900,400),interpolation=cv2.INTER_CUBIC)
        resize_image2 = cv2.resize(Image2, (900, 400), interpolation=cv2.INTER_CUBIC)
        op_and = cv2.bitwise_and(resize_image1, resize_image2)
        cv2.imshow('image 1 original', Image1)
        cv2.imshow('image 2 original', Image2)
        cv2.imshow('image AND', op_and)
        cv2.waitKey(0)

    def OR(self):
        Image1 = cv2.imread('anatomy.jfif', 1)
        Image2 = cv2.imread('jeroan.jfif', 1)
        Image1 = cv2.cvtColor(Image1, cv2.COLOR_BGR2RGB)
        Image2 = cv2.cvtColor(Image2, cv2.COLOR_BGR2RGB)
        resize_image1 = cv2.resize(Image1, (900, 400), interpolation=cv2.INTER_CUBIC)
        resize_image2 = cv2.resize(Image2, (900, 400), interpolation=cv2.INTER_CUBIC)
        op_or = cv2.bitwise_or(resize_image1, resize_image2)
        cv2.imshow('image 1 original', Image1)
        cv2.imshow('image 2 original', Image2)
        cv2.imshow('image OR', op_or)
        cv2.waitKey(0)

    def XOR(self):
        Image1 = cv2.imread('anatomy.jfif', 1)
        Image2 = cv2.imread('jeroan.jfif', 1)
        Image1 = cv2.cvtColor(Image1, cv2.COLOR_BGR2RGB)
        Image2 = cv2.cvtColor(Image2, cv2.COLOR_BGR2RGB)
        resize_image1 = cv2.resize(Image1, (900, 400), interpolation=cv2.INTER_CUBIC)
        resize_image2 = cv2.resize(Image2, (900, 400), interpolation=cv2.INTER_CUBIC)
        op_xor = cv2.bitwise_xor(resize_image1, resize_image2)
        cv2.imshow('image 1 original', Image1)
        cv2.imshow('image 2 original', Image2)
        cv2.imshow('image XOR', op_xor)
        cv2.waitKey(0)

    def konvolusi(self, X, F):
        X_height = X.shape[0]   #menentukan tinggi citra x
        X_width = X.shape[1]    #menentukan lebar citra x
        F_height = F.shape[0]   #menentukan tinggi Kernel f
        F_width = F.shape[1]    #menentukan lebar Kernel f

        H = (F_height) // 2
        W = (F_width) // 2
        out = np.zeros((X_height,X_width))
        for i in np.arange(H+1, X_height - H): #pergerakan tinggi
            for j in np.arange(W+1, X_width - H): #pergerakan lebar
                sum=0 #menampung hasil perkalian
                for k in np.arange(-H, H+1): # untuk loop kernelnya
                    for l in np.arange(-W, W+1):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += (w*a)
                out[i,j] = sum
        return out

    def konvolusi2(self, X, F):
        X_height = X.shape[0]   #menentukan tinggi citra x
        X_width = X.shape[1]    #menentukan lebar citra x
        F_height = F.shape[0]   #menentukan tinggi Kernel f
        F_width = F.shape[1]    #menentukan lebar Kernel f

        H = 0
        W = 0

        bound = (F_height) //2

        out = np.zeros((X_height,X_width))
        for i in np.arange(H+1, X_height - bound): #pergerakan tinggi
            for j in np.arange(W+1, X_width - bound): #pergerakan lebar
                sum=0                   #menampung hasil perkalian
                for k in np.arange(-H, H+1): # untuk loop kernelnya
                    for l in np.arange(-W, W+1):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += (w*a)
                out[i,j] = sum
        return out


    def kona(self):
        X = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        F = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]])
        hasil = self.konvolusi(X, F)

        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        print(hasil)
        plt.show()

    def konb(self):
        X = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        F = np.array([[6, 0, -6],
                      [6, 1, -6],
                      [6, 0, -6]])
        hasil = self.konvolusi(X, F)

        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        print(hasil)
        plt.show()

    def mena (self):
        X = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        F = (1 / 9) * np.array(
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]])
        hasil = self.konvolusi(X, F)

        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        print(hasil)
        plt.show()

    def menb (self):
        X = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        F = (1 / 4) * np.array(
            [[1, 1, 0],
             [1, 1, 0],
             [0, 0, 0]])
        hasil = self.konvolusi(X, F)

        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        print(hasil)
        plt.show()

    def gaus(self):
        X = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        F = (1 / 345) * np.array(
            [[0, 0, 0, 0, 0],
             [0, 20, 33, 20, 0],
             [0, 33, 55, 33, 0],
             [0, 20, 33, 20, 0],
             [0, 0, 0, 0, 0]])
        hasil = self.konvolusi(X, F)

        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        print(hasil)
        plt.show()

    def shrp1(self):
        X = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        F = np.array(
            [[-1,-1,-1],
             [-1,8,-1],
             [-1,-1,-1]])
        hasil = self.konvolusi(X, F)

        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        print(hasil)
        plt.show()

    def shrp2(self):
        X = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        F = np.array(
            [[-1, -1, -1],
             [-1, 9, -1],
             [-1, -1, -1]])
        hasil = self.konvolusi(X, F)

        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        print(hasil)
        plt.show()
    def shrp3(self):
        X = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        F = np.array(
            [[0, -1, 0],
             [-1, 5, -1],
             [0, -1, 0]])
        hasil = self.konvolusi(X, F)

        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        print(hasil)
        plt.show()

    def shrp4(self):
        X = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        F = np.array(
            [[1, -2, 1],
            [-2, 5, -2],
            [1, -2, 1]])
        hasil = self.konvolusi(X, F)

        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        print(hasil)
        plt.show()

    def shrp5(self):
        X = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        F = np.array(
            [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]])
        hasil = self.konvolusi(X, F)

        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        print(hasil)
        plt.show()

    def shrp6(self):
        X = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        F = np.array(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]])
        hasil = self.konvolusi(X, F)

        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        print(hasil)
        plt.show()

    def lape(self):
        X = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        F = (1 / 16) * np.array(
            [[0, 0, -1, 0, 0],
             [0, -1, -2, -1, 0],
             [-1, -2, 16, -2, -1],
             [0, -1, -2, -1, 0],
             [0, 0, -1, 0, 0]])
        hasil = self.konvolusi(X, F)

        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        print(hasil)
        plt.show()

    def median(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_out = img.copy()
        h, w = img.shape[:2]
        for i in np.arange(3, h - 3):
            for j in np.arange(3, w - 3):
                neighbors = []
                for k in np.arange(-3, 4):      #pergeseran nilai
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        neighbors.append(a) #buat nambahin ilai a ke dlm nilai neigbour
                neighbors.sort()        #buat ngurutin neighbour
                median = neighbors[24]  #negpoisiin nilai neighbour
                b = median  #masukan ke vriabel b
                img_out.itemset((i, j), b)

        print(img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()

    def medimax(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_out = img.copy()
        h, w = img.shape[:2]
        for i in np.arange(3, h - 3):
            for j in np.arange(3, w - 3):
                #neighbors = []
                max = 0
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        if a > max:
                            max = a
                            b = max
                img_out.itemset((i, j), b)

        print(img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()

    def medimin(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_out = img.copy()
        h, w = img.shape[:2]
        for i in np.arange(3, h - 3):
            for j in np.arange(3, w - 3):
                #neighbors = []
                min = 255
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        if a < min:
                            min = a
                            b = min
                img_out.itemset((i, j), b)

        print(img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()

    def smooth(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)

        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 120
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 1

        #mengembalikan titik origin citra ke kiri atas
        fshift = dft_shift * mask
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)
        #mengembalikan dari citra frekkuensi ke citra spasial
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('input image')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of image')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + mask')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse vourier')
        plt.show()

    def HPF(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)

        mask = np.ones((rows, cols, 2), np.uint8)
        r = 120
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 0

        fshift = dft_shift * mask
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)

        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('input image')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of image')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + mask')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse vourier')
        plt.show()

    def sobell(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        gx = self.konvolusi(img, sobelx)
        gy = self.konvolusi(img, sobely)
        hasil = np.sqrt((gx * gx) + (gy * gy))

        hasil = (hasil / np.max(hasil)) * 255

        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()
        print("nilai citra awal : ", img)
        print("nilai citra sobel : ", hasil)

    def prewit(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        prewittx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewitty = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        gx = self.konvolusi(img, prewittx)
        gy = self.konvolusi(img, prewitty)
        img_out = np.sqrt((gx * gx) + (gy * gy))

        hasil = (img_out / np.max(img_out)) * 255

        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()
        print("nilai citra awal : ", img)
        print("nilai citra prewitt: ", hasil)

    def robert(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ROBERTx = np.array([[0, 1],
                           [-1, 0]])
        ROBERTy = np.array([[1, 0],
                           [0, -1]])
        gx = self.konvolusi2(img, ROBERTx)
        gy = self.konvolusi2(img, ROBERTy)
        img_out = np.sqrt((gx * gx) + (gy * gy))

        hasil = (img_out / np.max(img_out)) * 255

        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()
        print("nilai citra awal : ", img)
        print("nilai citra roberts : ", hasil)

    def cany(self):
        #step 1 :
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        kernel = (1 / 57) * np.array(
            [[0,1,2,1,0],
             [1,3,5,3,1],
             [2,5,9,5,2],
             [0,1,2,1,0],
             [1,3,5,3,1]])
        gauss = self.konvolusi(img, kernel)
        s1 = gauss.astype("uint8")
        cv2.imshow("Reduksi Noise", s1)

        #step 2 :
        kobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        gx = self.konvolusi(img, kobelx)
        gy = self.konvolusi(img, kobely)
        img_out = np.sqrt((gx * gx) + (gy * gy))
        hasil = (img_out / np.max(img_out)) * 255

        s2 = hasil.astype("uint8")
        cv2.imshow("Finding Gradien", s2)

        theta = np.arctan2(gx,gy)

        #step 3 :
        angle = theta * 180. / np.pi
        angle[angle < 0] += 180

        H,W = img.shape[:2]

        for i in range(1, H-1):
            for j in range(1, W-1):
                try:
                    q = 255
                    r = 255

                    # angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img_out[i, j+1]
                        r = img_out[i, j-1]

                    # angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img_out[i+1, j - 1]
                        r = img_out[i-1, j + 1]

                    # angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img_out[i + 1, j]
                        r = img_out[i - 1, j]
                    # angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img_out[i - 1, j-1]
                        r = img_out[i + 1, j+1]

                    if (img_out[i,j] >= q) and (img_out[i,j] >= r):
                        s2[i,j] = img_out[i,j]
                    else:
                        s2[i,j] = 0

                except IndexError as e:
                    pass

        s3 = s2.astype("uint8")
        cv2.imshow("Non-Maximum Suppression", s3)

        #step 4 :
        weak = 50
        strong = 200
        for i in np.arange(H):
            for j in np.arange(W):
                a = s3.item(i,j)
                if (a>weak) :
                    b = weak
                    if (a>strong):
                        b=255
                else:
                    b=0

                s3.itemset((i,j), b)

        s4 = s3.astype ("uint8")
        cv2.imshow("Hysteresis part 1", s4)

        # hyterisis thresholding 2
        strong = 128
        for i in range(1, H-1):
            for j in range(1, W-1):
                if (s4[i,j] == weak):
                    try:
                        if ((s4[i+1, j-1] == strong) or (s4[i+1, j] == strong) or (s4[i+1, j+1] == strong) or
                                (s4[i,j-1] == strong) or (s4[i, j+1] == strong) or (s4[i-1, j-1] == strong) or
                                (s4[i-1, j] == strong) or (s4[i-1, j+1] == strong)):
                            s4[i,j] = strong
                        else:
                            s4[i,j] = 0

                    except IndexError as e:
                        pass

        s5 = s4.astype("uint8")
        cv2.imshow("Hysteresis part 2", s5)

    def lasi(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 128, 255, 0)
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        hasil = cv2.dilate(img, strel, iterations=1)
        self.image = hasil
        self.displayImage(2)

        height, width = img.shape[:2]
        for y in range(height):
            for x in range(width):
                print(img[y, x], end="\t")
            print("\t")

        height, width = hasil.shape[:2]
        for y in range(height):
            for x in range(width):
                print(hasil[y, x], end="\t")
            print("\t")

    def eros(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 128, 255, 0)
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hasil = cv2.erode(img, strel, iterations=1)
        self.image = hasil
        self.displayImage(2)

        height, width = hasil.shape[:2]
        for y in range(height):
            for x in range(width):
                print(hasil[y, x], end="\t")
            print("\t")

    def open(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        hasil = cv2.morphologyEx(img, cv2.MORPH_OPEN, strel, iterations=1)
        self.image = hasil
        self.displayImage(2)

        height, width = hasil.shape[:2]
        for y in range(height):
            for x in range(width):
                print(hasil[y, x], end="\t")
            print("\t")

    def losing(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hasil = cv2.morphologyEx(img, cv2.MORPH_CLOSE, strel, iterations=1)
        self.image = hasil
        self.displayImage(2)

        height, width = hasil.shape[:2]
        for y in range(height):
            for x in range(width):
                print(hasil[y, x], end="\t")
            print("\t")

    def skelet(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 127, 255, 0)
        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, strel)
        temp = cv2.subtract(img, open)
        erosi = cv2.erode(img, strel, iterations=1)
        skel = cv2.bitwise_or(skel, temp)
        self.image = skel
        self.displayImage(2)
        print("piksel awal :", img)
        print("")
        print("piksel akhir :", skel)

    def binar(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        T = 127
        max = 255
        ret, binary = cv2.threshold(img, T, max, cv2.THRESH_BINARY)
        self.image = binary
        self.displayImage(2)

        height, width = binary.shape[:2]
        for y in range(height):
            for x in range(width):
                print(binary[y, x], end="\t")
            print("\t")

    def binarinvers(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        T = 127
        max = 255
        ret, binaryInv = cv2.threshold(img, T, max, cv2.THRESH_BINARY_INV)
        self.image = binaryInv
        self.displayImage(2)

        height, width = binaryInv.shape[:2]  # image height and width
        for y in range(height):
            for x in range(width):
                print(binaryInv[y, x], end="\t")
            print("\t")


    def trunc(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        T = 127
        max = 255
        ret, trunc = cv2.threshold(img, T, max, cv2.THRESH_TRUNC)
        self.image = trunc
        self.displayImage(2)

        height, width = trunc.shape[:2]
        for y in range(height):
            for x in range(width):
                print(trunc[y, x], end="\t")
            print("\t")

    def actionzero(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        T = 127
        max = 255
        ret, toZero = cv2.threshold(img, T, max, cv2.THRESH_TOZERO)
        self.image = toZero
        self.displayImage(2)

        height, width = toZero.shape[:2]
        for y in range(height):
            for x in range(width):
                print(toZero[y, x], end="\t")
            print("\t")

    def zeroinvers(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        T = 127
        max = 255
        ret, toZeroInv = cv2.threshold(img, T, max, cv2.THRESH_TOZERO_INV)
        self.image = toZeroInv
        self.displayImage(2)

        height, width = toZeroInv.shape[:2]
        for y in range(height):
            for x in range(width):
                print(toZeroInv[y, x], end="\t")
            print("\t")

    def meanthres(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        max = 255
        mean_thres = cv2.adaptiveThreshold(img, max, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
        self.image = mean_thres
        self.displayImage(2)

        height, width = mean_thres.shape[:2]
        for y in range(height):
            for x in range(width):
                print(mean_thres[y, x], end="\t")
            print("\t")

    def gaussthress(self):

        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        max = 255
        gauss_thres = cv2.adaptiveThreshold(img, max, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
        self.image = gauss_thres
        self.displayImage(2)

        height, width = gauss_thres.shape[:2]
        for y in range(height):
            for x in range(width):
                print(gauss_thres[y, x], end="\t")
            print("\t")

    def otsuthres(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        T = 150
        max = 255
        ret, otsu_thres = cv2.threshold(img, T, max, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.image = otsu_thres
        self.displayImage(2)

        height, width = otsu_thres.shape[:2]  # image height and width
        for y in range(height):
            for x in range(width):
                print(otsu_thres[y, x], end="\t")
            print("\t")

    def countur(self):
        img = cv2.imread('jj.jpg')
        img = cv2.resize(img, (1000, 500))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        print(contours)
        i = 0
        for contour in contours:
            if i == 0:
                i = 1
                continue
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            cv2.drawContours(img, [contour], -1, (0, 0, 255), 3)
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10'] / M['m00'])
                y = int(M['m01'] / M['m00'])
            if len(approx) == 3:
                cv2.putText(img, 'Segitiga', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            elif len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                if abs(w - h) < 100:
                    cv2.putText(img, 'Persegi', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                else:
                    cv2.putText(img, 'Persegi Panjang', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            elif len(approx) == 10:
                cv2.putText(img, 'Bintang', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(img, 'Lingkaran', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self.image = img
        self.displayImage(2)
        cv2.imshow('hasil', img)

    def c_tracking(self):
        cam = cv2.VideoCapture(0)

        while True:
            _, frame = cam.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_color = np.array([66, 98, 100])
            upper_color = np.array([156, 232, 255])
            mask = cv2.inRange(hsv, lower_color, upper_color)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", mask)
            cv2.imshow("Result", result)

            key = cv2.waitKey(1)
            if key == 27:
                break
        cam.release()
        cv2.destroyAllWindows()

    def c_picker(self):
        def nothing(x):
            pass

        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Trackbars")

        cv2.createTrackbar("L-H","Trackbars",0,179,nothing)
        cv2.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
        cv2.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
        cv2.createTrackbar("U-H", "Trackbars", 179, 179, nothing)
        cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
        cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

        while True:
            _, frame = cam.read()
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

            l_h = cv2.getTrackbarPos("L-H","Trackbars")
            l_s = cv2.getTrackbarPos("L-S", "Trackbars")
            l_v = cv2.getTrackbarPos("L-V", "Trackbars")
            u_h = cv2.getTrackbarPos("U-H", "Trackbars")
            u_s = cv2.getTrackbarPos("U-S", "Trackbars")
            u_v = cv2.getTrackbarPos("U-V", "Trackbars")

            lower_color = np.array([l_h,l_s,l_v])
            upper_color = np.array([u_h,u_s,u_v])
            mask = cv2.inRange(hsv,lower_color,upper_color)
            result = cv2.bitwise_and(frame,frame,mask=mask)

            cv2.imshow("frame",frame)
            cv2.imshow("mask",mask)
            cv2.imshow("result", result)

            key = cv2.waitKey(1)
            if key==27:
                break
        cam.release()
        cv2.destroyAllWindows()


    def displayImage(self,windows=1):
        qformat = QImage.Format_Indexed8
        if len(self.image.shape)==3: #row[0],col[1],channel[2]
            if(self.image.shape[2])==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img=QImage(self.image,self.image.shape[1],self.image.shape[0],
            self.image.strides[0],qformat)

        img = img.rgbSwapped()

        if windows == 1:
            self.label.setPixmap(QPixmap.fromImage(img))
            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)
        if windows == 2:
            self.label_2.setPixmap(QPixmap.fromImage(img))
            self.label_2.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            self.label_2.setScaledContents(True)

app=QtWidgets.QApplication(sys.argv)
window=ShowImage()
window.setWindowTitle('Show Image GUI')
window.show()
sys.exit(app.exec_())