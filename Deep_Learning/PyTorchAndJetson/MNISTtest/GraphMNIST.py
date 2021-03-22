"""
Gui program to show the behaviour of a trained convolution NN.
You can design a pattern and ask the net for interpretation.
Or you can scan the dataset of test with the net result
Net model, net trained file (NetMNIST.pth), and data set (subdir data)
have to be located in the same dir.
Button <Response> : push after drawing pattern to have result
Button <Mod.Draw> : mode draw ; if pushed it becomes <Mod.Canc> to correct pattern
Button <New pattern> : blanks canvas
Button <Next sample> : load and decode next example from dataset

"""


import sys
import os
import pathlib as pat
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
from PIL import Image, ImageQt

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import cv2 as cv

NetMT: nn.Module
test_loader: DataLoader
idata = -1

def loadNet():
    global NetMT
    global test_loader
    osys = os.name
    dir = os.path.abspath(__file__)
    dir = os.path.dirname(dir)
    if osys == 'nt':
        netfile = pat.WindowsPath(dir+os.sep+'NetMNIST.pth')
        datadir = pat.WindowsPath(dir+os.sep+'data')
    else:
        netfile = pat.PosixPath(dir + os.sep + 'NetMNIST.pth')
        datadir = pat.PosixPath(dir + os.sep + 'data')
    print("Load net by file:", netfile)
    print("Load data from:  ", datadir)
    NetMT = torch.load(netfile)
    NetMT.cpu()
    test_loader = DataLoader(
        datasets.MNIST(datadir, train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1,  # 64
        shuffle=False)

def getNextData(test_loader, i= -1):
    test_loader : DataLoader
    global idata
    if (i == -1):
        idata = idata+1
        img, target = test_loader.dataset.__getitem__(idata)
    else:
        idata = i
        img, target = test_loader.dataset.__getitem__(idata)
    return img, target

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        size = QtCore.QSize(500, 360)
        self.resize(size)
        self.setWindowTitle("MNIST graph - draw pattern or load data")
        self.setMinimumSize(QtCore.QSize(size))
        self.setMaximumSize(QtCore.QSize(size))
        self.move(100,100)

        self.frame = QtWidgets.QLabel(self)
        self.frame.setCursor(Qt.CrossCursor)

        canvas = QtGui.QPixmap(280, 280)
        canvas.fill(QtGui.QColor(0, 0, 0))
        self.frame.setPixmap(canvas)
        self.pen = QtGui.QPen()
        self.pen.setColor(QtGui.QColor(255, 255, 255))
        self.pen.setWidth(16)
        self.frame.setGeometry(QtCore.QRect(20, 40, 280, 280))

        self.pB1 = QtWidgets.QPushButton(self)
        self.pB1.setGeometry(QtCore.QRect(360, 40, 93, 28))
        self.pB1.setText('Response')
        self.pB1.clicked.connect(self.click1)

        self.labt = QtWidgets.QLabel(self)
        self.labt.setText("...Result...")
        self.labt.setGeometry(370, 80, 100, 30)

        self.result = QtWidgets.QTextEdit(self)
        self.result.setGeometry(QtCore.QRect(360, 110, 93, 87))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.result.setFont(font)

        self.pB2 = QtWidgets.QPushButton(self)
        self.pB2.setGeometry(QtCore.QRect(360, 250, 93, 28))
        self.pB2.setText('New pattern')
        self.pB2.clicked.connect(self.click2)

        self.pB3 = QtWidgets.QPushButton(self)
        self.pB3.setGeometry(QtCore.QRect(360, 210, 93, 28))
        self.pB3.setText('Mod. Draw')
        self.pB3.clicked.connect(self.click3)

        self.pB4 = QtWidgets.QPushButton(self)
        self.pB4.setGeometry(QtCore.QRect(360, 290, 93, 28))
        self.pB4.setText('Next sample')
        self.pB4.clicked.connect(self.click4)

        self.fdraw = True

    def qimgtopimg(self, pixm):
        pixm = pixm.scaled(28, 28)
        qimg = pixm.toImage()
        pimg = Image.new('L', (28, 28))
        col = QtGui.QColor()
        for x in range(28):
            for y in range(28):
                col = qimg.pixelColor(x, y)
                pimg.putpixel((x, y), col.red())
        #pimg.show()
        pimg = TF.to_tensor(pimg)
        pimg = TF.normalize(pimg, (0.1307,), (0.3081,))
        pimg = pimg.reshape(1, 1, 28, 28)
        out = NetMT(pimg)
        val = int(np.argmax(out.data[0]))
        self.result.setText(str(val))

    def click1(self):
        pix = self.frame.pixmap()
        self.qimgtopimg(pix)

    def click2(self):
        self.frame.pixmap().fill(QtGui.QColor(0, 0, 0))
        self.result.setText("")
        self.update()

    def click3(self):
        if self.fdraw:
            self.pen.setColor(QtGui.QColor(0, 0, 0))
            self.fdraw = False
            self.pB3.setText('Mod. Canc')
        else:
            self.pen.setColor(QtGui.QColor(255, 255, 255))
            self.fdraw = True
            self.pB3.setText('Mod. Draw')

    def click4(self):
        img, result = getNextData(test_loader)
        img = img.reshape(1, 1, 28, 28)
        out = NetMT(img)
        val = int(np.argmax(out.data[0]))
        self.result.setText(str(val))
        arr = np.dot(np.uint8(np.round(img[0][0])), 80)
        arri = np.asarray(arr, dtype='uint8')
        arri = cv.resize(arri, (280, 280))
        cvimg = Image.fromarray(arri, 'L')
        #cvimg = cvimg.resize((280, 280))
        cvimg = cvimg.convert('RGB')
        qimg = ImageQt.ImageQt(cvimg)
        piximg = QtGui.QPixmap.fromImage(qimg)
        self.frame.setPixmap(piximg)
        self.update()

    def mouseMoveEvent(self, e):
        painter = QtGui.QPainter(self.frame.pixmap())
        painter.setPen(self.pen)
        painter.drawPoint(e.x()-20, e.y()-40)
        painter.end()
        self.update()


if __name__ == "__main__":
    loadNet()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()