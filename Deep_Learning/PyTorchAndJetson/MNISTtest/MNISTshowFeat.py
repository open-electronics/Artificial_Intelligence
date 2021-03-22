"""
This program shows feature map (result of convolution layer + maxpool2d function, for each filter)
You can show result of first convolution phase (20 features of 12x12) or the result after second convolution
phase (50 features 4x4 ). See help.

"""

import argparse
import os
import pathlib as pat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2 as cv
from torchvision import datasets, transforms


NetMT : nn.Module
test_loader : DataLoader
filenamenet = 'NetMNIST.pth'
conv = 1

def loadNetAndData():
    global NetMT
    global test_loader
    osys = os.name
    dir = os.path.abspath(__file__)
    dir = os.path.dirname(dir)
    if osys == 'nt':
        netfile = pat.WindowsPath(dir+os.sep + filenamenet)
        datadir = pat.WindowsPath(dir+os.sep + 'data')
    else:
        netfile = pat.PosixPath(dir + os.sep + filenamenet)
        datadir = pat.PosixPath(dir + os.sep + 'data')
    print("Load net by file:", netfile)
    print("Load data from:  ", datadir)
    NetMT = torch.load(netfile)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(datadir, train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1,  # 1 or 64(def)
        shuffle=False)


idata = -1

def imgdata(data):
    arr = np.dot(np.uint8(np.round(data)), 80)
    cvimg = np.asarray(arr, dtype='uint8')
    cvimg = cv.resize(cvimg, (280, 280))
    return cvimg


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


def imgFiltered(tout, nfilter, k, scale):
    arr = np.dot(np.uint8(np.round(tout.data[0][nfilter])), 80)
    tot = arr.ptp() / arr.std()
    imout = np.asarray(arr, dtype='uint8')
    dim = k * scale
    imout = cv.resize(imout, (dim, dim))
    return imout, tot


def MNISTimgFiltered1(net, img, scale):
    net : nn.Module
    img = img.reshape(1, 1, 28, 28)
    lnet = nn.ModuleList(net.children())
    mod = lnet[0]  #24x24
    out = mod(img)
    out = F.max_pool2d(out, kernel_size=2, stride=2) #12x12
    imd = 12*scale
    win = np.ones((4 * (imd + 5) + 5, 5 * (imd + 5) + 5), dtype='uint8') # 5 rows 4 col
    maximg = 0
    cmax = 0
    rmax = 0
    for y in range(4):
        py = y * (imd + 5) + 5
        for x in range(5):
            px = x * (imd + 5) + 5
            im, tot = imgFiltered(out, y * 5 + x, 12, scale)
            win[py:py + imd, px:px + imd] = im
            if tot > maximg:
                maximg = max(maximg, tot)
                cmax = x
                rmax = y
    px = cmax * (imd+5) + 2
    py = rmax * (imd+5) + 2
    point1 = (px + 5, py + 5)
    point2 = (px + imd +2, py + imd +2)
    cv.rectangle(win, point1, point2, 250, 2)
    return win, cmax, rmax, maximg

def MNISTimgFiltered2(net, img, scale):
    net : nn.Module
    img = img.reshape(1, 1, 28, 28)
    lnet = nn.ModuleList(net.children())
    mod = lnet[0]  #24x24
    out = mod(img)
    out = F.max_pool2d(out, kernel_size=2, stride=2) #12x12
    mod = lnet[1]  #8x8
    out = mod(out)
    out = F.max_pool2d(out, kernel_size=2, stride=2) #4x4
    imd = 4*scale
    win = np.ones((5 * (imd + 5) + 5, 10 * (imd + 5) + 5), dtype='uint8') # 5 rows 10 col
    maximg = 0
    cmax = 0
    rmax = 0
    for y in range(5):
        py = y * (imd + 5) + 5
        for x in range(10):
            px = x * (imd + 5) + 5
            im, tot = imgFiltered(out, y * 10 + x, 4, scale)
            win[py:py + imd, px:px + imd] = im
            if tot > maximg:
                maximg = max(maximg, tot)
                cmax = x
                rmax = y
    px = cmax * (imd+5) + 2
    py = rmax * (imd+5) + 2
    point1 = (px + 5, py + 5)
    point2 = (px + imd +2, py + imd +2)
    cv.rectangle(win, point1, point2, 250, 2)
    return win, cmax, rmax, maximg


def params():
    parse = argparse.ArgumentParser(description="Show feature map for MNIST net")
    parse.add_argument('-?', action='help', help='as -h, --help')
    parse.add_argument('-filenet', default='NetMNIST.pth', help='file net trained (def. NetMNIST.pth)')
    parse.add_argument('-featlevel', type=int, choices=[1, 2], default=1, help='conv 1 or 2 (def. 1)')
    par = parse.parse_args()
    global filenamenet, conv
    filenamenet = par.filenet
    conv = par.featlevel

def loop():
    while True:
        k = input("Next(just return) or sample number (0 to 9999) (-1 to end) : > ")
        if len(k) > 0:
            k = int(k)
            if k < 0: break
            data, tg = getNextData(test_loader, k)
        else:
            data, tg = getNextData(test_loader)
        if conv == 1:
            imgfiltered, c, r, m = MNISTimgFiltered1(NetMT, data, 8)
        else:
            imgfiltered, c, r, m = MNISTimgFiltered2(NetMT, data, 20)
        cv.imshow("Filtered", imgfiltered)
        cv.waitKey(100)


if __name__ == "__main__":
    params()
    loadNetAndData()
    print("Show feature map using net: ", filenamenet)
    loop()

"""
for i in range(100):
    k = input("Next or n ")
    if k.isnumeric():
        k = int(k)
        data, tg = getNextData(test_loader, k)
    else:
        data, tg = getNextData(test_loader)
    print(tg)
    #img = imgdata(data[0])
    #cv.imshow("Data "+str(tg), img)
    imgfiltered, c, r, m = MNISTimgFiltered1(NetMT, data, 8)
    print("Max correspondence: row= ", r, " col= ", c, " val= ", m)
    cv.imshow("Filtered", imgfiltered)
    cv.waitKey(1000)

"""