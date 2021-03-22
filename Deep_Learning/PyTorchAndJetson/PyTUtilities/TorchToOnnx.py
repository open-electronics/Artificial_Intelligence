"""
This utility transforms a pytorch net file to a ONNX file net format.
The file onnx will have same nome but extension .onnx
Unfortunately, because the pytorch net file format, onnx transformation needs to try a forward phase.
So, it needs a data source.
Data source has to be provided using a customized script like MNISTdsource.py where has to be defined
the function getsource() that returns a torch.utils.data.DataLoader object.
batch_size is 64 for default, but it can be changed.
To transform onnx file to TensorRT for using it in Jetson nano (CUDA core) batch_size should be 1,
because TensorRT inference compute one input at time (batch 1)

Ex.:
    py TorchToOnnx.py NetMNIST.pth -datasource=MNISTdsource

"""


import os
import pathlib as pat
import argparse
import torch
import torch.nn as nn
import importlib

Net : nn.Module
test_loader : torch.utils.data.DataLoader
filenamenet = ''
fileonnx = ''
datasource = ''
bsize = 64

def loadNet():
    global Net, filenamenet, fileonnx
    global test_loader
    osys = os.name
    dir = os.path.abspath(__file__)
    dir = os.path.dirname(dir)
    file = filenamenet.split(sep='.')
    fileonnx = file[0]+".onnx"
    if osys == 'nt':
        netfile = pat.WindowsPath(dir+os.sep + filenamenet)
    else:
        netfile = pat.PosixPath(dir + os.sep + filenamenet)
    print("Load net by file:", netfile)
    Net = torch.load(netfile)


def params():
    parse = argparse.ArgumentParser(description="torch net file (.pth) to onnx net file (same name + .onnx)")
    parse.add_argument('-?', action='help', help='as -h, --help')
    parse.add_argument('filenet', help='file net to transform (def. none)')
    parse.add_argument('-datasource', default='', required=True, help='file containing getsource() function (Ex.: MNISTdsource.py)(def. none)')
    parse.add_argument('-bsize', type=int, default=64, help='Batch size (loaded samples) (def. 64)')
    par = parse.parse_args()
    global filenamenet, datasource, bsize
    filenamenet = par.filenet
    datasource = par.datasource
    bsize = par.bsize


if __name__ == "__main__":
    params()
    loadNet()
    source = importlib.import_module(datasource)
    data_loader = source.getsource(bsize)
    print("using data from: ", datasource)
    lload = list(data_loader)
    data, target = lload[0]
    torch.onnx.export(Net, data, fileonnx, verbose=True)
    print(filenamenet, " tranformed in ", fileonnx)