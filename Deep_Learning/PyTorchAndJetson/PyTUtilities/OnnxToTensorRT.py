"""
This utility transforms NN file ONNX (.onnx) to a file TensorRT (.rt)
This utility needs just a file *.onnx and generates a file with same name but with extension .rt
NB. the onnx file must elaborate just a sample at time.
In other words it has to be configured with batch_size = 1

Ex.:
   py OnnxToTensorRT.py  NetMNIST.onnx

"""


import os
import argparse
import torch
import tensorrt as trt

filenamenet = ''
filert = ''
netfile = ''

def files():
    global filenamenet, filert, netfile
    dir = os.path.abspath(__file__)
    dir = os.path.dirname(dir)
    file = filenamenet.rsplit(sep='.', maxsplit=1)
    filert = file[0]+".rt"


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def buildEngine(netPath):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(netPath, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
        engine = builder.build_cuda_engine(network)
    return network, engine


def params():
    parse = argparse.ArgumentParser(description="torch net file (.pth) to onnx net file (same name + .onnx)")
    parse.add_argument('-?', action='help', help='as -h, --help')
    parse.add_argument('fileONNX', help='file path  .onnx transformed to .rt')
    par = parse.parse_args()
    global filenamenet
    filenamenet = par.fileONNX


if __name__ == "__main__":
    params()
    network, engine = buildEngine(filenamenet)
    with open(filert, 'wb') as f:
        f.write(engine.serialize())
    print(filenamenet, " tranformed in ", filert)