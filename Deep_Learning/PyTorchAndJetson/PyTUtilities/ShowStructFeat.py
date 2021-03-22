"""
Functions library for printing or visualizing network structure and convolution filters
Running as program it can print (or save) structure or convolution filter. See help.

"""


import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import argparse
import sys
import math

"""
  If outf is not provided in functions calling, functions don't print anythings but just returns values
  Es. struct(net) just returns state_dict
"""
def struct(net, outf = None):                   # Print structure (layers with names) of a net loaded or created
    net: nn.Module
    listm = list(net.named_children())
    if outf is not None:
        if outf is not 'stdout':
            outf = open(outf, 'w')
        else: outf = sys.stdout
    for  m in listm:
        if outf is not None: print(listm.index(m), "   ", m[0], " ", m[1:], file=outf, flush=True)
    return net.state_dict()


def getParams(net, outf = None):                 # Print parameters (weights) shapes and returns total param. number
    net: nn.Module
    param = net.named_parameters()
    totnd = 0
    if outf is not None:
        if outf is not 'stdout':
            outf = open(outf, 'w')
        else: outf = sys.stdout
    for name, p in param:
        np = 1
        for i in range(p.dim()): np = np * p.shape[i]
        if outf is not None: print(name, " ", p.shape, " Tot:", np, file=outf, flush=True)
        totnd = totnd + np
    return totnd


def getWeights(net, layername, outf = None):     # Returns weights array of a layer (named)
    net: nn.Module
    param = net.named_parameters()
    pname = layername+".weight"
    if outf is not None:
        if outf is not 'stdout':
            outf = open(outf, 'w')
        else: outf = sys.stdout
    for name, p in param:
        if name == pname:
            if outf is not None: print(p.shape, "\n", p, file=outf, flush=True)
            return p
        else:
            p = None
            continue
    return p


def getBias(net, layername, outf = None):        # Returns bias array of a layer (named)
    net: nn.Module
    param = net.named_parameters()
    pname = layername+".bias"
    if outf is not None:
        if outf is not 'stdout':
            outf = open(outf, 'w')
        else: outf = sys.stdout
    for name, p in param:
        if name == pname:
            if outf is not None: print(p.shape, "\n", p, file=outf, flush=True)
            return p
        else:
            p = None
            continue
    return p


def featureimg(cnvweigh, nfilter):  # Returns an image of a conv. filter of a single colour chan.
    warr = np.asarray(cnvweigh.data[nfilter][0])
    warr = warr - warr.min()
    warr = (warr / warr.max()) * 255
    warr = np.array(warr, dtype='uint8')
    cvimg = np.asarray(warr, dtype='uint8')
    return cvimg


def showFeature(cnvweigh, nfilter, scale):  # shows a window with feature map of a filter (one colour)
    cvimg = featureimg(cnvweigh, nfilter)
    cvimg = cv.resize(cvimg, (scale, scale), interpolation=cv.INTER_NEAREST)
    cv.imshow("F "+str(nfilter), cvimg)
    cv.waitKey(0)


def showfmap(weights, col, row, kernel, sc):  # returns an image of all filters of a conv. layer
    kd= kernel*sc                             # filters organized by columns and rows; kernel is the
    win = np.ones((row*(kd+5)+5, col*(kd+5)+5), dtype='uint8')  # dimension of kernel (3 for 3x3, 5 for 5x5)
    for y in range(row):                                        # sc is the scale factor of kernel visualization
        py = y*(kd+5)+5
        for x in range(col):
            px = x*(kd+5)+5
            #f = np.ones((kd, kd), dtype='uint8')*(x+y)*30
            f = featureimg(weights, y*col+x)
            #f = cv.resize(f, (kd, kd), interpolation=cv.INTER_NEAREST)
            f = cv.resize(f, (kd, kd), interpolation=cv.INTER_LINEAR)
            #f = cv.resize(f, (kd, kd), interpolation=cv.INTER_CUBIC)
            win[py:py + kd, px:px + kd] = f
    return win

def getLayer(net, lname):
    layers = net.named_children()
    module : nn.Module = None
    for name, module in layers:
        if name == lname:
            return module
    return None

# ******* Test functions ******************************************************************** #

def params():
    parse = argparse.ArgumentParser(description="Library function test (print function are mutually exclusive")
    parse.add_argument('-?', action='help', help='as -h, --help')
    parse.add_argument('-filenet', default='NetMNIST.pth', help='file net trained (def. NetMNIST.pth)')
    parse.add_argument('-pstruct', action='count', help='print complete structure with names (no value)')
    parse.add_argument('-showfmap', action='count', help='show feat. map (filters) of a layer')
    parse.add_argument('-layername', default='', help='layer name for filter map (def= None)')
    parse.add_argument('-outfile', default='stdout', help='output file (def. stdout)')
    par = parse.parse_args()
    if len(sys.argv) < 2:
        parse.print_help()
        exit()
    fnet = par.filenet
    prfile = par.outfile
    net = torch.load(fnet)
    if par.pstruct is not None:
        print("Printing structure: ")
        struct(net, outf=prfile)
        exit()
    if par.showfmap is not None:
        if par.layername == '':
            print("Please define layer name! (-layername=....)")
            exit()
        layer: nn.Module = getLayer(net, par.layername)
        if layer is None:
            print("No such layer name ", par.layername)
            exit(0)
        if type(layer) is not nn.Conv2d:
            print("No convolution layer ", par.layername, "  ", type(layer))
            exit(0)
        wgt = getWeights(net, par.layername)
        fdim = wgt.shape[0]
        chan = wgt.shape[1]
        kern = wgt.shape[2]
        row = int(math.sqrt(fdim))
        col = int(fdim / row)
        sc = int(100 / col)
        pic = showfmap(wgt, col, row, kern, sc)
        if prfile == 'stdout':
            cv.imshow("Filters", pic)
            cv.waitKey(0)
            exit(0)
        else:
            cv.imwrite(prfile, pic)
            exit(0)



if __name__ == "__main__":
    params()






