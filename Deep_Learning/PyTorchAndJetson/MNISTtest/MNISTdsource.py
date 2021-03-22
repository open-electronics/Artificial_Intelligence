"""
This script implements the getsource() function. This script is imported by utility TorchToOnnx.py.
The utility uses the getsource() function to input just a single input data sample for elaborating
the complete net structure so that utility can transform net to onnx format correctly.
This script is for MNIST net only.
Make a similar script for other net or data. You have just to implements getsource() function
that returns a torch.utils.data.DataLoader object.

"""


import os
import pathlib as pat
import torch
from torchvision import datasets, transforms


def getsource(bsize=1):
    osys = os.name
    dir = os.path.abspath(__file__)
    dir = os.path.dirname(dir)
    if osys == 'nt':
        datadir = pat.WindowsPath(dir + os.sep + 'data')
    else:
        datadir = pat.PosixPath(dir + os.sep + 'data')
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(datadir, train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=bsize,  # 1 or more
        shuffle=False)
    return test_loader