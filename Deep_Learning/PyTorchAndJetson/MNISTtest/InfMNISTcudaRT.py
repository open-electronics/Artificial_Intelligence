"""
Use this script on Jetson.
This script compute inference MNIST for 10000 test samples
This script loads MNIST trained net from TensorRT format and puts it on CUDA engine.
For TensorRT constrains, script uses batch_size = 1

Test: 58 sec.
"""

import os
import pathlib as pat
import torch
import torch.nn as nn
import numpy as np
import time
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as  cuda
import torch.nn.functional as F
from torchvision import datasets, transforms

filenamenet = 'NetMNIST.rt'
netfile = ''
test_loader = None

def loadNetAndData():
    global netfile
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
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(datadir, train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1,
        shuffle=False)

loadNetAndData()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

engine: trt.ICudaEngine
with open(netfile, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())


# Determine dimensions and create page-locked memory buffers(i.e.won't be swapped to disk) to hold host inputs/outputs.
h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
# Allocate device memory for inputs and outputs.
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
# Create a stream in which to copy inputs/outputs and run inference.
stream = cuda.Stream()

ndata = 0
ns = test_loader.dataset.__len__()
print("Start computation of", ns, "samples... (device: CUDA batch_size: 1)")
T0 = time.perf_counter()
with engine.create_execution_context() as context:
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        ndata = ndata + 1
        # Transfer input data to the GPU.
        h_input=(data.numpy()).copy()
        #cuda.memcpy_htod_async(d_input, data.numpy(), stream)
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # Run inference.
        context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        out = torch.as_tensor(h_output)
        #test_loss += F.nll_loss(out.data, target).data.item()
        #pred = h_output.data.max(1)[1]
        pred = out.data.max(0)[0]
        correct += pred.eq(target.data).cpu().sum()
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(test_loss, correct,
                                                                                 len(test_loader.dataset),
                                                                                 100. * correct / len(
                                                                                     test_loader.dataset)))

T1 = time.perf_counter()
print("End computation.")
print("Time: {:6.6f} for {:d} samples".format(T1-T0, ndata))

#CUDATensorRT: Time: 9.534905


