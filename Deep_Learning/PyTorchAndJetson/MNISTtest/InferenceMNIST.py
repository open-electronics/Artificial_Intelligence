"""
Test inference speed of convolution NN for MNIST digit detect.
Network model file is MNISTmodel.py. Network trained is loaded as NetMNIST.pth,
but model file has to be in the same directory.
Program use: py InferenceMNIST.py [-h][-bsize=...][-device=...]
bsize is the batch size utilized (that is the number of example loaded at time)
device is CPU or CUDA (if available)
Data are in data sub-dir

Test on Windows core i7 (cpu with 4 cores 8 threads) mem: 16GB
Test set 10000 samples: Average loss: 0.0342, Accuracy: 9887/10000 (98.9%) (batch=64 (def.))
Time: 4.940582
But with batch=1
Time: 13.503630

Test on Jetson
CPU
Batch =64 Time: 34
Batch =1  Time: 104

CUDA
Batch =64 Time: 20
Batch =1  Time: 81

"""


import os
import argparse
import time
import pathlib as pat
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch

NetMT: torch.nn.Module
test_loader: torch.utils.data.DataLoader
filenamenet = 'NetMNIST.pth'

bsize = 64
device = 'cpu'

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
        batch_size=bsize,  # 1 or 64(def)
        shuffle=False)

ndata = 0

def test():
    global ndata
    NetMT.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        nd = data.shape[0]
        ndata = ndata + nd
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            data = torch.as_tensor(data, device=device)
        output = NetMT(data)
        test_loss += F.nll_loss(output, target).data.item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(test_loss, correct,
                                                                                 len(test_loader.dataset),
                                                                                 100. * correct / len(
                                                                                     test_loader.dataset)))


def params():
    parse = argparse.ArgumentParser(description="Speed test for MNIST inference (10000 samples)")
    parse.add_argument('-?', action='help', help='as -h, --help')
    parse.add_argument('-bsize', type=int, default=64, help='Batch size (loaded samples) (def. 64)')
    parse.add_argument('-device', default='CPU', choices=['cpu', 'cuda'], help='CPU or CUDA (def. CPU)')
    parse.add_argument('-filenet', default='NetMNIST.pth', help='file net trained (def. NetMNIST.pth)')
    par = parse.parse_args()
    global bsize, device, filenamenet
    bsize = par.bsize
    device = par.device
    filenamenet = par.filenet

def setdevice():
    global NetMT, device
    if device == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.device('cuda')
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            NetMT = NetMT.cuda()
        else:
            device = 'cpu'
            print('CUDA not available! CPU used instead!')
    if device == 'cpu':
        torch.device('cpu')
        torch.set_default_tensor_type(torch.FloatTensor)
        NetMT = NetMT.cpu()



if __name__ == "__main__":
    params()
    loadNetAndData()
    setdevice()
    ns = test_loader.dataset.__len__()
    print("Start computation of", ns, "samples... (device: ", device, " batch_size: ", bsize, ")")
    T0 = time.perf_counter()
    test()
    T1 = time.perf_counter()
    print("Time: {:6.6f}  for {:d} samples".format(T1-T0, ndata))



