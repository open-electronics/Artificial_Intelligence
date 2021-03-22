"""
Train a convolution net model for MNIST (file model: MNISTmodel.py)
You can start with a new random net or you can continue the training loading the previous saved net
You can choose the number of epochs (def. 1)
You can also change the name of saved net (def. NetSavedMT.pth)
The program uses data located in data sub-directory
You can choose to use CPU or CUDA (if available)

"""


import sys, os
import pathlib as pat
import argparse
import time
import torch
from torchvision import datasets, transforms
import MNISTmodel as model

NetMT: torch.nn.Module
train_loader: torch.utils.data.DataLoader
test_loader: torch.utils.data.DataLoader
filenet = 'NEW'
filesave = 'NetSavedMT.pth'

batch_size = 64
test_batch_size = 100
epochs = 1



def loadNetAndData():
    global train_loader, test_loader, filenet, NetMT, filesave
    osys = os.name
    dir = os.path.abspath(__file__)
    dir = os.path.dirname(dir)
    if osys == 'nt':
        netfile = pat.WindowsPath(dir + os.sep + filenet)
        datadir = pat.WindowsPath(dir + os.sep + 'data')
        filesave = pat.WindowsPath(dir + os.sep + filesave)
    else:
        netfile = pat.PosixPath(dir + os.sep + filenet)
        datadir = pat.PosixPath(dir + os.sep + 'data')
        filesave = pat.PosixPath(dir + os.sep + filesave)

    if filenet == 'NEW':
        NetMT = model.Net()
        print("New MNIST random net")
    else:
        NetMT = torch.load(netfile)
        print("Loaded net from : ",netfile)

    print("Load data from:  ", datadir)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(datadir, train=True, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(datadir, train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size,
        shuffle=True)


def params():
    parse = argparse.ArgumentParser(description="Training net model for MNIST ")
    parse.add_argument('-?', action='help', help='as -h, --help')
    parse.add_argument('-device', default='CPU', choices=['CPU', 'CUDA'], help='CPU or CUDA (def. CPU)')
    parse.add_argument('-fnetload', default='NEW',  help='net filename to continue, or new net NEW (def: NEW)')
    parse.add_argument('-fnetsave', default='NetSavedMT.pth', help='file net trained (def. NetSavedMT.pth)')
    parse.add_argument('-epochs', type=int, default=1, help='number of train epochs (def. 1)')
    par = parse.parse_args()
    global device, filenet, filesave, epochs
    device = par.device.upper()
    filenet = par.fnetload
    filesave = par.fnetsave
    epochs = par.epochs

def setdevice():
    global NetMT, device
    if device == 'CUDA':
        if torch.cuda.is_available():
            torch.device('cuda')
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            NetMT = NetMT.cuda()
        else:
            device = 'CPU'
            print('CUDA not available! CPU used instead!')
    if device == 'CPU':
        torch.device('cpu')
        torch.set_default_tensor_type(torch.FloatTensor)
        NetMT = NetMT.cpu()



if __name__ == "__main__":
    params()
    loadNetAndData()
    setdevice()
    ManageMT = model.MnistManagement(NetMT, train_loader, test_loader)
    ns = train_loader.dataset.__len__()
    print("Start computation of", ns, "samples for ", epochs, "epochs (device: ", device, ")")
    T0 = time.perf_counter()
    ManageMT.learn(epochs)
    T1 = time.perf_counter()
    print("Time: {:6.6f} ".format(T1 - T0))
    torch.save(NetMT, filesave)
    print("Net trained saved as ", filesave)
