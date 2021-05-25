'''
Yolo deep learning objects detect using OpenCV as backend.
This program is dedicated to verify yolo model on photo. (yolo model default: yolo4)
OpenCV load network, adapt image from file to frame dimension used for training and detect objects using labels listed in
dedicated file.
You can change model (network structure). Weights of pre-trained model are defined by a file with similar name.
Labels file is coco.names except for yolov3-tiny-voc model (in this case is voc.names)
If you don't insert name photo-file as running parameter --photo (Ex. --photo mypicture.jpg) the program start a chooser
file dialog.
Use -h parameter to display help.
'''
########################################################################

import numpy as np
import argparse
import time
import cv2 as cv
import os, sys
from tkinter import filedialog as fd, Tk
from tkinter import messagebox as msg

thisdir = os.path.dirname(os.path.abspath(sys.argv[0]))     # directory of script

labels = "coco.names" #default labels file
netname = "yolov4"        #default yolo model
#netname = "yolov4-tiny"

root = "D:\\"    #Windows
#root = "/"        #Linux
prepath = os.path.join(root, "AIData")        # ex. windows
#prepath = os.path.join(root, "home", "jetson", "AIData")     # ex. linux

photoname = None
filephoto = None

wdt = 416
hgt = 416
confThreshold = 0.5
camn = 0
CUDA = True

def verifyFile(name):
    if not os.path.exists(name):
        print("File non found: ", name)
        exit(-1)
    else:
        return name

def verifycuda():                           # verify if CUDA is require and if it is available
    global CUDA
    info = cv.getBuildInformation()
    if info.count(" CUDA") > 0:
        print("CUDA present! ", end='')
        print(cv.cuda_DeviceInfo())
        iscuda = True
    else:
        print("CUDA not present!")
        iscuda = False
    if CUDA and iscuda:
        CUDA = True
    else:
        CUDA = False

def init():
    global flabels, photoname, net, confThreshold, filephoto, fwgt, fmodel, linet, textd
    parse = argparse.ArgumentParser(description="OpenCV-YOLO program")
    parse.add_argument('--model', default=netname, help='net model path (def: '+netname+')')
    parse.add_argument('--photo', default=photoname, help='photo file path (def: '+str(photoname)+')')
    parse.add_argument('--thresh', type=int, default= confThreshold,help='confidence threshold (def: '+str(confThreshold)+')')
    parse.add_argument('--linethick', type=int, default= 1,help='thickess of rectangle line (def: 1)')
    parse.add_argument('--textdim', type=float, default= 0.2, help='label text dimension (def: 0.2)')
    parse.add_argument('-?', action='help', help='Help as -h')
    par = parse.parse_args()
    confThreshold = par.thresh
    linet = par.linethick
    textd = par.textdim
    photoname = par.photo
    if photoname == None:
        win = Tk()
        win.withdraw()
        file = fd.askopenfile(title="Choose file", filetypes=[("Image file", (".jpg"))])
        if file == None:
            exit(0)
        else:
            filephoto = file.name
    else:
        filephoto = os.path.join(prepath, 'photo', photoname)
    if not os.path.exists(filephoto):
        print("File photo not found: ", filephoto)
        exit(0)
    verifycuda()
    fmodel = verifyFile(os.path.join(prepath, "darknet", "cfg", netname) + ".cfg")
    fwgt = verifyFile(os.path.join(prepath, "darknet", netname) + ".weights")
    flabels = verifyFile(os.path.join(prepath, "darknet", labels))
    print("Parameters used:")
    print("  Model file: ", fmodel)
    print("  Labels:     ", flabels)
    print("  Weigts:     ", fwgt)
    print("  Photo:      ", filephoto)
    print("  Frame size: ", wdt,"x",hgt)
    print("  Threshold:  ", confThreshold)
    print("  Use CUDA:   ", CUDA)


######## start

def netPrepare():
    global  fwgt, fmodel, nmsThreshold, classes, outNames, net, backend, target
    nmsThreshold = 0.4
    if CUDA:
        backend = cv.dnn.DNN_BACKEND_CUDA
        target = cv.dnn.DNN_TARGET_CUDA
    else:
        backend = cv.dnn.DNN_BACKEND_OPENCV
        target = cv.dnn.DNN_TARGET_CPU

    classes = None
    if flabels:
        with open(flabels, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

    net = cv.dnn.readNet(fwgt, fmodel, 'darknet')
    net.setPreferableBackend(backend)
    net.setPreferableTarget(target)
    outNames = net.getUnconnectedOutLayersNames()

#########
def preprocess(frame):
    # Create a 4D blob from a frame.
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    inpWidth = wdt if wdt else frameWidth
    inpHeight = hgt if hgt else frameHeight
    blob = cv.dnn.blobFromImage(frame, size=(inpWidth, inpHeight), swapRB=True, ddepth=cv.CV_8U)
    net.setInput(blob, scalefactor=0.00392, mean=1)
#########

#########
def drawPred(frame, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), linet)
    label = '%.2f' % conf
    # Print a label of class.
    if classes:
        assert (classId < len(classes))
        label = '%s: %s' % (classes[classId], label)
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, textd, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, textd, (0, 0, 0))
##########

##########
def postprocess(frame, outs):
    global mess
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    layerNames = net.getLayerNames()
    lastLayerId = net.getLayerId(layerNames[-1])
    lastLayer = net.getLayer(lastLayerId)
    classIds = []
    confidences = []
    boxes = []
    #if lastLayer.type == 'Region':
    # Network produces output blob with a shape NxC where N is a number of
    # detected objects and C is a number of classes + 4 where the first 4
    # numbers are [center_x, center_y, width, height]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    # NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    # or NMS is required if number of outputs > 1
    if len(outNames) > 1 or lastLayer.type == 'Region' and backend != cv.dnn.DNN_BACKEND_OPENCV:
        indices = []
        classIds = np.array(classIds)
        confidences = np.array(confidences)
        boxes = np.array(boxes)
        unique_classes = set(classIds)
        for cl in unique_classes:
            class_indices = np.where(classIds == cl)[0]
            conf = confidences[class_indices]
            box = boxes[class_indices].tolist()
            nms_indices = cv.dnn.NMSBoxes(box, conf, confThreshold, nmsThreshold)
            nms_indices = nms_indices[:, 0] if len(nms_indices) else []
            indices.extend(class_indices[nms_indices])
    else:
        indices = np.arange(0, len(classIds))
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
    mess ="Total objects detected: "+ str(len(indices))+"\n"
    objs = list()
    confs = list()
    for i in indices:
        cid = classIds[i]
        conf = confidences[i]
        objs.append(cid)
        confs.append(conf)
    otype = set(objs)
    for t in otype:
        n = 0
        confmin = 1
        confmax = 0
        for i in range(len(objs)):
            if t == objs[i]:
                n = n + 1
                confmin = min(confmin, confs[i])
                confmax = max(confmax, confs[i])
        mess= mess + "%i %s min.conf. %3.2f max.conf. %3.2f \n"%(n, classes[t], confmin, confmax)
    print(mess)
    #############

def readphoto(fileph):
    frame = cv.imread(fileph)
    return frame

################################### Main ##########################################
if __name__ == "__main__":
    init()
    netPrepare()
    frun = True
    while frun:
        frame = readphoto(filephoto)
        preprocess(frame)
        outs = net.forward(outNames)
        postprocess(frame, outs)
        w = int(frame.shape[1] * 1.5)
        h = int(frame.shape[0] * 1.5)
        frame = cv.resize(frame, (w, h))
        winName = photoname
        cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
        cv.imshow(winName, frame)
        msg.showinfo(message=mess)
        win = Tk()
        win.withdraw()
        file = fd.askopenfile(title="Choose file", filetypes=[("Image file", (".jpg"))])
        if file == None:
            exit(0)
        else:
            filephoto=file.name



