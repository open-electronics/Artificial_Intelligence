'''
Yolo object detect using OpenCV as backend
OpenCV load network, adapt image from camera to frame used for training and detect objects using labels listed in
dedicated file.
You can change model (network structure). Weights of pre-trained model are defined by a file with similar name.
Labels file is coco.names except for yolov3-tiny-voc model (in this case is voc.names)
Use -h parameter to disply help.
'''
########################################################################

import numpy as np
import argparse
import time
import cv2 as cv
import os, sys


thisdir = os.path.dirname(os.path.abspath(sys.argv[0]))     # directory of script

labels = "coco.names"
net = "yolov4-tiny"

root = "D:\\"    #Windows
#root = "/"        #Linux
prepath = os.path.join(root, "PythonPrg", "PyCharm")        # ex. windows
#prepath = os.path.join(root, "home", "jetson", "AIData")     # ex. linux


wdt = 416
hgt = 416
confThreshold = 0.3
camn = 0
CUDA = True
windim = 'L'
wwdt = 640
wwht = 480

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

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="OpenCV-YOLO program")
    parse.add_argument('--model', default=net, help='net model path (def: '+net+')')
    parse.add_argument('--labels', default=labels , help='path of net object classes (def: '+labels+')')
    parse.add_argument('--thresh', type=int, default= confThreshold,help='confidence threshold (def: '+str(confThreshold)+')')
    parse.add_argument('--frd', type=int, nargs=2, default=[wdt, hgt], help='samples frame dimension (def: '+str(wdt)+' '+str(hgt)+')')
    parse.add_argument('--camera', type=int, default=camn, help='camera number (def: '+str(camn)+')')
    parse.add_argument('--wdim', default=windim, help='win dimension (L:640x480, S:320x240) (def:'+str(windim)+')')
    parse.add_argument('--nocuda', action='store_true', help="don't use CUDA (def. use CUDA if present)")
    parse.add_argument('-?', action='help', help='Help as -h')
    par = parse.parse_args()
    wdt = par.frd[0]
    hgt = par.frd[1]
    confThreshold = par.thresh
    camn = par.camera
    if par.nocuda:
        CUDA = False
    if par.wdim == 'S':
        wwdt= 320
        wwht= 240

    verifycuda()
    fmodel = verifyFile(os.path.join(prepath, "darknet", "cfg", net) + ".cfg")
    fwgt = verifyFile(os.path.join(prepath, "darknet", net) + ".weights")
    flabels = verifyFile(os.path.join(prepath, "darknet", labels))

    print("Parameters used:")
    print("  Model file: ", fmodel)
    print("  Labels:     ", flabels)
    print("  Weigts:     ", fwgt)
    print("  Frame size: ", wdt,"x",hgt)
    print("  Threshold:  ", confThreshold)
    print("  Camera num: ", camn)
    print("  Use CUDA:   ", CUDA)
    print("  Window size:", wwdt, "x", wwht)


#cap = cv.VideoCapture(camn, cv.CAP_V4L2)
cap = cv.VideoCapture(camn, cv.CAP_ANY)
if not cap.isOpened():
    print("No webcam number ", camn)
    exit(-1)
cap.set(cv.CAP_PROP_FRAME_WIDTH, wwdt)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, wwht)
cap.set(cv.CAP_PROP_BUFFERSIZE, 10)

######## start

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

#print(classes)

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
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))
    label = '%.2f' % conf
    # Print a label of class.
    if classes:
        assert (classId < len(classes))
        label = '%s: %s' % (classes[classId], label)
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255),
        cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
##########
##########
def postprocess(frame, outs):
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
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        unique_classes = set(classIds)
        for cl in unique_classes:
            class_indices = np.where(classIds == cl)[0]
            conf = confidences[class_indices]
            box  = boxes[class_indices].tolist()
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
#############


winName = 'OpenCV-YOLO'
cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)

cf = 0
T0 = time.perf_counter()
fk = -1
fw = 1
while fk < 0 and fw > 0:
    ok, frame = cap.read()
    if not ok:
        print("No frame!")
        break
    preprocess(frame)
    outs = net.forward(outNames)
    postprocess(frame, outs)
    cv.imshow(winName, frame)
    fk = cv.waitKey(10)
    fw = cv.getWindowProperty(winName, cv.WND_PROP_VISIBLE)
    cf = cf+1
    if cf >= 100:
        cf = 0
        DT = time.perf_counter()-T0
        T0 = time.perf_counter()
        print("Frame/s : {:6.6f}".format(100/DT))
cv.destroyAllWindows()
exit()

# Jetson-CUDA F/s 7.5 (no show ssh)