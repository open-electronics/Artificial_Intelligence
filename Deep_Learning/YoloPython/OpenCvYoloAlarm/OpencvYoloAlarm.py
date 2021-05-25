'''
Yolo deep learning objects detect, using OpenCV as backend.
This program acts as an alarm using a camera for identifying presence of persons.
(It is possible to change object alarm from person to one of 80 object classes described in COCO names.)
This program utilize the yolo4 model for the best confidence degree.
Yolo4 is an heavy deep learning network, so it is really slower than yolo4-tiny.
On Jetson Nano yolo4-tiny allows about 8 frame/s and yolo4 just about 1 frame/s.
But, long latency is not really important because human speed action.
Any way, you can change model from yolo4 to yolo4-tiny swapping comment sign.
OpenCV load network, adapt image from file to frame dimension used for training and detect objects using labels listed in
dedicated file.
Labels file is coco.names.
This program is intended for using on Jetson Nano. But you can test it also on Windows swapping comment sign.
Use -h parameter to disply help.
'''

########################################################################

import numpy as np
import time
import cv2 as cv
import os, sys
import TelSenderLib as tel
import Common as sas
import asyncio

#hardware = 'jetson'
hardware = 'windows'

thisdir = os.path.dirname(os.path.abspath(sys.argv[0]))     # directory of script

labels = "coco.names" #default labels file
net = "yolov4"        #default yolo model
#net = "yolov4-tiny"


prepath = thisdir

wdt = 416
hgt = 416
confThreshold = 0.5
camn = 0
CUDA = True
nameobj = 'person'  # alarm if this object is detected
indexobj = 0
fvideo = os.path.join(thisdir, "videodetect.mp4")
falarm = False
cap = None

backend = None
target = None

frun = True

############################################
def verifyFile(name):
    if not os.path.exists(name):
        sas.log.messpr("File non found: "+name)
        exit(-1)
    else:
        return name
#############################################

#############################################
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
###############################################

###############################################
def verifycamera(camn):
    global cap, wwdt, wwht
    wwdt= 640
    wwht= 480
    if hardware == 'jetson' : cap = cv.VideoCapture(camn, cv.CAP_V4L2)
    else : cap = cv.VideoCapture(camn, cv.CAP_ANY)
    if not cap.isOpened():
        sas.log.messpr("No webcam number "+str(camn))
        exit(-1)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, wwdt)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, wwht)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 10)
###############################################

###############################################
def init():
    global flabels, net, confThreshold, camn, fwgt, fmodel, nameobj, indexobj
    nameobj = sas.cfg.getValue(sas.config, 'object')
    confThreshold = sas.cfg.getValueFloat(sas.config, 'threshold')
    camn = sas.cfg.getValueInt(sas.config, 'camera')
    verifycuda()
    net = sas.model
    fmodel = verifyFile(os.path.join(prepath, "darknet", net) + ".cfg")
    fwgt = verifyFile(os.path.join(prepath, "darknet", net) + ".weights")
    flabels = verifyFile(os.path.join(prepath, "darknet", labels))
    verifycamera(camn)
    with open(flabels) as myfile:
        clist = myfile.read().splitlines()
        if not nameobj in clist:
            sas.log.messpr(nameobj + ' not in COCO classes list :')
            for i in range(len(clist)):  print(i, " ", clist[i])
            exit(0)
        else:
            indexobj = clist.index(nameobj)
    tstart = str("\n"+
    "Parameters used:"+
    "  Model file:  "+fmodel+"\n"+
    "  Labels:      "+flabels+"\n"+
    "  Weigts:      "+fwgt+"\n"+
    "  Camera:      "+str(camn)+"\n"+
    "  Frame size:  "+str(wdt)+"x"+str(hgt)+"\n"+
    "  Threshold:   "+str(confThreshold)+"\n"+
    "  Use CUDA:    "+str(CUDA)+"\n"+
    "  Object alarm:"+nameobj+"\n"+
    "  Alarm phone: "+str(sas.alarmstate)+"\n"+
    "  Detection:   "+str(sas.detect)+"\n")
    sas.log.messpr(tstart)

###############################################


###############################################
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
###############################################

###############################################
def preprocess(frame):
    # Create a 4D blob from a frame.
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    inpWidth = wdt if wdt else frameWidth
    inpHeight = hgt if hgt else frameHeight
    blob = cv.dnn.blobFromImage(frame, size=(inpWidth, inpHeight), swapRB=True, ddepth=cv.CV_8U)
    net.setInput(blob, scalefactor=0.00392, mean=1)
###############################################

###############################################
def drawPred(frame, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))
    label = '%.2f' % conf
    # Print a label of class.
    if classes:
        assert (classId < len(classes))
        label = '%s: %s' % (classes[classId], label)
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.3, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255),
        cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))
###############################################

###############################################
def postprocess(frame, outs):
    global mess
    falarm = False
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
        if t == indexobj: falarm = True
        n = 0
        confmin = 1
        confmax = 0
        for i in range(len(objs)):
            if t == objs[i]:
                n = n + 1
                confmin = min(confmin, confs[i])
                confmax = max(confmax, confs[i])
        mess= mess + "%i %s min.conf. %3.2f max.conf. %3.2f \n"%(n, classes[t], confmin, confmax)
 #   print(mess)
    return falarm
###############################################

###############################################
def sendAlarm():
    apiid = sas.telid
    apihash = sas.telash
    botadd = sas.botadd
    netadd = sas.addmess
#    loop = asyncio.new_event_loop()
#    asyncio.set_event_loop(loop)
    if sas.alarmstate and sas.telok:
        try:
            client = tel.telinit(apiid, apihash)
            tel.sendMessage(client, netadd)
            tel.sendMessage(client, mess)
            tel.sendvideo(client, fvideo)
            tel.alarm(client, botadd)
            tel.disconnect(client)
        except Exception as e:
            sas.log.messpr("No phone link available")
            print(str(e))



###############################################

###############################################

state = 0

def alarm(al, frame, timer, tsuspend):
    global state, mt, dt, WrVid
    if not al:
        if state == 0: return
    if state == 0:
        type = cv.VideoWriter_fourcc(*'H2S4')
        WrVid = cv.VideoWriter(fvideo, type, 2, (wwdt, wwht))
        dt = time.perf_counter() + timer
        mt = dt + tsuspend
        sas.log.messpr("Alarm! "+mess)
        state = 1
        return
    if state == 1:
        WrVid.write(frame)
        if time.perf_counter() > dt:
            WrVid.release()
            sendAlarm()
            state = 2
            sas.log.messpr("Suspend alarm for "+str(tsuspend)+" second")
        return
    if state == 2:
        if time.perf_counter() > mt:
            state = 0
            sas.log.messpr("Ready to detect alarm!")
        return
###############################################

global loop

################################### Main ##########################################

def detectcycle():
    sas.log.messpr("Start detector and camera")
    init()
    netPrepare()
    tp = sas.cfg.getValueInt(sas.config, 'alarmtimepause')
    if tp is None: tp = 60
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = tel.telinit(sas.telid, sas.telash)
    tel.sendMessage(client, sas.addmess)
    while frun:
        ok, frame = cap.read()
        if not ok:  continue
        if sas.detect:
            preprocess(frame)
            outs = net.forward(outNames)
            falarm = postprocess(frame, outs)
            sas.putFrame(frame)
            alarm(falarm, frame, 10, tp)
        else:
            sas.putFrame(frame)
            time.sleep(0.5)
    sas.log.messpr("Detector stopped")



