""" Test different face detect algorithms (HAAR, LBP, HOG, CNN"""
""" See command parameters (end file)"""

import cv2
import face_recognition as fr
import time, os, sys
import argparse
import TextToSpeechLib as ts

webcam = None
camera = 0
cw = 640
ch = 480

namefile = "FacePhoto.jpg"
thisdir = os.path.dirname(os.path.abspath(sys.argv[0]))
filephoto = os.path.join(thisdir, namefile)

HAAR_CASCADE = "haarcascade_frontalface_default.xml"
LBP_CASCADE = "lbpcascade_frontalface_improved.xml"
face0 = None
face1 = None

#base = "D:\\AI"              #windows
base = "/home/pi/AIData"     #raspberry

fileHaar = os.path.join(base, "HaarCascade", "haarcascades_cuda", HAAR_CASCADE)
fileLbp = os.path.join(base, "HaarCascade", "lbpcascades", LBP_CASCADE)

METHOD = 0  # 0 : HAAR | 1 : LBP | 2 : HOG | 3 : CNN

def verifyhaarcascade_lbp():
    global face0, face1
    if not os.path.exists(fileHaar):
        print("File ",fileHaar, " not found!")
        exit(1)
    face0 = cv2.CascadeClassifier(fileHaar)
    if not os.path.exists(fileLbp):
        print("File ",fileLbp, " not found!")
        exit(1)
    face1 = cv2.CascadeClassifier(fileLbp)

def verifycam():
    global webcam
    webcam = cv2.VideoCapture(camera)
    if not webcam.isOpened():
        print("No webcam!")
        exit(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)

def detectHOG_CNN(frame, mod):
    frameg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fl = fr.face_locations(frameg, model=mod)
    if len(fl) == 0:
        return None
    for f in fl:
        y0, x1, y1, x0 = f
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        vis = frame[y0:y1, x0:x1]
        return vis

def detectHAAR_LBP(frame, face):
    frameg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fl = face.detectMultiScale(frameg, scaleFactor=1.1, minNeighbors=10)
    if len(fl) == 0:
        return None
    for f in fl:
        x, y, w, h = f
        #x -= 10 ; y -= 10 ; w += 40 ; h += 40
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        sel = frame[y:y+h, x:x+w]
        return sel

def cycle(photo):
    cf = 0
    TT = 0
    global first, METHOD
    first = True
    fvis = None
    WinName = 'Face detecting'
    cv2.namedWindow(WinName, cv2.WINDOW_AUTOSIZE)
    while cv2.waitKey(1) < 0:
        T0 = time.perf_counter()
        okf, frame = webcam.read()
        if not okf:
            print("No frame!")
            continue
        if METHOD == 0:
            fvis = detectHAAR_LBP(frame, face0)
        elif METHOD == 1:
            fvis = detectHAAR_LBP(frame, face1)
        elif METHOD == 2:
            fvis = detectHOG_CNN(frame, "hog")
        elif METHOD == 3:
            fvis = detectHOG_CNN(frame, "cnn")
        if fvis is None:
            continue

        cv2.imshow(WinName, frame)
        #cv2.imshow(WinName, fvis)
        if photo:
            if cf == 20:
                cv2.imwrite(filephoto, fvis)
                ts.talk("OK. Foto scattata!")
                print(filephoto)
                photo = False
                exit(0)

        T1 = time.perf_counter()
        TT = TT + (T1 - T0)
        cf = cf + 1
        if cf == 100:
            print("Frame/s : {:6.1f}".format(round(cf / TT, 1)))
            cf = 0
            TT = 0

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Test face detection")
    parse.add_argument('--model', default=0, choices=['0','1','2','3'], help='0 : HAAR | 1 : LBP | 2 : HOG | 3 : CNN (def: 0)')
    parse.add_argument('--saveface', action='store_true', help='Save face photo (def: False)')
    parse.add_argument('--camera', type=int, default=camera, help='camera number (def: '+str(camera)+')')
    parse.add_argument('--name', default=namefile, help='file photo name (def: '+namefile+')')
    par = parse.parse_args()
    METHOD = int(par.model)
    camera = par.camera
    filephoto = os.path.join(thisdir, par.name)
    verifyhaarcascade_lbp()
    verifycam()
    cycle(par.saveface)