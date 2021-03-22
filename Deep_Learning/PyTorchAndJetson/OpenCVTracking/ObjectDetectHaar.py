'''
This example achieves a object detection task, using OpenCV HaarCascade system.
Help description is explained, if you start script with -? parameter.
The object detected (just one: the first one) is framed by a thin rectangle, and its center is sent by serial port
at 57600 bauds.
In addition the selected portion of frame is extracted and can be showed in a small window.
The main thread starts a support thread to extract selected object in a small image, and starts a thread for serial
communication if serial parameter points to a serial port name available.
Parameters:
    -cuda (use CUDA if present) (def. don't use)
    -windowdim S or L (L:640x480, S:320x240) (def. L)
    -file haarfilename (complete path)
    -camnum camera_number (def. 0)
    -serial serial_port_name
    -nowin (don't display camera image) (def. false)
    -winsel (display selected object) (def. false)
    -trace (log center, error and correction sent by controller if present) (def. false)
    -react 0-:-1 (smoothing of detected center) (def. 0.3)
'''


import argparse
import cv2
import os, sys
import time
import seriallib as ser
import threading

thisdir = os.path.dirname(os.path.abspath(sys.argv[0]))     # directory of script

CUDA = False
cw = 640
ch = 480
HAAR_CASCADE = thisdir+"\\haarcascades_cuda\\haarcascade_frontalface_default.xml"
g = 0.3
camera = 0
portname = None
bauds = 57600
portser = None
f = None
traceon = False

isRunning = True      # flag for closing multithreads


cxm = None
cym = None
wt2m = None
ht2m = None

ok = False
detectev = False
showframe = True
showselect = False

wt2m, ht2m

def params():
    global HAAR_CASCADE, camera, g, portname, traceon, showframe, showselect, cxm, cym, wt2m, ht2m
    parse = argparse.ArgumentParser(description="Object detection with HAAR Cascade model")
    parse.add_argument('-?', action='help', help='as -h, --help')
    parse.add_argument('-cuda', action= 'store_true', help='use CUDA (if present)', default=CUDA)
    parse.add_argument('-windowdim', help='Window dimension L or S (L:640x480, S:320x240)', default='L')
    parse.add_argument('-file', help='Haar cascade file (complete path)', default=HAAR_CASCADE)
    parse.add_argument('-camnum', help='Camera number', default=camera)
    parse.add_argument('-react', help='Reactivity of tracking (0.1 - 1)', default=g)
    parse.add_argument('-serial', help='Serial port name (def. None)', default=None)
    parse.add_argument('-trace', action='store_true', help='write errors and corrections (def. False)', default=False)
    parse.add_argument('-nowin', action='store_true', help='No window (def. False)', default=False)
    parse.add_argument('-winsel', action='store_true', help='Selection window (def. False)', default=False)
    par = parse.parse_args()
    global cuda
    cuda = par.cuda
    global cw, ch
    if par.windowdim == 'L' :
        cw = 640
        ch = 480
    if par.windowdim == 'S' :
        cw = 320
        ch = 240
    cxm = cw/2
    cym = ch/2
    wt2m = cxm
    ht2m = cym
    HAAR_CASCADE = par.file
    g = par.react
    camera = par.camnum
    portname = par.serial
    traceon = par.trace
    showframe = not par.nowin
    showselect = par.winsel

def verifycuda():                           # verify if CUDA is require and if it is available
    global CUDA
    info = cv2.getBuildInformation()
    if info.count(" CUDA") > 0:
        print("CUDA present! ", end='')
        print(cv2.cuda_DeviceInfo())
        iscuda = True
    else:
        print("CUDA not present!")
        iscuda = False
    if cuda and iscuda:
        CUDA = True
    else:
        CUDA = False

def verifycam():                            # verify if camera is available
    global webcam
    webcam = cv2.VideoCapture(camera)
    if not webcam.isOpened():
        print("No webcam!")
        exit(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)

def verifyhaarcascade():                    # verify if haarcascade file can be loaded
    global HAAR_CASCADE
    if not os.path.exists(HAAR_CASCADE):
        print("File ",HAAR_CASCADE, " not found!")
        exit(1)
    global face
    if CUDA:
        face = cv2.cuda.CascadeClassifier_create(HAAR_CASCADE)
    else:
        face = cv2.CascadeClassifier(HAAR_CASCADE)
    if face.empty():
        print("File ", HAAR_CASCADE, " not loaded!")
        exit(1)
    global faces
    faces = []

def verifySerial():                          # verify if the serial port exist and can be open and start thread
    global portname, portser
    if portname is None:
        return
    portser = ser.openser(portname, bauds, tout=0.1)
    if portser is None:
        return
    thserial = threading.Thread(name='Serial', target=sendCenter, args=(detectev,))
    thserial.start()

def sendCenter(e):                           # serial thread function; send center and receive reply
    global portser, cxm, cym, f
    print("Start thread serial ...")
    ser.flushout(portser)
    ser.swriteln(portser, "r")   # Reset to start server position
#    ser.swriteln(portser, "s")   # Uncomment if you want send and receive from controller without servos action
    if traceon:
        tracefile = os.path.join(thisdir,"datatrack.txt")
        f = open(tracefile, "w")
        print("Trace data on: "+f.name)
    while isRunning:
        if portser is None:
            break
        e.wait()
        x = round(float(cxm)/cw, 3)
        y = round(float(cym)/ch, 3)
        center = str(x)+" "+str(y)
        ser.flushinp(portser)
        ser.swriteln(portser, center)
        reply = ser.sreadline(portser)
        e.clear()
        if not f is None:
            f.write(reply+'\n')
            f.flush()
    ser.closeser(portser)
    if not f is None:
        f.close()
    return

def selection(e):                                # selection thread function; extract subframe
    global frame, cxm, cym, wt2m, ht2m
    while isRunning:
        e.wait()
        # square selection
        if ht2m < wt2m:
            ht2m = wt2m
        else:
            wt2m = ht2m
        select = frame[int(cym - ht2m):int(cym + ht2m), int(cxm - wt2m):int(cxm + wt2m)]
        e.clear()
        if showselect:
            cv2.imshow(WinSelect, select)
    if showselect:
        cv2.destroyWindow(WinSelect)
    return

# principal function: detect object and compute rectangle and center
def detect(frame):
    global wt2m, ht2m, cxm, cym, first
    frameg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if CUDA:
        gframeg = cv2.cuda_GpuMat(frameg)
        gfaces = face.detectMultiScale(gframeg, scaleFactor=1.1, minNeighbors=10)
        faces = gfaces.download()
        if not faces is None:
            faces = faces[0]
    else:
        faces = face.detectMultiScale(frameg, scaleFactor=1.1, minNeighbors=10)
        #for (x, y, w, h) in faces:
    if len(faces) > 0:
        x, y, w, h = faces[0]       #just the first one (faces[0]) object detected
        wt2 = w/2
        ht2 = h/2
        cx = x + wt2
        cy = y + ht2
        if first:
            wt2m = wt2; ht2m = ht2; cxm = cx; cym = cy
            first = False
        else:
            wt2m = wt2m + g * (wt2-wt2m)
            ht2m = ht2m + g * (ht2-ht2m)
            cxm = cxm + g * (cx-cxm)
            cym = cym + g * (cy-cym)
        if showframe and not showselect:
            cv2.rectangle(frame, (int(cxm-wt2m), int(cym-ht2m)), (int(cxm+wt2m), int(cym+ht2m)), (0, 0, 255), 1)
            cv2.circle(frame,(int(cxm),int(cym)),5, (0,0,255),1)
            cv2.circle(frame,(int(cw/2),int(ch/2)),2, (255,0,0),2 )
    return

# basic repetition : frame read and used for detection
def cycle():
    cf = 0
    TT = 0
    global first, frame
    first = True
    while cv2.waitKey(100) < 0:            # repeat (waiting 1 ms) until a key is pressed
        T0 = time.perf_counter()
        okf, frame = webcam.read()
        if not okf:
            print("No frame!")
            continue
        detect(frame)                     # principal function
        detectev.set()                    # event set (used by other threads)
        if showframe:
            cv2.imshow(WinName, frame)
        T1 = time.perf_counter()
        TT = TT + (T1 - T0)
        cf = cf + 1
        if cf == 100:
            print("Frame timing: {:6.3f} sec  ".format(round(TT/cf, 3)), end='')
            print("Frame/s : {:6.1f}".format(round(cf / TT, 1)))
            cf = 0
            TT = 0
    return

if __name__ == "__main__":
    params()
    verifycuda()
    print ("Object detection.")
    if CUDA:
        print("  Using Cuda")
    else:
        print("  Without Cuda")
    print ("  Param: Window ", cw, "x", ch)
    print ("  Param: File=", HAAR_CASCADE)
    print ("  Param: Camera=", camera, "Smooth coeff.=", g)
    print ("  Param: Serial=", portname)
    print ("  Param: Trace=", traceon)
    verifycam()
    verifyhaarcascade()
    detectev = threading.Event()  # event set when new detection is performed. Used by other threads
    verifySerial()
    if showframe:
        WinName = 'HaarCascade object detect'
        cv2.namedWindow(WinName, cv2.WINDOW_AUTOSIZE)
    if showselect:
        WinSelect = 'Object selected'
        cv2.namedWindow(WinSelect, cv2.WINDOW_AUTOSIZE)
    thselection = threading.Thread(name='Selection', target=selection, args=(detectev,))
    thselection.start()
    print("Starting detection... (press a key to stop)")
    cycle()
    isRunning = False            # exit threads
    detectev.set()
    cv2.destroyAllWindows()
    print("End detection!")
    exit()



# 640x480 30f/s (ssh) 32f/s (direct)
# 320x240 32f/s (ssh) 32f/s (direct)