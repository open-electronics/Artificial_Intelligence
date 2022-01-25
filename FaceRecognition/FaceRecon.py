""" See help (end file)"""
import cv2
import face_recognition as fr
import os, sys, argparse, textwrap
import TextToSpeechLib as ts

cw = 640
ch = 480

thisdir = os.path.dirname(os.path.abspath(sys.argv[0]))
cascadename = "haarcascade_frontalface_default.xml"
cascadefile = os.path.join(thisdir, cascadename)
face = cv2.CascadeClassifier(cascadefile)
knownfile = "FacePhoto.jpg"
#knownfile = "FalseFace.jpg"
knownface = os.path.join(thisdir, knownfile)

def verifycam(cameranum):
    webcam = cv2.VideoCapture(cameranum)
    if not webcam.isOpened():
        print("No webcam!")
        exit(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)
    return webcam

def facedetect(frame):
    frameg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fl = face.detectMultiScale(frameg, scaleFactor=1.1, minNeighbors=10)
    if len(fl) == 0:
        return None
    for f in fl:
        x, y, w, h = f
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        sel = frame[y:y+h, x:x+w]
        return sel

def knownencode(knownface):
    known_image = fr.load_image_file(knownface)
    knowncode = fr.face_encodings(known_image)[0]
    return knowncode

def facerecogn(wface, known):
    result = fr.compare_faces([known], wface, tolerance=0.7)
    return result[0]

def cycle(webcam, known):
    while True:
        okf, frame = webcam.read()
        if not okf:
            print("No frame!")
            continue
        fvis = facedetect(frame)
        if fvis is None:
            continue
        fvis = cv2.cvtColor(fvis, cv2.COLOR_BGR2RGB)
        face = fr.face_encodings(fvis)
        if len(face) == 0:
            continue
        ok = facerecogn(face[0], known)
        if ok:
            break

if __name__ == "__main__":
    helptext="""
    Face recognition by webcam. \n
    This program uses webcam 0 and file FacePhoto.jpg (in the same directory) as reference face. \n
    Program exit when recognition has positive ending.
    """

    parse = argparse.ArgumentParser(description=helptext)
    par = parse.parse_args()
    camera = verifycam(0)
    known = knownencode(knownface)
    print("Identification active ...")
    cycle(camera, known)
    message = "Salve Daniele, come posso esserle utile?"
    print(message)
    ts.talk(message)