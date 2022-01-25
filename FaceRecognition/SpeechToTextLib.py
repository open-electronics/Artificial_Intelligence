"""
Speech to text library (Italian or English)
Principal functions:
- setLang (setLangIt() or setLangEn() default: It)
- listen() : return a string (text recognized); end phrase detected by 0.5 second of silence
- startcontinuelisten() : accumulate text detected in a queue; text can be extracted by getText function
- getText() : return next text extracted from queue (or None)
"""


import speech_recognition as sr
import threading  as thd
import time
import queue

recogn = sr.Recognizer()

audioq = queue.Queue(10)
textq = queue.Queue(5)

listenrun = True
decoderun = True
circular = False

lang = 'it-IT'

def setLangIt():
    global lang
    lang = 'it-IT'


def setLangEn():
    global lang
    lang = 'en-GB'

def init():
    for m in enumerate(sr.Microphone.list_microphone_names()):
        print(m)
    with sr.Microphone() as source:
        try:
            print("Using device num:", source.device_index)
            recogn.adjust_for_ambient_noise(source)
            print("Adjustment done!")
        except:
            print("No microphone found!")

def listen(timeout = None, pause = 0.5, level = 1000):
    with sr.Microphone() as source:
        recogn.energy_threshold = level
        recogn.pause_threshold = pause
        text=""
        okcode= 0
        try:
            audio = recogn.listen(source, timeout)
            text = recogn.recognize_google(audio, language=lang)
        except sr.WaitTimeoutError:
            okcode= -1
        except sr.UnknownValueError:
            okcode= -2
        except sr.RequestError:
            okcode= -3
        finally:
            return text, okcode


####################################

def putAudio(audio):
    try:
        audioq.put_nowait(audio)
    except queue.Full:
        if circular:
            audioq.get_nowait()
            audioq.put_nowait(audio)
        return

def getAudio():
    try:
        audio = audioq.get_nowait()
        return audio
    except:
        return None

def putText(txt):
    try:
        textq.put_nowait(txt)
    except queue.Full:
        if circular:
            textq.get_nowait()
            textq.put_nowait(txt)
        return



def listencontinue():
    global  recogn, listenrun, circular
    recogn.pause_threshold = 0.5
    timeout = 2
    print("Start listening...")
    while listenrun:
        try:
            with sr.Microphone() as source:
                recogn.energy_threshold = 2000
                #recogn.adjust_for_ambient_noise(source, 1)
                audio = recogn.listen(source)
                putAudio(audio)
                #print(audioq.qsize())
        except:
            time.sleep(0.1)
            continue


def decodecontinue():
    global  recogn, decoderun, circular
    while decoderun:
        audio = getAudio()
        if audio is not None:
            try:
                text = recogn.recognize_google(audio, language=lang)
                putText(text)
                #print(textq.qsize())
            except:
                time.sleep(0.1)
                continue


#################################

def startContinuousListen(circularqueue = False):
    global listenrun, decoderun, circular
    circular = circularqueue
    listenrun = True
    decoderun = True
    tha = thd.Thread(name="Listen", target=listencontinue)
    thb = thd.Thread(name="Recogn", target=decodecontinue)
    tha.start()
    thb.start()

def stopContinuousListen():
    lock = thd.Lock()
    lock.acquire()
    global listenrun, decoderun
    listenrun = False
    decoderun = False
    lock.release()

def getText():
    try:
        text = textq.get_nowait()
        return text
    except:
        return None



