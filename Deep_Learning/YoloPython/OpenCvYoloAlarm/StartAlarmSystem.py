'''
Alarm system with YOLO deep-learning network for objects detection.
This system uses OpenCV for loading and forwarding network, and for capturing webcam frames.
Parameters used are read from "alarmsystem.cfg" file.
You can decide object detected for alarm (def. person) and threshold used.
This system is configured to send alarm to telegram user, that is the same user
that load BOT on his own phone.
Besides, the system send a short video and the address to connect by browser to see camera.
'''

############ Load configuration parameters and common data #############

import Common as sas
import threading
import time

############ Start detector #############
import OpencvYoloAlarm as detector

thc = threading.Thread(name='CameraStream', target=detector.detectcycle)
thc.start()

############ Start webserverr #############
import WebServer as ws

thw = threading.Thread(name='WebServer', target=ws.startServer)
thw.start()

while True:
    print('To stop process type: "Stop"')
    inp = input()
    if inp == 'Stop':
        sas.log.messpr("Ending alarm system...")
        ws.server_object.shutdown()
        ws.server_object.server_close()
        time.sleep(1)
        sas.log.messpr("Web server stopped")
        detector.frun = False
        time.sleep(1)
        break
    else:
        print("> ", inp)




