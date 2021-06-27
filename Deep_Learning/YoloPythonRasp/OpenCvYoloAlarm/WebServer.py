from http.server import HTTPServer, ThreadingHTTPServer, SimpleHTTPRequestHandler
from http import HTTPStatus
import os, sys
import cv2 as cv
import threading
import base64
import Common as sas

thisdir = os.path.dirname(os.path.abspath(sys.argv[0]))     # directory of script
webdir = os.path.join(thisdir, 'WEBROOT')


class HandlerS(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=webdir, **kwargs)

    def responseStat(self, stat):
        print(stat)
        self.send_response_only(HTTPStatus.OK)
        self.send_header("Content-type", 'text/html')
        self.send_header("Content-Length", len(stat))
        self.end_headers()
        self.wfile.write(bytearray(stat, 'ASCII'))
        self.wfile.flush()
    def responseFrame(self, fr):
        frame = base64.encodebytes(bytearray(fr))
        self.send_response_only(HTTPStatus.OK)
        self.send_header("Keep-Alive", "timeout=5")
        self.send_header("Content-type", 'image/jpeg')
        self.send_header("Content-Length", len(frame))
        #print(len(frame))
        self.end_headers()
        self.wfile.write(frame)
        self.wfile.flush()
    def responseNoFrame(self):
        self.send_response_only(HTTPStatus.OK)
        self.send_header("Keep-Alive", "timeout=5")
        self.send_header("Content-type", 'image/jpeg')
        self.send_header("Content-Length", '0')
        self.end_headers()
        self.wfile.flush()
    def do_POST(self):
        req = self.path
        #print(req)
        if req == '/getstate':
            stat = "ind"
            self.responseStat(stat)
            return
        if req == '/setalarm':
            stat = "ON"
            self.responseStat(stat)
            sas.alarmstate = True
            return
        if req == '/resetalarm':
            stat = "OFF"
            self.responseStat(stat)
            sas.alarmstate = False
            return
        if req =='/getframe':
            fr = sas.getFrame()
            if fr is not None:
                ret, fjpg = cv.imencode('.jpg', fr)
                self.responseFrame(fjpg)
            else:
                self.responseNoFrame()
        return


server_object = None

def startServer():
    global server_object
    server_object = ThreadingHTTPServer(server_address=('', 4000), RequestHandlerClass=HandlerS)
    sas.log.messpr("Start web server")
    server_object.serve_forever(0.3)

