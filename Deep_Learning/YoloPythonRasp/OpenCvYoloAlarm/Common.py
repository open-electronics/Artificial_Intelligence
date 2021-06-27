import GetConfigLib as cfg
import GetNetAddressLib as nadd
config = cfg.readConfigParam()
model = cfg.getValue(config, 'modelnet')
if model is None: model = 'yolov4'
alarmstate = cfg.getValueBoolean(config, 'alarmphone')
telid = cfg.getValueInt(config, 'telegramid')
telash = cfg.getValue(config, 'telegramash')
botadd = cfg.getValue(config, 'telegrambotadd')
detect = cfg.getValueBooleanFalse(config, 'detector')

telok = True

ladd, padd = nadd.getnetAddress()
addmess = "http://"+ladd+":4000/   http://"+padd+":4000/ "

import sys, os, datetime
class Log:
    maxlen = 64000
    thisdir = os.path.dirname(os.path.abspath(sys.argv[0]))
    logfile0 = os.path.join(thisdir, 'Alarm.log0')
    logfile1 = os.path.join(thisdir, 'Alarm.log1')
    flog = open(logfile1, 'w')
    def createmess(self, text):
        date = datetime.datetime.now()
        sdate = date.isoformat(sep=' ', timespec='minutes')
        return sdate+" "+text
    def mess(self, text):
        message = self.createmess(text)
        self.flog.write(message+"\n")
        self.flog.flush()
        len = os.lstat(self.logfile1).st_size
        if len > self.maxlen:
            self.flog.close()
            os.renames(self.logfile1, self.logfile0)
            self.flog = open(self.logfile1, 'w')
        return message
    def messpr(self, text):
        message = self.mess(text)
        print(message)
log = Log()

if telid is None:
    log.messpr('Telegram Id incorrect!')
    telok = False
if telash is None:
    log.messpr('Telegram ash incorrect!')
    telok = False



import threading, queue
frameq = queue.Queue(5)

def putFrame(frame):
    try:
        frameq.put_nowait(frame)
    except:
        return

def getFrame():
    try:
        frm = frameq.get_nowait()
        return frm
    except:
        return None