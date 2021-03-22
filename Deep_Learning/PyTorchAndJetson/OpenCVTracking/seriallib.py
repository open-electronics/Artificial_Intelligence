import sys
import serial
import serial.tools.list_ports
import time


def prserialports():
    portinfo = list(serial.tools.list_ports.comports())
    for dev in portinfo:
        print(dev.device)

def portexist(port):
    portinfo = list(serial.tools.list_ports.comports())
    for dev in portinfo:
        if dev.device == port:
            return True
    return False


def openser(port, bauds, tout=1):
    if not portexist(port):
        print("No serial: " + port)
        return None
    ser = serial.Serial(port, baudrate=bauds, timeout=tout)
    time.sleep(2)
    if ser.isOpen():
        print(ser.name + " is open...  ")
        return ser
    else:
        print("No serial: "+ser.name)
        return None

def closeser(port):
    port.close()

def swriteln(port,data):
    if port is None:
        return
    data = data+'\n'
    bdata = bytes(data,'utf-8')
    port.write(bdata)

def sreadline(port):
    if port is None:
        return ""
    data = port.readline()
    rec = str(data,'utf-8')
    rec = rec.replace('\n', '')
    rec = rec.replace('\r', '')
    return rec

def available(port):
    if port.in_waiting > 0:
        return True
    else:
        return False

def setTimeout(port, time):
    port.timeout = time

def flushinp(port):
    port.reset_input_buffer()

def flushout(port):
    port.reset_output_buffer()
