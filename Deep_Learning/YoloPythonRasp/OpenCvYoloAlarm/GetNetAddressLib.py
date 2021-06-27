import socket
import urllib.request
import subprocess

def getnetAddress():
#    localIp = socket.gethostbyname(socket.gethostname())
    sp = subprocess.run(["hostname",'-I'], stdout= subprocess.PIPE, encoding= 'utf-8')
    iplist = sp.stdout.split()
    localIp= iplist[0]
    #print("My local IP address  : ", localIp)
    externIp = urllib.request.urlopen('https://ident.me').read().decode('utf8')
    #print("My public IP address : ", external_ip)
    return localIp, externIp
