import socket
import urllib.request

def getnetAddress():
    localIp = socket.gethostbyname(socket.gethostname())
    #print("My local IP address  : ", localIp)
    externIp = urllib.request.urlopen('https://ident.me').read().decode('utf8')
    #print("My public IP address : ", external_ip)
    return localIp, externIp
