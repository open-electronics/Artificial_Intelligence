#
# Utility program for simple test.
# This program has tu be lunched as separate process possibly in another computer
# locate on the same local network.
# It is a server client that does a simple echo.
#

import socket as sk
import sys
import time

PORT = 50880
#####################################################################################
def find_and_connect(port=PORT):
    tm = 0.1 # Ridotto per velocità
    for i in range(-1, 255):
        ip = "127.0.0.1" if i == -1 else f"192.168.1.{i}"
        try:
            # Tenta la connessione
            sock = sk.create_connection((ip, port), timeout=tm)
            # Rimuove il timeout di scansione per l'uso normale
            sock.settimeout(None)
            print(f"Connected to Voice Server at {ip}")
            return sock, ip
        except OSError:
            continue
    return None, None


#####################################################################################

def echoloop(sock):
    while True:
        try:
            data = sock.recv(1000)
            if not data: break
            print(str(data, encoding='utf-8'), end='')
            sock.send(data)
        except OSError as err:
            endconnection(err, sock)


def endconnection(err, sock):
    print(f"Network problem: {err}. Connection closed.")
    print("Client closed!")
    sock.close()
    sys.exit(-1)

if __name__ == '__main__':
    if __name__ == '__main__':
        # Invece di scanserver() che chiude il socket, usiamo una funzione
        # che ci restituisce il socket già pronto all'uso.
        print("Looking for Voice Server on this LAN 192.168.1.* ...")
        sock, server_ip = find_and_connect(PORT)
        if sock:
            echoloop(sock)
        else:
            print("No Voice Server on LAN!")
            input("Enter for exit")
