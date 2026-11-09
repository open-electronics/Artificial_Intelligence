#
# Main program. Starting SocketServer, VoiceTranscribe and Vocalize
# Warning: if you prefer to use Python version of this software, you have
# to change the launch of processes VoiceTranscribe and Vocalize. 
# See lines 168 and 171. Present configuration launches binary version of processes.
# See configuration file "SocketVoiceServerConfig.cfg" to change port address
# or other parameters.
# This server process communicates whith others process by pipes and a common
# area for flags.
#
from multiprocessing import shared_memory
import subprocess as sp
import sys
import signal, os, pathlib

PORT = 50880
def socketLoadconfig():
    global PORT
    import configparser
    fileconfig = 'SocketVoiceServerConfig.cfg'
    thisdir: str = os.path.dirname(os.path.abspath(sys.argv[0]))
    parentdir: str = pathlib.Path(thisdir).parent
    conffile = os.path.join(parentdir, "Utility", fileconfig)
    if not os.path.exists(conffile):
        print(f"File {conffile} not found. Socket server uses default values", file=sys.stderr)
        return
    conf = configparser.ConfigParser()
    try:
        conf.read(conffile)
        PORT = conf.getint('SOCKET_SERVER', 'port')
    except Exception as ex:
        print(ex, file=sys.stderr)
        print("Used default values.", file=sys.stderr)
    return



try :
    shm = shared_memory.SharedMemory(name='VoiceFlag', create=True, size=2)
except:
    shm = shared_memory.SharedMemory(name='VoiceFlag')


def setrun(val: bool):
    if val: shm.buf[1] = 1
    else: shm.buf[1] = 0

def setmute(val: bool):
    if val: shm.buf[0] = 1
    else: shm.buf[0] = 0

setrun(True)
setmute(False)

pidT = None
pidV = None

# --- Imposta ESPEAK_DATA_PATH se espeak-ng-data è nella directory del progetto ---
#_espeak_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "espeak-ng-data")
_espeak_local = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "espeak-ng-data")
os.environ["ESPEAK_DATA_PATH"] = _espeak_local

_ended = False

def endprocesses(signum=None, frame=None):
    # Chiamata sia come handler di signal (2 argomenti) sia direttamente.
    global _ended
    if _ended:  # evita la doppia chiusura (handler + finally)
        return
    _ended = True
    print("End processes!", file=sys.stderr, flush=True)
    for pid in (pidT, pidV):
        if pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # processo già terminato
    shm.close()
    try:
        shm.unlink()
    except FileNotFoundError:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, endprocesses)
signal.signal(signal.SIGTERM, endprocesses)

import socketserver
import threading
import queue


class VoiceServer(socketserver.TCPServer):
    allow_reuse_address = True
    request_queue_size = 5

    def __init__(self, server_address, RequestHandlerClass, procMic, procVoc):
        self.procmic = procMic
        self.procvoc = procVoc
        # Coda globale: il microfono scrive qui, l'handler legge da qui
        self.msg_queue = queue.Queue()

        super().__init__(server_address, RequestHandlerClass)

        # Thread unico di lettura microfono (attivo per tutta la vita del server)
        def mic_reader_thread():
            print("[SERVER] Lettore microfono avviato.")
            # Itera sulle righe della pipe (bloccante, ma non importa perché è un thread dedicato)
            for line in self.procmic.stdout:
                if line:
                    self.msg_queue.put(line)
            print("[SERVER] Lettore microfono arrestato.")

        t = threading.Thread(target=mic_reader_thread, daemon=True)
        t.start()
        self.procvoc.stdin.write("Server pronto per ricevere collegamenti\n")
        self.procvoc.stdin.flush()

class VoiceInterfaceHandler(socketserver.StreamRequestHandler):
    def setup(self):
        super().setup()
        print(f"\n[SERVER] Client connesso: {self.client_address}")

        # OPZIONALE: Svuota la coda dai messaggi vecchi accumulati senza client
        while not self.server.msg_queue.empty():
            try:
                self.server.msg_queue.get_nowait()
            except queue.Empty:
                break

        setrun(True)
        self.active = True
        self.server.procvoc.stdin.write("Cliente connesso. Parlate pure\n")
        self.server.procvoc.stdin.flush()

    def handle(self):
        # Thread di invio al client remoto
        def send_to_client():
            while self.active:
                try:
                    # Controlla la coda ogni secondo per vedere se c'è testo
                    # Il timeout permette di controllare periodicamente 'self.active'
                    msg = self.server.msg_queue.get(timeout=1.0)

                    # Se arriviamo qui, abbiamo un messaggio
                    msg_bytes = msg.encode('utf-8') if isinstance(msg, str) else msg
                    self.request.sendall(msg_bytes)
                except queue.Empty:
                    # Timeout scaduto, nessun messaggio: ricomincia il ciclo e controlla active
                    continue
                except (OSError, BrokenPipeError):
                    break
                except Exception as e:
                    print(f"[DEBUG] Errore invio: {e}")
                    break
            print(f"[THREAD] Chiuso per {self.client_address}")

        sender_thread = threading.Thread(target=send_to_client, daemon=True)
        sender_thread.start()

        # Ciclo principale: ricezione dal client per vocalizzare
        try:
            while self.active:
                data = self.request.recv(4096)
                if not data: break

                testo = data.decode('utf-8')
                # Invia al processo vocalizzatore
                self.server.procvoc.stdin.write(testo)
                self.server.procvoc.stdin.flush()
        except Exception as e:
            print(f"[DEBUG] Errore ricezione: {e}")
        finally:
            self.active = False

    def finish(self):
        self.server.procvoc.stdin.write("Cliente scollegato. In attesa di nuovo collegamento\n")
        self.server.procvoc.stdin.flush()
        setrun(False)
        print(f"[SERVER] Client {self.client_address} scollegato.")
        super().finish()

#Se si vogliono compilare gli script python commentare procT e procV 
#nella versione .py e togliere il commento alla versione .bin

if __name__ == "__main__":
    socketLoadconfig()
    procT = sp.Popen(['python', 'VoiceTranscribe.py', 'VoiceFlag'],
    #procT = sp.Popen(['./VoiceTranscribe.bin', 'VoiceFlag'],
                    stdin=sp.PIPE, stdout=sp.PIPE, text=True)
    procV = sp.Popen(['python', 'Vocalize.py', 'VoiceFlag'],
    #procV = sp.Popen(['./Vocalize.bin', 'VoiceFlag'],
                    stdin=sp.PIPE, stdout=sp.PIPE, text=True)
    pidT = procT.pid
    pidV = procV.pid
    print("Starting processes pid: ", pidT, pidV )
    VoiceServer.allow_reuse_address = True
    server = VoiceServer(('0.0.0.0', PORT), VoiceInterfaceHandler, procT, procV)
    print(f"[MAIN] Server in ascolto sulla porta {PORT}...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[MAIN] Chiusura in corso...")
    finally:
        server.server_close()
        endprocesses()
