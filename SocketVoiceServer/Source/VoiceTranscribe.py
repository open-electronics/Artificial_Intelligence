#
# Trascrizione da microfono basata su processo indipendente.
# Basata su Vosk (modello italiano) + openWakeWord per la parola di attivazione.
#
# Funzionamento:
#   IDLE   : il microfono viene analizzato in memoria SOLO da openWakeWord
#            (nessuna trascrizione, nessun audio salvato o inviato).
#   ACTIVE : dopo il rilevamento della wake word (es. "alexa") si attiva Vosk;
#            il testo riconosciuto viene scritto su stdout (pipe verso il server).
#            Si torna in IDLE a fine frase oppure per timeout (vedi config).
#
# Comunica con gli altri processi tramite pipe e area comune per i flag
# (buf[0] = mute, buf[1] = run). Una coda locale funge da buffer microfono.
#
# Se [WAKEWORD] enabled = false si comporta come la versione precedente
# (ascolto continuo).
#

import queue
import sys, os, pathlib
import time
import json
import numpy as np
import sounddevice as sd

from vosk import Model, KaldiRecognizer, SetLogLevel

VOSKMODEL = 'vosk-model-small-it-0.22'

# --- Parametri wake word (sovrascrivibili da sezione [WAKEWORD] del .cfg) ---
WAKE_ENABLED = True
WAKE_MODEL = 'alexa'      # nome modello predefinito oppure file .onnx (anche in Models/)
WAKE_THRESHOLD = 0.5      # soglia di rilevamento 0..1 (alzare se scatta a vuoto)
LISTEN_TIMEOUT = 6.0      # s: dopo l'attivazione, attesa massima dell'inizio del parlato
MAX_ACTIVE = 30.0         # s: durata massima assoluta di una sessione di ascolto
END_ON_PHRASE = True      # True: torna in IDLE alla fine della prima frase
                          # False: resta attivo finché c'è silenzio > LISTEN_TIMEOUT
FOLLOWUP = 0.0            # s: finestra di ascolto senza wake word dopo che il
                          # sintetizzatore ha finito di parlare (0 = disabilitata)

######## Beep
class SdBeep():
    def __init__(self, rate=16000, freq=880, dur=0.3):
        self.sample_rate = rate  # Frequenza di campionamento (Hz)
        self.frequency = freq    # Frequenza del tono (es. La a 440 Hz)
        self.duration = dur      # Durata in secondi
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        self.lt2 = int(len(t)/2)
        self.waveform = 0.5 * np.sin(2 * np.pi * self.frequency * t)
    def beep(self):
        sd.play(self.waveform, self.sample_rate)
        sd.wait()
    def beepend(self):
        sd.play(self.waveform[:self.lt2], self.sample_rate/2)
        sd.wait()
##############

def transcrLoadconf():
    global VOSKMODEL, WAKE_ENABLED, WAKE_MODEL, WAKE_THRESHOLD
    global LISTEN_TIMEOUT, MAX_ACTIVE, END_ON_PHRASE, FOLLOWUP
    import configparser
    fileconfig = 'SocketVoiceServerConfig.cfg'
    thisdir: str = os.path.dirname(os.path.abspath(sys.argv[0]))
    parentdir: str = pathlib.Path(thisdir).parent
    conffile = os.path.join(parentdir, "Utility", fileconfig)
    if not os.path.exists(conffile):
        print(f"File {conffile} not found. Transcribe uses default values", file=sys.stderr)
        return
    conf = configparser.ConfigParser()
    try:
        conf.read(conffile)
        VOSKMODEL = conf.get('STT', 'dir_Vosk_Model')
    except Exception as ex:
        print(ex, file=sys.stderr)
        print("Used default values.", file=sys.stderr)
    # Sezione opzionale: se manca restano i valori di default
    try:
        WAKE_ENABLED = conf.getboolean('WAKEWORD', 'enabled', fallback=WAKE_ENABLED)
        WAKE_MODEL = conf.get('WAKEWORD', 'model', fallback=WAKE_MODEL)
        WAKE_THRESHOLD = conf.getfloat('WAKEWORD', 'threshold', fallback=WAKE_THRESHOLD)
        LISTEN_TIMEOUT = conf.getfloat('WAKEWORD', 'listen_timeout', fallback=LISTEN_TIMEOUT)
        MAX_ACTIVE = conf.getfloat('WAKEWORD', 'max_active', fallback=MAX_ACTIVE)
        END_ON_PHRASE = conf.getboolean('WAKEWORD', 'end_on_phrase', fallback=END_ON_PHRASE)
        FOLLOWUP = conf.getfloat('WAKEWORD', 'followup', fallback=FOLLOWUP)
    except Exception as ex:
        print(ex, file=sys.stderr)
        print("Wake word: used default values.", file=sys.stderr)
    return

sdb=SdBeep()

q = queue.Queue()  # coda audio del microfono

dirname: str = os.path.dirname(os.path.abspath(sys.argv[0]))
parentdir: str = pathlib.Path(dirname).parent

samplerate = 16000
# openWakeWord lavora meglio con frame da 80 ms (1280 campioni a 16 kHz).
# Lo stesso blocco va bene anche per Vosk.
blocksize = 1280

from multiprocessing import shared_memory

nome_shm = sys.argv[1]  # Prende 'VoiceFlag' dagli argomenti
shm = shared_memory.SharedMemory(name=nome_shm)


def setmute(val: bool):
    if val: shm.buf[0] = 1
    else: shm.buf[0] = 0


def mute():
    return shm.buf[0] == 1


def frun():
    return shm.buf[1] == 1


def callback(indata, frames, time_info, status):
    """Chiamata (da thread separato) per ogni blocco audio."""
    if mute() or not frun():
        return
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


def drain(qq: queue.Queue):
    while True:
        try:
            qq.get_nowait()
        except queue.Empty:
            break


def load_wakeword():
    """Carica openWakeWord. Cerca un file .onnx locale, altrimenti usa il
    nome di un modello predefinito (es. 'alexa', 'hey_jarvis')."""
    import openwakeword
    from openwakeword.model import Model as WakeModel

    spec = WAKE_MODEL
    for c in (WAKE_MODEL,
              os.path.join(parentdir, "Models", WAKE_MODEL),
              os.path.join(parentdir, "Models", WAKE_MODEL + ".onnx")):
        if os.path.isfile(c):
            spec = c
            break
    try:
        return WakeModel(wakeword_models=[spec], inference_framework="onnx")
    except Exception as ex:
        print(f"Wake word: caricamento fallito ({ex}). Provo a scaricare i modelli...",
              file=sys.stderr)
        openwakeword.utils.download_models()
        return WakeModel(wakeword_models=[spec], inference_framework="onnx")


def voicetraslate():
    SetLogLevel(-1)
    fvoskmodel = os.path.join(parentdir, "Models", VOSKMODEL)
    print("Loading model:", fvoskmodel, file=sys.stderr)
    model = Model(fvoskmodel)
    rec = KaldiRecognizer(model, samplerate)
    oww = load_wakeword() if WAKE_ENABLED else None

    # --- stato della sessione di ascolto ---
    active = (oww is None)   # senza wake word: sempre attivo (modo continuo)
    t_hard = 0.0             # istante di attivazione (per MAX_ACTIVE)
    t_start = 0.0            # istante da cui misurare il silenzio
    t_listen = LISTEN_TIMEOUT
    speech = False           # è stato sentito parlato in questa sessione/frase?

    def activate(listen_timeout: float):
        nonlocal active, t_hard, t_start, t_listen, speech
        sdb.beep()
        active = True
        t_hard = t_start = time.monotonic()
        t_listen = listen_timeout
        speech = False
        rec.Reset()
        print("*** Wake word: ascolto attivo ***", file=sys.stderr, flush=True)

    def deactivate():
        nonlocal active
        sdb.beepend()
        if oww is None:
            return
        active = False
        rec.Reset()
        oww.reset() if hasattr(oww, "reset") else None
        print("*** In attesa della wake word ***", file=sys.stderr, flush=True)
        

    def emit(text: str):
        if text:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()

    stm = sd.RawInputStream(samplerate=samplerate, blocksize=blocksize,
                            dtype="int16", channels=1, callback=callback)
    was_muted = False
    with stm:
        if oww is None:
            print("*** Ricezione attiva (continua) ***", file=sys.stderr)
        else:
            print(f"*** In attesa della wake word '{WAKE_MODEL}' ***", file=sys.stderr)

        while True:
            if not frun():
                # Nessun client collegato: si torna in attesa
                if oww is not None and active:
                    deactivate()
                drain(q)
                time.sleep(0.1)
                continue

            if mute():
                # Il sintetizzatore sta parlando: stop microfono (half duplex)
                if stm.active: stm.stop()
                drain(q)
                was_muted = True
                time.sleep(0.01)
                continue

            if stm.stopped: stm.start()

            if was_muted:
                # Appena finito di parlare: ripulisce i modelli da residui audio
                was_muted = False
                rec.Reset()
                if oww is not None:
                    if hasattr(oww, "reset"): oww.reset()
                    if FOLLOWUP > 0:
                        activate(FOLLOWUP)   # finestra di risposta senza wake word
                    else:
                        active = False

            try:
                data = q.get(timeout=0.1)
            except queue.Empty:
                data = None

            now = time.monotonic()

            # ---------------- IDLE: solo rilevamento wake word ----------------
            if not active:
                if data is None:
                    continue
                frame = np.frombuffer(data, dtype=np.int16)
                scores = oww.predict(frame)
                if scores and max(scores.values()) >= WAKE_THRESHOLD:
                    activate(LISTEN_TIMEOUT)
                continue

            # ---------------- ACTIVE: trascrizione Vosk ----------------
            if data is not None and rec.AcceptWaveform(data):
                restxt = json.loads(rec.Result()).get("text", "")
                if len(restxt) > 0:
                    emit(restxt)
                    if oww is not None:
                        if END_ON_PHRASE:
                            deactivate()
                            continue
                        # modo "a tempo": rinnova la finestra di silenzio
                        t_start = now
                        speech = False
                        continue
            elif oww is None:
                continue  # modo continuo: niente timeout

            if oww is None:
                continue

            # Controllo timeout (solo con wake word)
            if now - t_hard > MAX_ACTIVE:
                emit(json.loads(rec.FinalResult()).get("text", ""))
                deactivate()
                continue

            # Dopo un risultato finale (anche vuoto) Vosk può restituire un JSON
            # senza la chiave "partial": va letta in modo difensivo.
            partial = json.loads(rec.PartialResult()).get("partial", "")
            if partial:
                speech = True
            elif not speech and now - t_start > t_listen:
                deactivate()


if __name__ == "__main__":
    transcrLoadconf()
    voicetraslate()
