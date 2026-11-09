#
# Indipendent process for vocalizing message received by socket
# It uses a interprocess pipe for data and a interprocess common area for
# flag as mute.
# It mutes microphon acquisition when talks. Half duplex mode.
# Based on Piper software using a Italian model.
#

import pyaudio
import piper
import onnxruntime
import os, sys, pathlib

PIPERMODEL = 'it_IT-paola-medium.onnx'
LENGTH_SCALE = 1.5
NOISE_SCALE = 0.667
NOISE_W_SCALE = 0.8
PITCH = 1.0
def vocalLoadconf():
    global PIPERMODEL, LENGTH_SCALE, NOISE_SCALE, NOISE_W_SCALE, PITCH
    import configparser
    fileconfig = 'SocketVoiceServerConfig.cfg'
    thisdir: str = os.path.dirname(os.path.abspath(sys.argv[0]))
    parentdir: str = pathlib.Path(thisdir).parent
    conffile = os.path.join(parentdir, "Utility", fileconfig)
    if not os.path.exists(conffile):
        print(f"File {conffile} not found. Vocalize uses default values", file=sys.stderr)
        return
    conf = configparser.ConfigParser()
    try:
        conf.read(conffile)
        PIPERMODEL = conf.get('TTS', 'Piper_Model')
        LENGTH_SCALE = conf.getfloat('TTS', 'length_scale')
        NOISE_SCALE = conf.getfloat('TTS', 'noise_scale')
        NOISE_W_SCALE = conf.getfloat('TTS', 'noise_w_scale')
        PITCH = conf.getfloat('TTS', 'pitch')
    except Exception as ex:
        print(ex, file=sys.stderr)
        print("Used default values.", file=sys.stderr)
    return


# FIX: Path(__file__) in binari onefile Nuitka punta a /tmp.
# sys.argv[0] punta sempre alla directory reale dell'eseguibile.
dirname: str = os.path.dirname(os.path.abspath(sys.argv[0]))
parentdir: str = pathlib.Path(dirname).parent

samplerate = None

from multiprocessing import shared_memory


nome_shm = sys.argv[1]  # Prende 'VoiceFlag' dagli argomenti
shm = shared_memory.SharedMemory(name=nome_shm)

def setmute(val: bool):
    if val: shm.buf[0] = 1
    else: shm.buf[0] = 0

def vocalize():
    global samplerate
    # Il percorso va calcolato qui, dopo vocalLoadconf(), altrimenti
    # Piper_Model del file di configurazione verrebbe ignorato.
    VOICE = os.path.join(parentdir, "Models", PIPERMODEL)
    TTS = piper.voice.PiperVoice(onnxruntime.InferenceSession, piper.voice.PiperConfig)
    TTS = TTS.load(VOICE, config_path=VOICE+".json")
    TTS.config.length_scale = LENGTH_SCALE
    TTS.config.noise_scale = NOISE_SCALE
    TTS.config.noise_w_scale = NOISE_W_SCALE
    if samplerate is None: samplerate = TTS.config.sample_rate
    samplerate = int(samplerate * PITCH)
    pa = pyaudio.PyAudio()
    strm = pa.open(output=True, rate=samplerate, channels=1, format=pyaudio.paInt16)
    txt = "Pronto a parlare"
    print("*** "+txt+" ***", file=sys.stderr, flush=True)
    #for ck in TTS.synthesize(txt): strm.write(ck.audio_int16_bytes)
    while True:
        txt = sys.stdin.readline()
        if txt == "":  # EOF: il server ha chiuso la pipe
            break
        setmute(True)
        if len(txt) > 0:
            for ck in TTS.synthesize(txt):
                strm.write(ck.audio_int16_bytes)
        setmute(False)
        del (txt)

if __name__ == "__main__":
    vocalLoadconf()
    vocalize()
