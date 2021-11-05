"""
Read and speak a text in Italian (default) or English (for a English text)
Principal functions:
- setLang (setLangIt() or setLangEn() default: It)
- talk(text) : read out text
- talktofile(text, [file]) : if not file parameter , default file TextSpeech.mp3
"""


from gtts import gTTS
import miniaudio as maud
import io, os, sys
import time

lang = 'it'

thisdir = os.path.dirname(os.path.abspath(sys.argv[0]))
filesp= os.path.join(thisdir, "TextSpeech.mp3")

def setLangIt():
    global lang
    lang = 'it'


def setLangEn():
    global lang
    lang = 'en'

def talk(text):
    tts = gTTS(text=text, lang=lang)
    mf = io.BytesIO()
    tts.write_to_fp(mf)
    bd = mf.getvalue()
    strm = maud.stream_memory(bd)
    t = maud.mp3_get_info(bd).duration+0.1
    with maud.PlaybackDevice() as device:
        device.start(strm)
        time.sleep(t)
    device.close()
    mf.close()

def talktofile(text, file=filesp, progress= 0):
    tts = gTTS(text=text, lang=lang)
    if progress == 0:
        tts.save(file)
    else:
        name, ext=os.path.splitext(file)
        name = name+str(progress)+"."+ext
        tts.save(name)

