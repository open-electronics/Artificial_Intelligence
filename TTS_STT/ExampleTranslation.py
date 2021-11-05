import SpeechToTextLib as st
import TextToSpeechLib as ts
import time
from googletrans import Translator

tr = Translator()

st.setLangIt()
ts.setLangEn()

st.startContinuousListen()

print("Avvio traduzione, parla...")

fend = False
while not fend:
    txt = st.getText()
    if txt is None:
        time.sleep(0.2)
        continue
    tdata = tr.translate(txt, src='it', dest='en')
    ts.talk(tdata.text)
    if txt.startswith("Ascolta stop"):
        print("Ending...")
        st.stopContinuousListen()
        break
