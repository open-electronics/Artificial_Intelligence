"""
Listen (def. in Italian) continuously and transcribe (print) everything
Say "ascolta stop" to stop
"""

import SpeechToTextLib as st
import time

st.setLangIt()
#st.setLangEn()

circque = False
#circque = True

print("Say stop to end")

st.startContinuousListen(circularqueue = circque)
if st.lang == 'it-IT':
    print("Sono in ascolto...")
else:
    print("I'm listening...")

fend = False

while not fend:
    txt = st.getText()
    if txt is None:
        time.sleep(0.2)
        continue
    print(txt)
    if txt.startswith("Ascolta stop"):
        print("Ending...")
        st.stopContinuousListen()
        break

