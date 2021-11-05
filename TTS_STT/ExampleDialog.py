import SpeechToTextLib as st

print("Ti ascolto")
timeout= 1
while True:
    print("Dimmi...")
    txt, ok = st.listen(timeout= timeout)
    if ok == -1:
        print("Parla, per favore!")
        timeout = 5
        continue
    if ok == -2:
        print("Scusa, non ho capito. Ripeti per favore")
        continue
    if ok == -3:
        print("Errore di sistema (manca connessione?) !")
        break
    print("Hai detto: ",txt)
    if txt.startswith("ascolto stop"):
        break
    timeout= 0.5
