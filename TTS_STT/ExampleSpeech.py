import TextToSpeechLib as ts

ts.setLangIt()
#ts.setLangEn()

print("Just return to end")
while True:
    txt = input("Testo: ")
    if txt == '':
        break
    ts.talk(txt)