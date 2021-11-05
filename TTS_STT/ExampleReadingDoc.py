"""
Read text document in Italian (default) or English (but document has to be English).
If text file is not provided as parameter, a file chooser window will be open
-h for help:
    usage: ExampleReadingDoc.py [-h] [--filetxt FILETXT] [--filemp3 FILEMP3] [--audio {True,False}] [--print {True,False}] [--lang {'IT','EN'}]
    Read text doc and speech (default in Italian)
    optional arguments:
      -h, --help            show this help message and exit
      --filetxt FILETXT     File doc (def: filechooser)
      --filemp3 FILEMP3     file output mp3 (def: TextSpeech.mp3)
      --audio {True,False}  output audio or mp3 (def: True; out -> audio)
      --print {True,False}  print each line (def: True)
      --lang {IT,EN}    Language (def: IT)

"""


from tkinter import filedialog as fd, Tk
from tkinter import messagebox as msg
import TextToSpeechLib as ts
import argparse

outsound = False
filemp3 = None
filetxt = None
prline= False


def filechooser():
        win = Tk()
        win.withdraw()
        file = fd.askopenfile(title="Choose file", filetypes=[("Text file", (".txt"))])
        if file == None:
            exit(0)
        else:
            return file.name


def readfile(file):
    fdoc = open(file, "rb")
    fmp3 = open(filemp3, "wb")
    eof = False
    endf = False
    while not eof:
        rec = list()
        endf = False
        while not endf:
            b = fdoc.read(1)
            if b == b'':
                eof = True
                break
            if b < b' ':
                b = b' '
            rec.append(b.decode('CP437')) #use 8 bits non just 7 as utf-8 (latin accent too)
            if b == b'.' or b == b'!' or b == b'?':
                srec =''.join(rec)
                if prline:
                    print(srec)
                if outsound:
                    ts.talk(srec)
                else:
                    tts = ts.gTTS(text=srec, lang=ts.lang)
                    tts.write_to_fp(fmp3)
                endf = True



def init():
    global outsound, filemp3, filetxt, prline
    parse = argparse.ArgumentParser(description="Read text doc and speech (default in Italian)")
    parse.add_argument('--filetxt', default=None, help='File doc (def: filechooser)')
    parse.add_argument('--filemp3', default=ts.filesp, help='file ouput mp3 (def: TextSpeech.mp3)')
    parse.add_argument('--audio', choices=['True', 'False'], default='True', help='output audio no mp3 (def: True; out audio)')
    parse.add_argument('--print', choices=['True', 'False'], default='True', help='print each line (def: True)')
    parse.add_argument('--lang', choices=['IT', 'EN'], default='IT', help="Language (def: IT)")
    par = parse.parse_args()
    outsound= eval(par.audio)
    filemp3= par.filemp3
    filetxt= par.filetxt
    prline= eval(par.print)
    language= par.lang
    if language == 'IT':
        ts.setLangIt()
    else:
        ts.setLangEn()
    if filetxt is None:
        filetxt= filechooser()
    if filetxt is None:
        exit(0)


######################################################

if __name__ == "__main__":
    init()
    readfile(filetxt)
    print("Ok. Completed!")
