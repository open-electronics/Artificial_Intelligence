import face_recognition as fr
import time, os, sys
""" See help (end file)"""
import argparse
from tkinter import filedialog as fd, Tk
from tkinter import messagebox as msg

mode = "large"
#mode = "small"
jitter = 1

def choseFile(text):
    win = Tk()
    win.withdraw()
    file = fd.askopenfile(title=text, filetypes=[("Image file", (".jpg"))])
    if file == None:
        return None
    else:
        filephoto = file.name
    return filephoto

def cycle():
    while True:
        basephoto = choseFile("Base Photo for comparison")
        if basephoto is None:
            break
        baseimage = fr.load_image_file(basephoto)
        t0 = time.perf_counter()
        basecode = fr.face_encodings(baseimage, num_jitters=jitter, model=mode)[0]
        dt = time.perf_counter() - t0
        print("Code time: {:.6f}".format(dt))
        while True:
            comparephoto = choseFile("Photo to compare")
            if comparephoto is None:
                break
            compimage = fr.load_image_file(comparephoto)
            compcode = fr.face_encodings(compimage, num_jitters=jitter, model=mode)
            t0 = time.perf_counter()
            res = fr.face_distance(basecode, compcode)
            dt = time.perf_counter() - t0
            print("Time {:.6f}".format(dt))
            msg.showinfo("Comparison result", message= f"{(1-res[0]):.3f}")
            for v in res:
                print("Confidence {:.6f}".format(1-v))

if __name__ == "__main__":
    helptext="""
    Verify face photos. \n
    Program starts file selecting window for reference photo. \n
    Then it starts new file selecting window for comparing photo.\n
    Result is showed by a message dialog and printed on console.\n
    Also timing is printed.
    """
    parse = argparse.ArgumentParser(description=helptext)
    par = parse.parse_args()
    cycle()
