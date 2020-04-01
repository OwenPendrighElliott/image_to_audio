import numpy as np
import librosa
import cv2
import IPython.display as ipd
from os import path
from tkinter import *

def img_to_audio(image=None, out_wav=None, sr=48000, hl=None, wl=None):

    if image==None or ".png" not in image:
        print("Please Specify an image file! (e.g. my_image.png)")
        return None
    elif out_wav==None or ".wav" not in out_wav:
        print("Please Specify an output file! (e.g. my_sound.wav)")
        return None
    
    if wl=="None":
        wl = None
    if hl=="None":
        hl = None

    basepath = path.dirname(__file__)
    filepath = path.abspath(path.join(basepath, 'images', image))

    print(f"Image location: {filepath}")

    img = cv2.imread(filepath)

    print(f"Read image of shape {img.shape}")

    avg_img = np.mean(img, axis=2)

    snd = librosa.griffinlim(avg_img, n_iter=64, hop_length=hl, win_length=wl)

    print(f"Output audio with {len(snd)} samples and a sample rate of {sr} Hz")

    librosa.output.write_wav(f"audio/{out_wav}", snd, sr)

# GUI 
window = Tk()

window.title("Image to Audio Converter")

window.geometry('400x200')

lbl = Label(window, text="Image name:")
lbl.grid(column=0, row=1, sticky=W)
image = Entry(window,width=30)
image.grid(column=1, row=1, sticky=W)

lbl = Label(window, text="Output wav name:")
lbl.grid(column=0, row=2, sticky=W)
out_file = Entry(window,width=30)
out_file.grid(column=1, row=2, sticky=W)

lbl = Label(window, text="Sample Rate (optional):")
lbl.grid(column=0, row=3, sticky=W)
sr = Entry(window,width=10)
sr.insert(END, '48000')
sr.grid(column=1, row=3, sticky=W)

lbl = Label(window, text="Hop Length (optional):")
lbl.grid(column=0, row=4, sticky=W)
hl = Entry(window,width=10)
hl.insert(END, 'None')
hl.grid(column=1, row=4, sticky=W)

lbl = Label(window, text="Window Length (optional):")
lbl.grid(column=0, row=5, sticky=W)
wl = Entry(window,width=10)
wl.insert(END, 'None')
wl.grid(column=1, row=5, sticky=W)

def btn_helper():
    print("Button Press")
    img_to_audio(image.get(), out_file.get(), int(sr.get()), hl.get(), wl.get())

btn = Button(window, text="Convert!", command=btn_helper)

btn.grid(column=0, row=10, sticky=W)

window.mainloop()