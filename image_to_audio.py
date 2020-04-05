import numpy as np
import librosa
import cv2
import IPython.display as ipd
from os import path

sr = 48000
hl = None
wl = None
image = "test1.png"
out_wav = "test1.wav"

def img_to_audio(image, out_wav, sr, hl, wl):
    basepath = path.dirname(__file__)
    filepath = path.abspath(path.join(basepath, 'images', image))

    print(f"Image location: {filepath}")

    img = cv2.imread(filepath)

    print(f"Read image of shape {img.shape}")

    avg_img = np.mean(img, axis=2)

    snd = librosa.griffinlim(avg_img, n_iter=64, hop_length=hl, win_length=wl)

    print(f"Output audio with {len(snd)} samples and a sample rate of {sr} Hz")

    filepath = path.abspath(path.join(basepath, 'audio', out_wav))
    librosa.output.write_wav(out_wav, snd, sr)


img_to_audio(image, out_wav, sr, hl, wl)