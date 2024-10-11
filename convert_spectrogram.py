import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

import _thread
import time


def convert_to_spectrogram(dir):
    asshole = "./data/{}".format(dir)
    for _, _, files in os.walk(asshole + "/"):
        for file in files:
            full_name = "./data/{}/{}".format(dir, file)
            save_name = "./train/{}/{}.png".format(dir, file)
            if os.path.isfile(save_name):
                print("Skipping {}".format(save_name))
                continue
            x, sr = librosa.load(full_name)
            S = librosa.feature.melspectrogram(x)
            S_dB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_dB)
            plt.tight_layout()
            plt.savefig(save_name)
            plt.close()


def aaa(directory="./data"):
    for _, dirs, _ in os.walk(directory):
        for dir in dirs:
            convert_to_spectrogram(dir)


aaa()
