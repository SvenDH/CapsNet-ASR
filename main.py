import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from utils import butter_bandpass_filter, get_timit_dict, get_target, create_mel_filter, pretty_spectrogram, make_mel
from python_speech_features import mfcc
import os

dirlist = "./data/"
labels = get_timit_dict("phonedict.txt")

# preprocess data
spectograms = []
mfccs = []
phones = []
for d in dirlist:
    for dirName, subdirList, fileList in os.walk("./data/TRAIN/"):
        for fname in fileList:
            if not fname.endswith('.PHN') or (fname.startswith("SA")):
                continue

            phn_fname = dirName + '\\' + fname
            wav_fname = dirName + '\\' + fname[0:-4] + '.WAV'

            rate, data = wavfile.read(wav_fname)
            data = butter_bandpass_filter(data, 500, 7999, rate, order=1)

            spectograms.append(pretty_spectrogram(data.astype('float64')))
            mfccs.append(mfcc(data, rate))
            phones.append(get_target(phn_fname, labels, data.shape[0]))

# TODO: build and train model