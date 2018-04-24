import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from utils import get_timit_dict, get_target, pretty_spectrogram, sliding_window, get_batch_data
from python_speech_features import mfcc
import os
from model import CapsNet

labels = get_timit_dict('phonedict.txt')
batch_size = 64

rate = 16000 #16000fps - 0.0625ms per frame
stepsize = 64 #for spectogram reduction

frame_size = (int)((0.030 * rate) / stepsize) #30ms
frame_step = (int)((0.015 * rate) / stepsize) #15ms

print('Frame size: {}, frame step size: {}'.format(frame_size, frame_step))

# preprocess data
audio = []
spectograms = []
#mfccs = []
phones = []
for dirName, subdirList, fileList in os.walk('./data/TRAIN/'):
    for fname in fileList:
        if not fname.endswith('.phn') and not fname.endswith('.PHN') or (fname.startswith('SA')):
            continue

        phn_fname = dirName + fname
        wav_fname = dirName + fname[0:-4] + '.WAV'

        _, data = wavfile.read(wav_fname)

        audio.append(data)
        spectogram = pretty_spectrogram(data.astype('float64'), step_size=stepsize)
        phone_ids = get_target(phn_fname, labels, data.shape[0])
        for x, window in sliding_window(spectogram, frame_step, frame_size):
            w = window.astype(np.float32)
            spectograms.append(w)
            idx = x * stepsize + (int)(stepsize * frame_size / 2)
            phones.append(phone_ids[idx])

        #mfccs.append(mfcc(data, rate))

        print('Loaded: {}'.format(fname[0:-4]))
audio = np.concatenate(audio)
spectograms = np.expand_dims(np.stack(spectograms), axis=-1)
#mfccs = np.concatenate(mfccs)
phones = np.array(phones)

X, y = get_batch_data(spectograms, phones, batch_size, 8)
print(X.shape)
model = CapsNet(X, y)

#TODO: fix dimensions for capsule net