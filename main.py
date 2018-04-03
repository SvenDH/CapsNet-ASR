import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from utils import butter_bandpass_filter, get_timit_dict, get_target, create_mel_filter, pretty_spectrogram, make_mel
import matplotlib.pyplot as plt

# preprocess feature data
rate, data = wavfile.read("data/LDC93S1.wav")
data = butter_bandpass_filter(data, 500, 7000, rate, order=1)
wav_spectrogram = pretty_spectrogram(data.astype('float64'), fft_size=2048,
                                     step_size=128, thresh=4)
mel_filter, mel_inversion_filter = create_mel_filter(fft_size=2048,
                                                     n_freq_components=64,
                                                     start_freq=300,
                                                     end_freq=8000)
mel_spec = make_mel(wav_spectrogram, mel_filter, shorten_factor=10)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,4))

cax = ax.matshow(mel_spec, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
fig.colorbar(cax)
plt.title('mel Spectrogram')
plt.show()

# preprocess label data
labels = get_timit_dict("phonedict.txt")
Y = get_target("data/LDC93S1.phn", labels, data.shape[0])

# TODO: build and train model