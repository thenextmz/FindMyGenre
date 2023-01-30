import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.stats as st
from scipy.stats import kurtosis, skew
import glob
import os

# x = audio data, sr = sample rate

def calcStats(feature, axis=1):
    kurtosis = st.kurtosis(feature, axis=axis)
    max = np.max(feature, axis=axis)
    mean = np.mean(feature, axis=axis)
    median = np.median(feature, axis=axis)
    min = np.min(feature, axis=axis)
    skew = st.skew(feature, axis=axis)
    std = np.std(feature, axis=axis)

    return np.concatenate((kurtosis,max, mean, median,min,skew,std), axis=None)

def MP3toSoundStats(path):
    x, sr = librosa.load(path, sr=None, mono=True)

    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    mfcc = calcStats(mfcc)

    return mfcc



