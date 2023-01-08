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

    '''zcr = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)

    cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7 * 12, tuning=None))
    assert \
    cqt.shape[
        0] == 7 * 12
    assert np.ceil(len(x) / 512) <= \
           cqt.shape[
               1] <= np.ceil(len(x) / 512) + 1

    chroma_cqt = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)

    chroma_cens = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)

    tonnetz = librosa.feature.tonnetz(chroma=chroma_cens)

    del cqt
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    assert \
    stft.shape[
        0] == 1 + 2048 // 2
    assert np.ceil(len(x) / 512) <= \
           stft.shape[
               1] <= np.ceil(len(x) / 512) + 1
    del x

    chroma_stft = librosa.feature.chroma_stft(S=stft ** 2, n_chroma=12)
    chroma_stft = calcStats(chroma_stft)

    rmse = librosa.feature.rms(S=stft)[0]
    rmse = calcStats(rmse, 0)

    spectral_centroid = librosa.feature.spectral_centroid(S=stft)
    spectral_centroid = calcStats(spectral_centroid)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft)
    spectral_contrast = librosa.feature.spectral_contrast(S=stft, n_bands=6)
    spectral_rolloff = librosa.feature.spectral_rolloff(S=stft)'''

    return mfcc
    #return np.concatenate((mfcc,chroma_stft,spectral_centroid,rmse), axis=None)



