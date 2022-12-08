import librosa
import numpy as np
import sklearn as skl

# x = audio data, sr = sample rate

def MP3toMFCC(x, sr):
    #x, sr = librosa.load(filename, sr=None, mono=True)

    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
    log_mel = librosa.logamplitude(mel)

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    mfcc = skl.preprocessing.StandardScaler().fit_transform(mfcc)
    return mfcc