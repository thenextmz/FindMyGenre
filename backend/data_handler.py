import pandas as pd
import numpy as np
import os
import librosa
import sklearn as skl

class DataHandler:
    def __init__(self):
        # Target
        self._genres_train = pd.DataFrame()
        self._genres_test = pd.DataFrame()
        # Feature
        self._mfcc_train = pd.DataFrame()
        self._mfcc_test = pd.DataFrame()

    def read(self, path):
        try:
            dataset_target = pd.read_csv(path+'tracks.csv', index_col=0, header=[0,1], low_memory=False)
            dataset_feature = pd.read_csv(path+'features.csv', index_col=0, header=[0,1,2], low_memory=False)
        except:
            print('Could not find file')
            return -1

        dataset_target.dropna()

        small = dataset_target['set', 'subset'] <= 'small'
        train = dataset_target['set', 'split'] == 'training'
        val = dataset_target['set', 'split'] == 'validation'
        test = dataset_target['set', 'split'] == 'test'

        self._genres_train = dataset_target.loc[small & train, ('track', 'genre_top')]
        self._genres_test = dataset_target.loc[small & test, ('track', 'genre_top')]
        self._mfcc_train = dataset_feature.loc[small & train, 'mfcc']
        self._mfcc_test = dataset_feature.loc[small & test, 'mfcc']

        // TODO Michael remove lines with nan

        self._mfcc_train, self._genres_train = skl.utils.shuffle(self._mfcc_train, self._genres_train, random_state=9876)

        scaler = skl.preprocessing.StandardScaler(copy=False)
        scaler.fit_transform(self._mfcc_train)
        scaler.transform(self._mfcc_test)

        '''clf = skl.svm.SVC()
        clf.fit(self._mfcc_train, self._genres_train)
        score = clf.score(self._mfcc_test, self._genres_test)
        print('Accuracy: {:.2%}'.format(score))'''

    def __getitem__(self, item):
        return self._genres[item]

    @property
    def genres_train(self):
        return self._genres_train

    @property
    def genres_test(self):
        return self._genres_test

    @property
    def mfcc_train(self):
        return self._mfcc_train

    @property
    def mfcc_test(self):
        return self._mfcc_test



