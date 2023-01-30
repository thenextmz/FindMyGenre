import pandas as pd
import numpy as np
from sklearn import preprocessing
import sklearn as skl
from sklearn.model_selection import train_test_split

class DataHandler:
    def __init__(self):
        # Target
        self._genres_train = pd.DataFrame()
        self._genres_test = pd.DataFrame()
        # Feature
        self._mfcc_train = pd.DataFrame()
        self._mfcc_test = pd.DataFrame()

        self._target_mapping = {}

        self._different_genres_names = []
        self._different_genres = []

        self._mfcc_all = pd.DataFrame()
        self._genre_all = pd.DataFrame()

    def read(self, path):
        try:
            dataset_target = pd.read_csv(path+'tracks.csv', index_col=0, header=[0,1], low_memory=False)
            dataset_feature = pd.read_csv(path+'features.csv', index_col=0, header=[0,1,2], low_memory=False)
        except:
            print('Could not find file')
            return -1



        # Create test and train datasets
        temp = pd.concat([dataset_feature['mfcc'], dataset_target[('track', 'genre_top')]], axis=1).dropna()
        genres = temp.pop(('track', 'genre_top'))
        mfccs = temp

        self._mfcc_train, self._mfcc_test, self._genres_train, self._genres_test = train_test_split(mfccs, genres, train_size=0.8, random_state=4242, stratify=genres)

        # Replace unique names of genre to a unique number
        unique_targets = np.unique(self._genres_train.values)
        self._different_genres_names = unique_targets
        self._different_genres = [i for i in range(len(self._different_genres_names))]
        print(f"Genres: {self._different_genres_names}\nGenre numbers: {self._different_genres}")

        for index, key in enumerate(unique_targets):
            self._target_mapping[key] = index
        
        self._genres_train = self._genres_train.replace(self._target_mapping.keys(), self._target_mapping.values())
        self._genres_test = self._genres_test.replace(self._target_mapping.keys(), self._target_mapping.values())

        resampled = pd.DataFrame()
        all = self._mfcc_train
        all[('track', 'genre_top')] = self._genres_train
        for i in range(len(unique_targets)):
            temp = all[all[('track', 'genre_top')] == i]
            resamp = skl.utils.resample(temp, replace=True, n_samples=3000, random_state=123)
            if resampled.empty:
                resampled = resamp
            else:
                resampled = pd.concat([resampled, resamp])

        self._genres_train = resampled.pop(('track', 'genre_top'))
        self._mfcc_train = resampled

        # Shuffle data
        self._mfcc_train, self._genres_train = skl.utils.shuffle(self._mfcc_train, self._genres_train, random_state=9876)

        self._mfcc_all = mfccs
        self._genre_all = genres

    @property
    def mfcc_all(self):
        return self._mfcc_all

    @property
    def genre_all(self):
        return self._genre_all

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

    @property
    def different_genres_names(self):
        return self._different_genres_names

    @property
    def different_genres(self):
        return self._different_genres



