import pandas as pd
import numpy as np
from sklearn import preprocessing
import sklearn as skl

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

    def read(self, path):
        try:
            dataset_target = pd.read_csv(path+'tracks.csv', index_col=0, header=[0,1], low_memory=False)
            dataset_feature = pd.read_csv(path+'features.csv', index_col=0, header=[0,1,2], low_memory=False)
        except:
            print('Could not find file')
            return -1

        # Create indexes for different dataframes
        small = dataset_target['set', 'subset'] <= 'small'
        train = dataset_target['set', 'split'] == 'training'
        val = dataset_target['set', 'split'] == 'validation'
        test = dataset_target['set', 'split'] == 'test'

        # Create test and train datasets
        self._genres_train = dataset_target.loc[train, ('track', 'genre_top')]
        self._genres_test = dataset_target.loc[test, ('track', 'genre_top')]

        self._mfcc_train = dataset_feature.loc[train, 'mfcc']
        self._mfcc_test = dataset_feature.loc[test, 'mfcc']

        train_data = pd.concat([self._mfcc_train, self._genres_train], axis=1)
        test_data = pd.concat([self._mfcc_test, self._genres_test], axis=1)

        # Drop lines with nan
        train_data = train_data.dropna()
        test_data = test_data.dropna()

        self._mfcc_train = train_data
        self._genres_train = self._mfcc_train.pop(('track', 'genre_top'))

        self._mfcc_test = test_data
        self._genres_test = self._mfcc_test.pop(('track', 'genre_top'))

        # Replace unique names of genre to a unique number
        unique_targets = np.unique(self._genres_train.values)
        self._different_genres_names = unique_targets
        self._different_genres = [i for i in range(len(self._different_genres_names))]
        print(f"Genres: {self._different_genres_names}\nGenre numbers: {self._different_genres}")

        for index, key in enumerate(unique_targets):
            self._target_mapping[key] = index
        self._genres_train = self._genres_train.replace(self._target_mapping.keys(), self._target_mapping.values())
        self._genres_test = self._genres_test.replace(self._target_mapping.keys(), self._target_mapping.values())

        # Shuffle data
        self._mfcc_train, self._genres_train = skl.utils.shuffle(self._mfcc_train, self._genres_train, random_state=9876)

        scaler = skl.preprocessing.StandardScaler(copy=False)
        scaler.fit_transform(self._mfcc_train)
        scaler.transform(self._mfcc_test)

        self._mfcc_all = dataset_feature.mfcc

    @property
    def mfcc_all(self):
        return self._mfcc_all

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



