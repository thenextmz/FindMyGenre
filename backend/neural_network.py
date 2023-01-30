import torch
import pandas as pd
import numpy as np
import optuna
import sklearn.metrics
from mp3_to_mfcc_converter import MP3toSoundStats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import tensorflow as tf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 4
TRANSFER_HEIGHT = 35
TRANSFER_WIDTH = 32

'''
0: Blues
1: Classical
2: Country
3: Easy Listening
4: Electronic
5: Experimental
6: Folk
7: Hip-Hop
8: Instrumental
9: International
10: Jazz
11: Old-Time / Historic
12: Pop
13: Rock
14: Soul-RnB
15: Spoken
'''

class ScikitFeatDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.array, y: np.array):
        self.X = X.astype(float)
        self.y = y.astype(int)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class GenreNeuralNetwork:
    def __init__(self, data, epochs = 10):
        self._data = data
        train_mfcc = self._data.mfcc_train.to_numpy()
        train_genre = self._data.genres_train.to_numpy()

        dataset = ScikitFeatDataset(train_mfcc, train_genre)
        self._train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

        test_mfcc = self._data.mfcc_test.to_numpy()
        test_genres = self._data.genres_test.to_numpy()

        dataset = ScikitFeatDataset(test_mfcc, test_genres)
        self._val_dataloader = torch.utils.data.DataLoader(dataset)
        
        self._model = 0
        try:
            self._model = torch.load('model')
        except:
            pass

        self._epochs = epochs

        print('GenreNeuralNetwork created')

    def _train_func(self, model, train_dataloader, loss_function, optimiser):
        model.train()
        running_loss = 0
        for X, y in train_dataloader:
            optimiser.zero_grad()

            X = X.to(DEVICE, dtype=torch.float)
            y = y.to(DEVICE, dtype=torch.long)

            output = model(X)
            loss = loss_function(output, y)
            loss.backward()
            optimiser.step()
            running_loss += loss * len(X)
    
    def _val_func(self, model, val_dataloader, loss_function):
        model.eval()
        y_real = []
        outputs = []
        running_loss = 0
        with torch.no_grad():
            for X, y in val_dataloader:
                X = X.to(DEVICE, dtype=torch.float)
                y = y.to(DEVICE, dtype=torch.long)
                output = model(X)
                loss = loss_function(output, y)
                running_loss += loss * len(X)
                outputs += [output.argmax(dim=1).cpu().detach().numpy()]
                y_real += [y.cpu().detach().numpy()]
            validation_loss = running_loss / len(val_dataloader)
            accuracy = sklearn.metrics.accuracy_score(y_real, outputs)
            
        return validation_loss, accuracy, outputs, y_real
    
    def _define_model(self, trial):
        layers = []
        activation_function_selection = {"ReLU": torch.nn.ReLU(),
                                         "Softmax": torch.nn.Softmax(dim=1),
                                         "Sigmoid": torch.nn.Sigmoid(),
                                         "None": None}
        input_units = self._data.mfcc_train.to_numpy().shape[1]
        n_layers = trial.suggest_int("n_layers", 0, 12)
        for i in range(n_layers):
            units = trial.suggest_int(f"layer_{i}_units", 1, 100)
            layers.append(torch.nn.Linear(input_units, units))
            activation_function = activation_function_selection[trial.suggest_categorical(f"layer_{i}_activation", activation_function_selection.keys())]
            if activation_function != None:
                layers.append(activation_function)
            input_units = units
        classes = len(self._data.different_genres)
        output_activation = torch.nn.Softmax(dim=1)
        output_layer = torch.nn.Linear(input_units, classes)
        layers.append(output_layer)
        layers.append(output_activation)

        return torch.nn.Sequential(*layers)
    
    def _objective(self, trial):
        model = self._define_model(trial).to(DEVICE)
        loss_function = torch.nn.CrossEntropyLoss()
        lr = trial.suggest_float("lr", 1e-5, 1e-2)
        optimiser_selection = {"Adam": torch.optim.Adam(model.parameters(), lr=lr),
                               "SGD": torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)}
        optimiser = optimiser_selection[trial.suggest_categorical("optim", optimiser_selection.keys())]
        validation_loss = 0
        for i in range(EPOCHS):
            self._train_func(model, self._train_dataloader, loss_function, optimiser)
            validation_loss_ep, accuracy, outputs, y_real = self._val_func(model, self._val_dataloader, loss_function)
            validation_loss += validation_loss_ep
        return validation_loss

    def searchModel(self):
        study = optuna.create_study(study_name="FindMyGenre", direction="minimize")
        study.optimize(self._objective, n_trials=50)
        print(study.best_trial)
        print(study.best_params)

    def fit(self):
        input_units = self._data.mfcc_train.to_numpy().shape[1]
      
        output_units = len(np.unique(self._data.genres_train))

        layer = [torch.nn.Linear(input_units, 300),
            torch.nn.ELU(),
            torch.nn.Linear(300, 150),
            torch.nn.ELU(),
            torch.nn.Linear(150, 80),
            torch.nn.ELU(),
            torch.nn.Linear(80, output_units),
            torch.nn.Sigmoid()]

        model = torch.nn.Sequential(*layer).to(DEVICE)
        loss_function = torch.nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.0003)

        validation_loss = 0
        print("Start fitting")
        accuracies = []
        losses = []
        outputs = []
        y_real = []
        for i in range(self._epochs):
            self._train_func(model, self._train_dataloader, loss_function, optimiser)
            validation_loss_ep, accuracy, outputs, y_real = self._val_func(model, self._val_dataloader, loss_function)
            validation_loss += validation_loss_ep
            accuracies.append(accuracy)
            losses.append(validation_loss_ep)
            print(f"{str(i+1)} / {self._epochs} (accuracy: {str(accuracy)})")

        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots(3,1, figsize=(10,18))
        ax[0].plot(accuracies)
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel('accuracy')
        ax[1].plot(losses)
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel('loss')

        conf_matrix = sklearn.metrics.confusion_matrix(y_real, outputs)

        sns.heatmap(conf_matrix, ax=ax[2])
        ax[2].set_xlabel('true value')
        ax[2].set_ylabel('predicted value')
        plt.show()
        plt.savefig('graphics.pdf')

        self._model = model
        torch.save(self._model, 'model')

    def predict(self, path: str):
        if self._model is None:
            print("Train network before predicting")
            return

        sound_stats = MP3toSoundStats(path)
        sound_stats = sound_stats.reshape((1,-1))
        res =  self._model(torch.Tensor(sound_stats))
        res = res.argmax(dim=1).cpu().detach().numpy()
        return res[0]

    def predict_mfcc(self, mfcc):
        if self._model is None:
            print("Train network before predicting")
            return
        res =  self._model(torch.Tensor(mfcc))
        print(res)
        res = res.argmax(dim=0).cpu().detach().numpy()
        return res

class GenreNeuralNetwork2D:
    def __init__(self, data, epochs = 10):
        self._data = data
        train_mfcc = self._data.mfcc_train.to_numpy()
        train_mfcc = train_mfcc.reshape((train_mfcc.shape[0], 1, 14, 10))
        train_genre = self._data.genres_train.to_numpy()
        dataset = ScikitFeatDataset(train_mfcc, train_genre)

        self._train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

        test_mfcc = self._data.mfcc_test.to_numpy()
        test_mfcc = test_mfcc.reshape((test_mfcc.shape[0], 1, 14, 10))
        dataset = ScikitFeatDataset(test_mfcc, self._data.genres_test.to_numpy())

        self._val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
        
        self._model = 0
        try:
            self._model = torch.load('model_conv2d')
        except:
            pass
        self._epochs = epochs

        print('GenreNeuralNetwork2D created')

    def _train_func(self, model, train_dataloader, loss_function, optimiser):
        model.train()
        running_loss = 0
        for X, y in train_dataloader:
            optimiser.zero_grad()
            X = X.to(DEVICE, dtype=torch.float)
            y = y.to(DEVICE, dtype=torch.long)
            output = model(X)
            loss = loss_function(output, y)
            loss.backward()
            optimiser.step()
            running_loss += loss * len(X)
    
    def _val_func(self, model, val_dataloader, loss_function):
        model.eval()
        y_real = []
        outputs = []
        running_loss = 0
        with torch.no_grad():
            for X, y in val_dataloader:
                X = X.to(DEVICE, dtype=torch.float)
                y = y.to(DEVICE, dtype=torch.long)

                output = model(X)
                loss = loss_function(output, y)
                running_loss += loss * len(X)
                outputs += output.argmax(dim=1).cpu().detach().numpy().tolist()
                y_real += y.cpu().detach().numpy().tolist()
            validation_loss = running_loss / len(val_dataloader)
            y_real = np.array(y_real)
            y_real.flatten()
            outputs = np.array(outputs)
            outputs.flatten()
            accuracy = sklearn.metrics.accuracy_score(y_real, outputs)
        return validation_loss, accuracy, outputs, y_real
    
    def fit(self):
        output_units = len(np.unique(self._data.genres_train))

        layer = [
            torch.nn.Conv2d(1, 64, kernel_size=3),
            torch.nn.Conv2d(64, 32, kernel_size=3),
            torch.nn.Conv2d(32, 16, kernel_size=3),
            torch.nn.MaxPool2d(kernel_size=3),
            torch.nn.Flatten(),
            torch.nn.Linear(32, output_units),
            torch.nn.Sigmoid()]


        model = torch.nn.Sequential(*layer).to(DEVICE)
        loss_function = torch.nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.0003)

        validation_loss = 0
        print("Start fitting")
        accuracies = []
        losses = []
        outputs = []
        y_real = []
        for i in range(self._epochs):
            self._train_func(model, self._train_dataloader, loss_function, optimiser)
            validation_loss_ep, accuracy, outputs, y_real = self._val_func(model, self._val_dataloader, loss_function)
            validation_loss += validation_loss_ep
            accuracies.append(accuracy)
            losses.append(validation_loss_ep)
            print(f"{str(i+1)} / {self._epochs} (accuracy: {str(accuracy)})")

        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots(3,1, figsize=(10,18))
        ax[0].plot(accuracies)
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel('accuracy')
        ax[1].plot(losses)
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel('loss')

        conf_matrix = sklearn.metrics.confusion_matrix(y_real, outputs)

        sns.heatmap(conf_matrix, ax=ax[2], norm=LogNorm())
        ax[2].set_xlabel('true value')
        ax[2].set_ylabel('predicted value')
        plt.show()
        plt.savefig('graphics.pdf')

        self._model = model
        torch.save(self._model, 'model_conv2d')

    def predict(self, path: str):
        if self._model is None:
            print("Train network before predicting")
            return

        sound_stats = MP3toSoundStats(path)
        sound_stats = sound_stats.reshape((1,1,14,10))

        res =  self._model(torch.Tensor(sound_stats))
        res = res.argmax(dim=1).cpu().detach().numpy()
        return res[0]

    def predict_mfcc(self, mfcc):
        if self._model is None:
            print("Train network before predicting")
            return

        sound_stats = mfcc
        sound_stats = sound_stats.reshape((1,1,14,10))

        res =  self._model(torch.Tensor(sound_stats))
        res = res.argmax(dim=1).cpu().detach().numpy()
        return res[0]

class GenreNeuralNetwork2DTransferLearned:
    def __init__(self, data, epochs = 10):
        self._data = data
        self._train_mfcc = self._data.mfcc_train.to_numpy()
        self._train_mfcc = np.hstack((self._train_mfcc, self._train_mfcc))
        self._train_mfcc = np.hstack((self._train_mfcc, self._train_mfcc))
        self._train_mfcc = np.hstack((self._train_mfcc, self._train_mfcc))
        self._train_mfcc = np.repeat(self._train_mfcc[..., np.newaxis], 3, -1)
        self._train_mfcc = self._train_mfcc.reshape((self._train_mfcc.shape[0], TRANSFER_HEIGHT, TRANSFER_WIDTH, 3))
        self._train_mfcc = tf.keras.applications.densenet.preprocess_input(self._train_mfcc)

        self._train_genre = self._data.genres_train.to_numpy()
        self._train_genre = tf.keras.utils.to_categorical(self._train_genre, dtype="uint8")

        self._test_mfcc = self._data.mfcc_test.to_numpy()
        self._test_mfcc = np.hstack((self._test_mfcc, self._test_mfcc))
        self._test_mfcc = np.hstack((self._test_mfcc, self._test_mfcc))
        self._test_mfcc = np.hstack((self._test_mfcc, self._test_mfcc))
        self._test_mfcc = np.repeat(self._test_mfcc[..., np.newaxis], 3, -1)
        self._test_mfcc = self._test_mfcc.reshape((self._test_mfcc.shape[0], TRANSFER_HEIGHT, TRANSFER_WIDTH, 3))
        self._test_mfcc = tf.keras.applications.densenet.preprocess_input(self._test_mfcc)

        self._test_genre = self._data.genres_test.to_numpy()

        self._model = 0
        try:
            self._model = tf.keras.models.load_model('densenet201')
        except:
            pass
        self._epochs = epochs

        print('GenreNeuralNetwork2DTransferLearned created')

    def fit(self):
        output_units = len(np.unique(self._data.genres_train))

        base = tf.keras.applications.DenseNet201(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=(TRANSFER_HEIGHT,TRANSFER_WIDTH,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax"
        )

        inputs = base.inputs
        model = base.output
        model = tf.keras.layers.GlobalAveragePooling2D()(model)
        model = tf.keras.layers.Dropout(0.2)(model)
        model = tf.keras.layers.Dense(100, activation='elu')(model)
        model = tf.keras.layers.Dense(output_units, activation='sigmoid')(model)
        model = tf.keras.Model(inputs=inputs, outputs=model)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'])

        print("Start fitting")

        model.fit(self._train_mfcc, self._train_genre, batch_size=64, epochs=self._epochs, validation_split=0.1)
        prediction = model.predict(self._test_mfcc, batch_size=128)
        prediction = np.squeeze(prediction)
        prediction = prediction.argmax(axis=1)

        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots(1,1, figsize=(10,18))

        conf_matrix = sklearn.metrics.confusion_matrix(self._test_genre, prediction)

        sns.heatmap(conf_matrix, ax=ax)
        ax.set_xlabel('true value')
        ax.set_ylabel('predicted value')
        plt.show()
        plt.savefig('graphics.pdf')

        self._model = model
        self._model.save('densenet201')

    def predict(self, path):
        if self._model is None:
            print("Train network before predicting")
            return

        sound_stats = MP3toSoundStats(path)
        sound_stats = np.hstack((sound_stats, sound_stats))
        sound_stats = np.hstack((sound_stats, sound_stats))
        sound_stats = np.hstack((sound_stats, sound_stats))
        sound_stats = np.repeat(sound_stats[..., np.newaxis], 3, -1)
        sound_stats = sound_stats.reshape((1, TRANSFER_HEIGHT, TRANSFER_WIDTH, 3))

        res =  self._model(sound_stats)
        res = np.argmax(res)
        return res

