import torch
import pandas as pd
import numpy as np
import optuna
import sklearn.metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 4


class ScikitFeatDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.array, y: np.array):
        self.X = X.astype(float)
        self.y = y.astype(int)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class GenreNeuralNetwork:
    def __init__(self, data):
        self._model = None
        self._data = data
        dataset = ScikitFeatDataset(self._data.mfcc_train.to_numpy(), self._data.genres_train.to_numpy())
        self._train_dataloader = torch.utils.data.DataLoader(dataset)
        dataset = ScikitFeatDataset(self._data.mfcc_test.to_numpy(), self._data.genres_test.to_numpy())
        self._val_dataloader = torch.utils.data.DataLoader(dataset)

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
        input_units = self._data.mfcc_train.shape[1]
        output_units = len(np.unique(self._data.genres_train))

        # 0.555
        #layer = [torch.nn.Linear(input_units, 61),
        #    torch.nn.Sigmoid(),
        #    torch.nn.Linear(61, output_units),
        #    torch.nn.Softmax(dim=1)]

        # 0.583
        layer = [
            torch.nn.Linear(input_units, 61),
            torch.nn.Sigmoid(),
            torch.nn.Linear(61, output_units)]


        model = torch.nn.Sequential(*layer).to(DEVICE)
        loss_function = torch.nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.00036570505894471393)

        validation_loss = 0
        for _ in range(10):
            self._train_func(model, self._train_dataloader, loss_function, optimiser)
            validation_loss_ep, accuracy, outputs, y_real = self._val_func(model, self._val_dataloader, loss_function)
            validation_loss += validation_loss_ep

        self._model = model

    def predict(self, x):
        if self._model is None:
            print("Train network before predicting")
            return


