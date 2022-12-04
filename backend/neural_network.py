import torch
import pandas as pd
import numpy as np
import optuna
import sklearn.metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GenreNeuralNetwork:
    def __init__(self):
        pass

    def searchModel(self, dataset):
        pass

    def train(self, X, y):
        pass

    def predict(self):
        pass
