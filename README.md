# FindMyGenre - Group B

## Installation Steps

All needed dependencies

- [Python3](https://www.python.org/)
- [Node.js](https://nodejs.org/en/)
- [npm](https://www.npmjs.com/)
- [yarn](https://yarnpkg.com/)
- [expo-cli](https://docs.expo.dev/get-started/installation/)
- [flask](https://flask.palletsprojects.com/en/2.2.x/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [sklearn](https://scikit-learn.org/stable/)
- [torch](https://pytorch.org/)
- [optuna](https://optuna.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [tensorflow](https://www.tensorflow.org/)
- [librosa](https://librosa.org/doc/latest/index.html)
- [scipy](https://scipy.org/)

## Download Data and Create Folder Structure

1. Download the [Metadata](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip)
2. Inside the Project folder go to `backend/` folder and create a folder called `data`
3. Inside the `data` folder extract the downloaded folder with the metadata
4. Download [Models](TODO) if you dont want to train them yourself.
5. Put Models into the `backend/` folder.

```
FindMyGenre
├── App/
├── backend/
│      ├── ...
|      ├── data/
|           ├── fma_metadata/
|                ├── ...
|                ├── ...
|                └── ...
|      ├── model
|      ├── model_conv2d
|      ├── densenet201/
|      └── efficientnetv2l/
│   ...
└── README.md
```

## Run the Project

1. Inside the `backend` folder run `$ python3 app.py`. As alternative run `$ python3 app.py -t` to train all models
2. Inside the `App` folder run `$ yarn install`
3. Inside the `App` folder run `$ expo start`
4. Download and open the [Expo Go](https://expo.dev/client) App on your Smartphone
5. Scann the QR Code to start the App (you must be in the same network)
