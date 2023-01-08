from flask import Flask, redirect, url_for, render_template, request, session
from neural_network import GenreNeuralNetwork
from data_handler import DataHandler
import os

app = Flask('FindMyGenre')
data_handler = DataHandler()
data_handler.read('data/fma_metadata/')
neural_network = GenreNeuralNetwork(data_handler)

@app.route('/', methods=['GET'])
@app.route('/index/', methods=['GET'])
def home():
    return 'Hello World!'

def main():

    #neural_network.searchModel()
    #neural_network.fit()

    result = neural_network.predict(os.getcwd() + '/data/fma_small/000/000194.mp3')
    print(result)
    result = neural_network.predict(os.getcwd() + '/data/fma_small/000/000193.mp3')
    print(result)
    result = neural_network.predict(os.getcwd() + '/data/fma_small/000/000190.mp3')
    print(result)
    result = neural_network.predict(os.getcwd() + '/data/fma_small/000/000140.mp3')
    print(result)
    result = neural_network.predict(os.getcwd() + '/data/fma_small/000/000141.mp3')
    print(result)
    result = neural_network.predict(os.getcwd() + '/data/other_songs/western-125865.mp3')
    print(result)
    result = neural_network.predict(os.getcwd() + '/data/other_songs/midnight-blues-21179.mp3')
    print(result)
    result = neural_network.predict(os.getcwd() + '/data/other_songs/Hard-Official-.mp3')
    print(result)
    result = neural_network.predict(os.getcwd() + '/data/other_songs/Lord-McDeath.mp3')
    print(result)
    result = neural_network.predict(os.getcwd() + '/data/other_songs/Energetic-Indie-Rock.mp3')
    print(result)
    result = neural_network.predict(os.getcwd() + '/data/other_songs/Maximalism.mp3')
    print(result)
    result = neural_network.predict(os.getcwd() + '/data/other_songs/Upbeat-Forever.mp3')
    print(result)


    app.run(host="0.0.0.0", port=12345)



if __name__=="__main__":
    main()
