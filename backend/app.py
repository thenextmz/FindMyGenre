from flask import Flask
from neural_network import GenreNeuralNetwork
from data_handler import DataHandler

app = Flask('FindMyGenre')
data_handler = DataHandler()

def main():
    data_handler.read('data/fma_metadata/')

    @app.route('/')
    def home():
        return 'Hello World!'

    app.run(port=9876)



if __name__=="__main__":
    main()
