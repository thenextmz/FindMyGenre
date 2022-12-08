from flask import Flask, redirect, url_for, render_template, request, session
from neural_network import GenreNeuralNetwork
from data_handler import DataHandler

app = Flask('FindMyGenre')
data_handler = DataHandler()


@app.route('/', methods=['GET'])
@app.route('/index/', methods=['GET'])
def home():
    return 'Hello World!'

def main():
    data_handler.read('data/fma_metadata/')

    app.run(port=9876)



if __name__=="__main__":
    main()
