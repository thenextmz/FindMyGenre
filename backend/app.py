from flask import Flask, redirect, url_for, render_template, request, session
from neural_network import GenreNeuralNetwork
from data_handler import DataHandler

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
    neural_network.fit()

    app.run(host="0.0.0.0", port=12345)



if __name__=="__main__":
    main()
