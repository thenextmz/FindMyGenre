from flask import Flask, redirect, url_for, render_template, request, session
#from neural_network import GenreNeuralNetwork
from data_handler import DataHandler
import json
import pandas as pd



app = Flask('FindMyGenre')
data_handler = DataHandler()
data = pd.read_csv("../../data/fma_metadata/raw_tracks.csv")


@app.route('/getSongsByGenre', methods=['GET'])
def getSongsByGenre():
    args = request.args
    target_genre = args.get("genre")
    songList = []
    for i, genres in enumerate(data["track_genres"]):
        try:
            dict = json.loads(genres.replace("'", "\""))    
            for genre in dict:
                if genre["genre_title"] == target_genre:
                    songList.append({"artist": data["artist_name"][i], "song": data["track_title"][i]})
        except:
            pass
    return songList  


@app.route('/', methods=['GET'])
@app.route('/index/', methods=['GET'])
def home():
    return 'Hello World!'

def main():
    data_handler.read('data/fma_metadata/')

    app.run(host="0.0.0.0", port=12345)



if __name__=="__main__":
    main()
