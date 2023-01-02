from flask import Flask, redirect, url_for, render_template, request, session
#from neural_network import GenreNeuralNetwork
from data_handler import DataHandler
import json
import pandas as pd
import os
from flask import jsonify
from pydub import AudioSegment
import io


app = Flask('FindMyGenre')
data_handler = DataHandler()
data = pd.read_csv(os.getcwd() + "/data/fma_metadata/raw_tracks.csv")


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
    return jsonify(songList)

@app.route('/uploadAudio', methods=['POST'])
def uploadAudio():
    bytesOfSong = request.get_data()
    song = AudioSegment.from_file(io.BytesIO(bytesOfSong), 'm4a')
    song.export('tmpAudioRecording.mp3', format="mp3")

    # samplerate 44100
    # TODO: process data
    return "Rock"
        
        

@app.route('/', methods=['GET'])
@app.route('/index/', methods=['GET'])
def home():
    return 'Hello World!'

def main():
    data_handler.read('data/fma_metadata/')

    app.run(host="0.0.0.0", port=12345)



if __name__=="__main__":
    main()
