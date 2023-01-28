from flask import Flask, redirect, url_for, render_template, request, session
from neural_network import GenreNeuralNetwork, GenreNeuralNetwork2D, GenreNeuralNetwork2DTransferLearned
from data_handler import DataHandler
import os
import json
import pandas as pd
from flask import jsonify
from pydub import AudioSegment
import io
import math

Genres = ['Blues','Classical','Country','Easy Listening','Electronic','Experimental','Folk','Hip-Hop','Instrumental','International','Jazz','Old-Time / Historic','Pop','Rock','Soul-RnB','Spoken']

app = Flask('FindMyGenre')
data_handler = DataHandler()
data_handler.read('data/fma_metadata/')
# neural_network_2d_transfer_learning = GenreNeuralNetwork2DTransferLearned(data_handler, 10)
neural_network_2d = GenreNeuralNetwork2D(data_handler, 10)
neural_network = GenreNeuralNetwork(data_handler, 10)

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
                    url = -1
                    if str(data["track_url"][i]) != None and str(data["track_url"][i]) != "" and str(data["track_url"][i]) != "nan":
                        url = data["track_url"][i]    
                    songList.append({"artist": data["artist_name"][i], "song": data["track_title"][i], "url": url})
        except Exception as e: 
            pass
            # print(e)
            
    return jsonify(songList)

@app.route('/uploadAudio', methods=['POST'])
def uploadAudio():
    bytesOfSong = request.get_data()
    song = AudioSegment.from_file(io.BytesIO(bytesOfSong), 'm4a')
    song.export('tmpAudioRecording.mp3', format="mp3")

    prediction = neural_network.predict('tmpAudioRecording.mp3')
    prediction_2d = neural_network_2d.predict('tmpAudioRecording.mp3')
    # prediction_2d_transfer_learning = neural_network_2d_transfer_learning.predict('tmpAudioRecording.mp3')

    # Mockup
    simSongs = [{"artist": "X", "song": "XSong", "url": "https://reactnative.dev/docs/linking"}, 
                {"artist": "Y", "song": "YSong", "url": "https://reactnative.dev/docs/linking"}, 
                {"artist": "Z", "song": "ZSong", "url": "https://reactnative.dev/docs/linking"}, 
                {"artist": "X", "song": "XSong", "url": "https://reactnative.dev/docs/linking"}, 
                {"artist": "Y", "song": "YSong", "url": "https://reactnative.dev/docs/linking"}, 
                {"artist": "Z", "song": "ZSong", "url": "https://reactnative.dev/docs/linking"}, 
                {"artist": "X", "song": "XSong", "url": "https://reactnative.dev/docs/linking"}, 
                {"artist": "Y", "song": "YSong", "url": "https://reactnative.dev/docs/linking"}, 
                {"artist": "Z", "song": "ZSong", "url": "https://reactnative.dev/docs/linking"},]



    return [Genres[prediction[0]], Genres[prediction_2d[0]], Genres[prediction_2d[0]], simSongs]



@app.route('/', methods=['GET'])
@app.route('/index/', methods=['GET'])
def home():
    return 'Hello World!'

def main():

    #neural_network.searchModel()
    #neural_network.fit()
    '''

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

    '''


    app.run(host="0.0.0.0", port=12345)



if __name__=="__main__":
    main()
