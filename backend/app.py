import sys
from flask import Flask, request, jsonify
from neural_network import GenreNeuralNetwork, GenreNeuralNetwork2D, GenreNeuralNetwork2DTransferLearned
from data_handler import DataHandler
import os
import json
import pandas as pd
from flask import jsonify
from pydub import AudioSegment
import io
import math
from sklearn.metrics.pairwise import cosine_similarity
from mp3_to_mfcc_converter import MP3toSoundStats
import numpy as np

Genres = ['Blues','Classical','Country','Easy Listening','Electronic','Experimental','Folk','Hip-Hop','Instrumental','International','Jazz','Old-Time / Historic','Pop','Rock','Soul-RnB','Spoken']

app = Flask('FindMyGenre')
data_handler = DataHandler()
data_handler.read('data/fma_metadata/')
all_features_ids = data_handler.mfcc_all
neural_network_2d_transfer_learning = GenreNeuralNetwork2DTransferLearned(data_handler, 10)
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

@app.route('/getSongsByGenreAndSong', methods=['GET'])
def getSongsByGenreAndSong():
    bytesOfSong = request.get_data()
    song = AudioSegment.from_file(io.BytesIO(bytesOfSong), 'm4a')
    song.export('tmpAudioRecording.mp3', format="mp3")

    mfcc_song = MP3toSoundStats('tmpAudioRecording.mp3')
    cos_sim = cosine_similarity(all_features_ids, mfcc_song)
    max_indices = np.argpartition(cos_sim, -10)[-10:]
    song_indices = all_features_ids.iloc[max_indices].index
    songs = data.loc[song_indices]
    songList = []
    for song in songs:
        songList.append({"artist": song.artist_name, "song": song.track_title})

    return jsonify(songList)

@app.route('/uploadAudio', methods=['POST'])
def uploadAudio():
    bytesOfSong = request.get_data()
    song = AudioSegment.from_file(io.BytesIO(bytesOfSong), 'm4a')
    song.export('tmpAudioRecording.mp3', format="mp3")

    #prediction = neural_network.predict('tmpAudioRecording.mp3')
    #prediction_2d = neural_network_2d.predict('tmpAudioRecording.mp3')
    #prediction_2d_transfer_learning = neural_network_2d_transfer_learning.predict('tmpAudioRecording.mp3')
    return 0#Genres[prediction_2d]#Genres[prediction], Genres[prediction_2d], Genres[prediction_2d_transfer_learning]



@app.route('/', methods=['GET'])
@app.route('/index/', methods=['GET'])
def home():
    return 'Hello World!'

def main():

    neural_network_2d.fit()

    #neural_network_2d_transfer_learning.fit()

    '''
    result = neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/fma_small/000/000194.mp3')
    print(result)
    result = neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/fma_small/000/000193.mp3')
    print(result)
    result = neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/fma_small/000/000190.mp3')
    print(result)
    result = neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/fma_small/000/000140.mp3')
    print(result)
    result = neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/fma_small/000/000141.mp3')
    print(result)
    result = neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/other_songs/western-125865.mp3')
    print(result)
    result = neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/other_songs/midnight-blues-21179.mp3')
    print(result)
    result = neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/other_songs/Hard-Official-.mp3')
    print(result)
    result = neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/other_songs/Lord-McDeath.mp3')
    print(result)
    result = neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/other_songs/Energetic-Indie-Rock.mp3')
    print(result)
    result = neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/other_songs/Maximalism.mp3')
    print(result)
    result = neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/other_songs/Upbeat-Forever.mp3')
    print(result)
    '''




    app.run(host="0.0.0.0", port=12345)



if __name__=="__main__":
    main()
