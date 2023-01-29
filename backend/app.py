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

    return jsonify(songList)

@app.route('/getSongsByGenreAndSong', methods=['GET'])
def getSongsByGenreAndSong():
    # bytesOfSong = request.get_data()
    # song = AudioSegment.from_file(io.BytesIO(bytesOfSong), 'm4a')
    # song.export('tmpAudioRecording.mp3', format="mp3")

    mfcc_song = MP3toSoundStats('tmpAudioRecording.mp3')
    cos_sim = cosine_similarity(all_features_ids, [mfcc_song])
    cos_sim = cos_sim.flatten()
    max_indices = np.argpartition(cos_sim, -10)[-10:]
    song_indices = all_features_ids.iloc[max_indices].index
    songs = data.loc[data.track_id.isin(song_indices)]
    songList = []
    for song_index in range(len(songs)):
        url = -1
        if str(songs.iloc[song_index].track_url) != None and str(songs.iloc[song_index].track_url) != "" and str(songs.iloc[song_index].track_url) != "nan":
            url = songs.iloc[song_index].track_url
        songList.append({"artist": songs.iloc[song_index].artist_name, "song": songs.iloc[song_index].track_title, "url": url})

    return songList

@app.route('/uploadAudio', methods=['POST'])
def uploadAudio():
    bytesOfSong = request.get_data()
    song = AudioSegment.from_file(io.BytesIO(bytesOfSong), 'm4a')
    song.export('tmpAudioRecording.mp3', format="mp3")

    prediction = neural_network.predict('tmpAudioRecording.mp3')
    prediction_2d = neural_network_2d.predict('tmpAudioRecording.mp3')
    prediction_2d_transfer_learning = neural_network_2d_transfer_learning.predict('tmpAudioRecording.mp3')
    simSongs = getSongsByGenreAndSong()
    print("1", prediction)
    print("2", prediction_2d)
    print("3", prediction_2d_transfer_learning)
    return [Genres[prediction], Genres[prediction_2d], Genres[prediction_2d_transfer_learning], simSongs]



@app.route('/', methods=['GET'])
@app.route('/index/', methods=['GET'])
def home():
    return 'Hello World!'

def main():

    app.run(host="0.0.0.0", port=12345)

if __name__=="__main__":
    main()
