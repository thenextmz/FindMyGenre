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


def getSongsByGenreAndSong(path = 'tmpAudioRecording.mp3'):
    mfcc_song = MP3toSoundStats(path)
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
    return [Genres[prediction], Genres[prediction_2d], Genres[prediction_2d_transfer_learning], simSongs]

def classifyDemoSongs():
    print("##########################################")
    print("#         Classify Demo Audio            #")
    print("##########################################")
    print("Song 1")
    print('NN1:', Genres[neural_network.predict(os.getcwd() + '/data/test_songs/Rock/PopRock.mp3')], "(Orig. Lable: POP/ROCK)")
    print('NN2:', Genres[neural_network_2d.predict(os.getcwd() + '/data/test_songs/Rock/PopRock.mp3')], "(Orig. Lable: POP/ROCK)")
    print('NN3:', Genres[neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/test_songs/Rock/PopRock.mp3')], "(Orig. Lable: POP/ROCK)")
    print("------------------------------------------------------")
    print("Song 2")
    print('NN1:', Genres[neural_network.predict(os.getcwd() + '/data/test_songs/Rock/PopRock1.mp3')], "(Orig. Lable: POP/ROCK)")
    print('NN2:', Genres[neural_network_2d.predict(os.getcwd() + '/data/test_songs/Rock/PopRock1.mp3')], "(Orig. Lable: POP/ROCK)")
    print('NN3:', Genres[neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/test_songs/Rock/PopRock1.mp3')], "(Orig. Lable: POP/ROCK)")
    print("------------------------------------------------------")
    print("Song 3")
    print('NN1:', Genres[neural_network.predict(os.getcwd() + '/data/test_songs/Rock/PopRock2.mp3')], "(Orig. Lable: POP/ROCK)")
    print('NN2:', Genres[neural_network_2d.predict(os.getcwd() + '/data/test_songs/Rock/PopRock2.mp3')], "(Orig. Lable: POP/ROCK)")
    print('NN3:', Genres[neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/test_songs/Rock/PopRock2.mp3')], "(Orig. Lable: POP/ROCK)")
    print("------------------------------------------------------")
    print("Song 4")
    print('NN1:', Genres[neural_network.predict(os.getcwd() + '/data/test_songs/Rock/RockPunkIndieElectro.mp3')], "(Orig. Lable: ROCK/PUNK/INDIE/ELECTRO)")
    print('NN2:', Genres[neural_network_2d.predict(os.getcwd() + '/data/test_songs/Rock/RockPunkIndieElectro.mp3')], "(Orig. Lable: ROCK/PUNK/INDIE/ELECTRO)")
    print('NN3:', Genres[neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/test_songs/Rock/RockPunkIndieElectro.mp3')], "(Orig. Lable: ROCK/PUNK/INDIE/ELECTRO)")
    print("------------------------------------------------------")
    print("Song 5")
    print('NN1:', Genres[neural_network.predict(os.getcwd() + '/data/test_songs/Rock/RockPunkIndieElectro1.mp3')], "(Orig. Lable: ROCK/PUNK/INDIE/ELECTRO)")
    print('NN2:', Genres[neural_network_2d.predict(os.getcwd() + '/data/test_songs/Rock/RockPunkIndieElectro1.mp3')], "(Orig. Lable: ROCK/PUNK/INDIE/ELECTRO)")
    print('NN3:', Genres[neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/test_songs/Rock/RockPunkIndieElectro1.mp3')], "(Orig. Lable: ROCK/PUNK/INDIE/ELECTRO)")
    print("------------------------------------------------------")
    print("Song 6")
    print('NN1:', Genres[neural_network.predict(os.getcwd() + '/data/test_songs/Folk/CountryFolk2.mp3')], "(Orig. Lable: OUNTRY/FOLK)")
    print('NN2:', Genres[neural_network_2d.predict(os.getcwd() + '/data/test_songs/Folk/CountryFolk2.mp3')], "(Orig. Lable: OUNTRY/FOLK)")
    print('NN3:', Genres[neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/test_songs/Folk/CountryFolk2.mp3')], "(Orig. Lable: OUNTRY/FOLK)")
    print("------------------------------------------------------")
    print("Song 7")
    print('NN1:', Genres[neural_network.predict(os.getcwd() + '/data/test_songs/Folk/CountryFolk4.mp3')], "(Orig. Lable: OUNTRY/FOLK)")
    print('NN2:', Genres[neural_network_2d.predict(os.getcwd() + '/data/test_songs/Folk/CountryFolk4.mp3')], "(Orig. Lable: OUNTRY/FOLK)")
    print('NN3:', Genres[neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/test_songs/Folk/CountryFolk4.mp3')], "(Orig. Lable: OUNTRY/FOLK)")
    print("------------------------------------------------------")
    print("Song 8")
    print('NN1:', Genres[neural_network.predict(os.getcwd() + '/data/test_songs/Folk/BluesCountryFolkIndie.mp3')], "(Orig. Lable: BLUES/COUNTRY/FOLK/INDIE)")
    print('NN2:', Genres[neural_network_2d.predict(os.getcwd() + '/data/test_songs/Folk/BluesCountryFolkIndie.mp3')], "(Orig. Lable: BLUES/COUNTRY/FOLK/INDIE)")
    print('NN3:', Genres[neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/test_songs/Folk/BluesCountryFolkIndie.mp3')], "(Orig. Lable: BLUES/COUNTRY/FOLK/INDIE)")
    print("------------------------------------------------------")
    print("Song 9")
    print('NN1:', Genres[neural_network.predict(os.getcwd() + '/data/test_songs/Historic/Historic.mp3')], "(Orig. Lable: HISTORIC)")
    print('NN2:', Genres[neural_network_2d.predict(os.getcwd() + '/data/test_songs/Historic/Historic.mp3')], "(Orig. Lable: HISTORIC)")
    print('NN3:', Genres[neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/test_songs/Historic/Historic.mp3')], "(Orig. Lable: HISTORIC)")
    print("------------------------------------------------------")
    print("Song 10")
    print('NN1:', Genres[neural_network.predict(os.getcwd() + '/data/test_songs/Historic/Historic1.mp3')], "(Orig. Lable: HISTORIC)")
    print('NN2:', Genres[neural_network_2d.predict(os.getcwd() + '/data/test_songs/Historic/Historic1.mp3')], "(Orig. Lable: HISTORIC)")
    print('NN3:', Genres[neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/test_songs/Historic/Historic1.mp3')], "(Orig. Lable: HISTORIC)")
    print("------------------------------------------------------")
    print("Song 11")
    print('NN1:', Genres[neural_network.predict(os.getcwd() + '/data/test_songs/Historic/Historic2.mp3')], "(Orig. Lable: HISTORIC)")
    print('NN2:', Genres[neural_network_2d.predict(os.getcwd() + '/data/test_songs/Historic/Historic2.mp3')], "(Orig. Lable: HISTORIC)")
    print('NN3:', Genres[neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/test_songs/Historic/Historic2.mp3')], "(Orig. Lable: HISTORIC)")
    print("------------------------------------------------------")
    print("Song 12")
    print('NN1:', Genres[neural_network.predict(os.getcwd() + '/data/test_songs/Historic/Historic3.mp3')], "(Orig. Lable: HISTORIC")
    print('NN2:', Genres[neural_network_2d.predict(os.getcwd() + '/data/test_songs/Historic/Historic3.mp3')], "(Orig. Lable: HISTORIC)")
    print('NN3:', Genres[neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/test_songs/Historic/Historic3.mp3')], "(Orig. Lable: HISTORIC)")
    print("------------------------------------------------------")
    print("Song 13")
    print('NN1:', Genres[neural_network.predict(os.getcwd() + '/data/test_songs/Historic/HistoricInstrumental.mp3')], "(Orig. Lable: HISTORIC/INSTRUMENTAL")
    print('NN2:', Genres[neural_network_2d.predict(os.getcwd() + '/data/test_songs/Historic/HistoricInstrumental.mp3')], "(Orig. Lable: HISTORIC/INSTRUMENTAL)")
    print('NN3:', Genres[neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/test_songs/Historic/HistoricInstrumental.mp3')], "(Orig. Lable: HISTORIC/INSTRUMENTAL)")
    print("------------------------------------------------------")
    print("Song 14")
    print('NN1:', Genres[neural_network.predict(os.getcwd() + '/data/test_songs/Jazz/JazzElectronic.mp3')], "(Orig. Lable: JAZZ/ELECTRONIC")
    print('NN2:', Genres[neural_network_2d.predict(os.getcwd() + '/data/test_songs/Jazz/JazzElectronic.mp3')], "(Orig. Lable:  JAZZ/ELECTRONIC)")
    print('NN3:', Genres[neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/test_songs/Jazz/JazzElectronic.mp3')], "(Orig. Lable:  JAZZ/ELECTRONIC)")
    print("------------------------------------------------------")
    print("Song 15")
    print('NN1:', Genres[neural_network.predict(os.getcwd() + '/data/test_songs/Jazz/JazzElectronicHiphot.mp3')], "(Orig. Lable:  JAZZ/ELECTRONIC/HIPHOP)")
    print('NN2:', Genres[neural_network_2d.predict(os.getcwd() + '/data/test_songs/Jazz/JazzElectronicHiphot.mp3')], "(Orig. Lable: JAZZ/ELECTRONIC/HIPHOP)")
    print('NN3:', Genres[neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/test_songs/Jazz/JazzElectronicHiphot.mp3')], "(Orig. Lable: JAZZ/ELECTRONIC/HIPHOP)")
    print("------------------------------------------------------")
    print("Song 16")
    print('NN1:', Genres[neural_network.predict(os.getcwd() + '/data/test_songs/HIPHOP/HIPHOP.mp3')], "(Orig. Lable:  HIPHOP)")
    print('NN2:', Genres[neural_network_2d.predict(os.getcwd() + '/data/test_songs/HIPHOP/HIPHOP.mp3')], "(Orig. Lable: HIPHOP)")
    print('NN3:', Genres[neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/test_songs/HIPHOP/HIPHOP.mp3')], "(Orig. Lable: HIPHOP)")
    print("------------------------------------------------------")
    print("Song 17")
    print('NN1:', Genres[neural_network.predict(os.getcwd() + '/data/test_songs/HIPHOP/HIPHOP2.mp3')], "(Orig. Lable:  HIPHOP)")
    print('NN2:', Genres[neural_network_2d.predict(os.getcwd() + '/data/test_songs/HIPHOP/HIPHOP2.mp3')], "(Orig. Lable: HIPHOP)")
    print('NN3:', Genres[neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/test_songs/HIPHOP/HIPHOP2.mp3')], "(Orig. Lable: HIPHOP)")
    print("------------------------------------------------------")
    print("Song 19")
    print('NN1:', Genres[neural_network.predict(os.getcwd() + '/data/test_songs/Rock/ACDC.mp3')], "(Orig. Lable:  ROCK)")
    print('NN2:', Genres[neural_network_2d.predict(os.getcwd() + '/data/test_songs/Rock/ACDC.mp3')], "(Orig. Lable: ROCK)")
    print('NN3:', Genres[neural_network_2d_transfer_learning.predict(os.getcwd() + '/data/test_songs/Rock/ACDC.mp3')], "(Orig. Lable: ROCK)")
    print("------------------------------------------------------")

def simDemo():
    print("##########################################")
    print("#              Sim Demo Audio            #")
    print("##########################################")
    print("Song 1")
    simSongs = getSongsByGenreAndSong(path = os.getcwd() + '/data/test_songs/Rock/PopRock.mp3')
    for i, song in enumerate(simSongs):
        print(str(i) + ":", str(song["url"]))
    print("------------------------------------------------------")
    print("Song 2")
    simSongs = getSongsByGenreAndSong(path = os.getcwd() + '/data/test_songs/Rock/PopRock1.mp3')
    for i, song in enumerate(simSongs):
        print(str(i) + ":", str(song["url"]))
    print("------------------------------------------------------")
    print("Song 3")
    simSongs = getSongsByGenreAndSong(path = os.getcwd() + '/data/test_songs/Rock/PopRock2.mp3')
    for i, song in enumerate(simSongs):
        print(str(i) + ":", str(song["url"]))
    print("------------------------------------------------------")
    print("Song 4")
    simSongs = getSongsByGenreAndSong(path = os.getcwd() + '/data/test_songs/Rock/RockPunkIndieElectro.mp3')
    for i, song in enumerate(simSongs):
        print(str(i) + ":", str(song["url"]))
    print("------------------------------------------------------")
    print("Song 5")
    simSongs = getSongsByGenreAndSong(path = os.getcwd() + '/data/test_songs/Rock/RockPunkIndieElectro1.mp3')
    for i, song in enumerate(simSongs):
        print(str(i) + ":", str(song["url"]))
    print("------------------------------------------------------")
    print("Song 6")
    simSongs = getSongsByGenreAndSong(path = os.getcwd() + '/data/test_songs/Folk/CountryFolk2.mp3')
    for i, song in enumerate(simSongs):
        print(str(i) + ":", str(song["url"]))
    print("------------------------------------------------------")
    print("Song 7")
    simSongs = getSongsByGenreAndSong(path = os.getcwd() + '/data/test_songs/Folk/CountryFolk4.mp3')
    for i, song in enumerate(simSongs):
        print(str(i) + ":", str(song["url"]))
    print("------------------------------------------------------")
    print("Song 8")
    simSongs = getSongsByGenreAndSong(path = os.getcwd() + '/data/test_songs/Folk/BluesCountryFolkIndie.mp3')
    for i, song in enumerate(simSongs):
        print(str(i) + ":", str(song["url"]))
    print("------------------------------------------------------")
    print("Song 9")
    simSongs = getSongsByGenreAndSong(path = os.getcwd() + '/data/test_songs/Historic/Historic.mp3')
    for i, song in enumerate(simSongs):
        print(str(i) + ":", str(song["url"]))
    print("------------------------------------------------------")
    print("Song 10")
    simSongs = getSongsByGenreAndSong(path = os.getcwd() + '/data/test_songs/Historic/Historic1.mp3')
    for i, song in enumerate(simSongs):
        print(str(i) + ":", str(song["url"]))
    print("------------------------------------------------------")
    print("Song 11")
    simSongs = getSongsByGenreAndSong(path = os.getcwd() + '/data/test_songs/Historic/Historic2.mp3')
    for i, song in enumerate(simSongs):
        print(str(i) + ":", str(song["url"]))
    print("------------------------------------------------------")
    print("Song 12")
    simSongs = getSongsByGenreAndSong(path = os.getcwd() + '/data/test_songs/Historic/Historic3.mp3')
    for i, song in enumerate(simSongs):
        print(str(i) + ":", str(song["url"]))
    print("------------------------------------------------------")
    print("Song 13")
    simSongs = getSongsByGenreAndSong(path = os.getcwd() + '/data/test_songs/Historic/HistoricInstrumental.mp3')
    for i, song in enumerate(simSongs):
        print(str(i) + ":", str(song["url"]))
    print("------------------------------------------------------")
    print("Song 14")
    simSongs = getSongsByGenreAndSong(path = os.getcwd() + '/data/test_songs/Jazz/JazzElectronic.mp3')
    for i, song in enumerate(simSongs):
        print(str(i) + ":", str(song["url"]))
    print("------------------------------------------------------")
    print("Song 15")
    simSongs = getSongsByGenreAndSong(path = os.getcwd() + '/data/test_songs/Jazz/JazzElectronicHiphot.mp3')
    for i, song in enumerate(simSongs):
        print(str(i) + ":", str(song["url"]))
    print("------------------------------------------------------")
    print("Song 16")
    simSongs = getSongsByGenreAndSong(path = os.getcwd() + '/data/test_songs/HIPHOP/HIPHOP.mp3')
    for i, song in enumerate(simSongs):
        print(str(i) + ":", str(song["url"]))
    print("------------------------------------------------------")
    print("Song 17")
    simSongs = getSongsByGenreAndSong(path = os.getcwd() + '/data/test_songs/HIPHOP/HIPHOP2.mp3')
    for i, song in enumerate(simSongs):
        print(str(i) + ":", str(song["url"]))
    print("------------------------------------------------------")
    print("Song 19")
    simSongs = getSongsByGenreAndSong(path = os.getcwd() + '/data/test_songs/Rock/ACDC.mp3')
    for i, song in enumerate(simSongs):
        print(str(i) + ":", str(song["url"]))
    print("------------------------------------------------------")
    

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "-t":
        print("Start Training...")
        neural_network.fit()
        neural_network_2d.fit()
        neural_network_2d_transfer_learning.fit()
        print("Training Done.")

    if len(sys.argv) > 1 and sys.argv[1] == "-d":
        classifyDemoSongs()

    if len(sys.argv) > 1 and sys.argv[1] == "-s":
        simDemo()    

   
    app.run(host="0.0.0.0", port=12345)


if __name__=="__main__":
    main()
