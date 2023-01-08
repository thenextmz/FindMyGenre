from pandas import *


def getData():
    data = read_csv("../../data/fma_metadata/raw_tracks.csv")
    for artist in data["artist_name"]:
        print(artist)
    return 0


if __name__ == '__main__':
    getData()

