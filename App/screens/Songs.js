import { StyleSheet, Text, View } from "react-native";
import React, { useState } from "react";
import { COLORS } from "../colors";
import { SafeAreaView } from "react-native-safe-area-context";
import GenreHeader from "../components/Songs/GenreHeader";
import SongList from "../components/Songs/SongList";
import { songs } from "../mockupData/songs";

export default function Songs({ navigation, route }) {
  const { genre } = route.params;
  const [song, setSong] = useState(null);

  const SongInfo = () => {
    return (
      <View style={styles.container}>
        <View style={{ justifyContent: "center", alignItems: "center" }}>
          <Text style={styles.name}>{songs[song].name}</Text>
        </View>

        <View style={styles.container2}>
          <Text style={styles.artist}>Artist(s): </Text>
          {songs[song].artists.map((artist, index) => {
            return (
              <Text key={index} style={styles.artist}>
                {index !== songs[song].artists.length - 1
                  ? artist + ", "
                  : artist}
              </Text>
            );
          })}
        </View>

        <View style={styles.container2}>
          <Text style={styles.genre}>Genre(s): </Text>
          {songs[song].genres.map((genres, index) => {
            return (
              <Text key={index} style={styles.genre}>
                {index !== songs[song].genres.length - 1
                  ? genres + ", "
                  : genres}
              </Text>
            );
          })}
        </View>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <GenreHeader genre={genre} navigation={navigation} />
      <SongList
        genre={genre}
        navigation={navigation}
        setSong={setSong}
        song={song}
      />

      {song && <SongInfo song={song} />}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: COLORS.dark,
  },

  container: {
    flex: 0.5,
    backgroundColor: COLORS.dark2,
    marginHorizontal: 10,
    marginVertical: 10,
    borderRadius: 5,
    padding: 10,
  },

  container2: {
    flexDirection: "row",
    alignItems: "center",
  },

  name: {
    color: "white",
    fontSize: 25,
    fontWeight: "bold",
    marginRight: 10,
    marginBottom: 10,
  },

  artist: {
    color: "white",
    fontSize: 15,
    marginBottom: 10,
  },

  genre: {
    color: "white",
    fontSize: 15,
    marginBottom: 10,
  },
});
