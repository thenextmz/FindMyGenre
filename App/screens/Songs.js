import {
  Linking,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import React, { useState } from "react";
import { COLORS } from "../colors";
import { SafeAreaView } from "react-native-safe-area-context";
import GenreHeader from "../components/Songs/GenreHeader";
import SongList from "../components/Songs/SongList";
import AntDesign from "react-native-vector-icons/AntDesign";

export default function Songs({ navigation, route }) {
  const { genre } = route.params;
  const [song, setSong] = useState(null);

  const SongInfo = () => {
    return (
      <View style={styles.container}>
        <View style={{ justifyContent: "center", alignItems: "center" }}>
          <Text style={styles.name}>{song.song}</Text>
        </View>

        <View style={styles.container2}>
          <Text style={styles.artist}>Artist: {song.artist}</Text>
        </View>

        {song.url !== -1 && (
          <TouchableOpacity
            style={styles.container2}
            onPress={() => {
              Linking.openURL(song.url);
            }}
            activeOpacity={1}
          >
            <View style={{ marginRight: 10 }}>
              <AntDesign name={"link"} size={20} color={"white"} />
            </View>
            <Text style={styles.artist}>Song Url</Text>
          </TouchableOpacity>
        )}
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
