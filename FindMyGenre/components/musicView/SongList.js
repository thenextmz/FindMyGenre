import React, { useEffect, useState } from "react";
import { songs } from "../../data/data";
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { ScrollView } from "react-native-gesture-handler";
import { Divider } from "@rneui/base";
import { COLORS } from "../../theme/colors";

export default function SongList(props) {
  const [songList, setSongList] = useState(null);

  const getSongs = () => {
    setSongList(null);
    let matching_songs = [];
    for (const [key, value] of Object.entries(songs)) {
      let index = value.findIndex((genre) => props.genre === genre);
      if (index !== -1) {
        matching_songs.push(key);
      }
    }
    setSongList(matching_songs);
    return matching_songs;
  };

  useEffect(() => {
    getSongs();
  }, [props.genre]);

  return (
    <ScrollView style={styles.container}>
      <View style={styles.songView}>
        {!songList && (
          <View style={{ flex: 1 }}>
            <ActivityIndicator size={40} color="black" />
          </View>
        )}
        {songList &&
          songList.map((song, index) => {
            return (
              <View key={index}>
                <View style={styles.song}>
                  <Text style={{ color: "black", fontSize: 20 }}>{song}</Text>
                  <View
                    style={{
                      flexDirection: "row",
                      padding: 10,
                    }}
                  >
                    {songs[song].map((genre, index) => {
                      return (
                        <TouchableOpacity
                          key={index}
                          onPress={() => {
                            props.navigation.navigate("MusicView", {
                              genre: genre,
                            });
                          }}
                        >
                          <Text style={{ marginRight: 10 }}>{genre}</Text>
                        </TouchableOpacity>
                      );
                    })}
                  </View>
                </View>
                <Divider
                  color={COLORS.theme}
                  width={1.2}
                  style={{
                    marginTop: 10,
                    marginBottom: 10,
                    marginHorizontal: 0,
                  }}
                />
              </View>
            );
          })}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  song: {
    marginTop: 10,
  },
  songView: {
    marginHorizontal: 20,
  },
});
