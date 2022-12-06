import {
  ActivityIndicator,
  FlatList,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import React, { useEffect, useState } from "react";
import { COLORS } from "../../colors";
import { songs } from "../../mockupData/songs";

export default function SongList({ navigation, ...props }) {
  const [songArr, setSongArr] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!songArr) {
      getSongs();
    }
  });

  const getSongs = () => {
    setLoading(true);
    let tmp = [];
    for (var song in songs) {
      let index = songs[song].genres.findIndex((genre) => genre == props.genre);
      if (index !== -1) {
        tmp.push(song);
      }
    }
    setSongArr(tmp);
    setLoading(false);
  };

  const renderItem = ({ item, index }) => {
    return (
      <TouchableOpacity
        onPress={() => {
          if (item == props.song) {
            props.setSong(null);
          } else {
            props.setSong(item);
          }
        }}
        style={[
          styles.songButton,
          index % 2 === 0 ? { marginRight: 2.5 } : { marginLeft: 2.5 },
          item == props.song && { borderColor: COLORS.green, borderWidth: 1 },
        ]}
      >
        <Text style={styles.name}>{songs[item].name}</Text>
      </TouchableOpacity>
    );
  };

  return (
    <View style={styles.container}>
      {!loading && songArr && (
        <FlatList
          data={songArr}
          renderItem={renderItem}
          horizontal={false}
          numColumns={2}
        />
      )}

      {loading && (
        <View style={{ flex: 1 }}>
          <ActivityIndicator size={23} color="white" />
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginHorizontal: 15,
    marginTop: 20,
    flexDirection: "row",
    justifyContent: "space-between",
    flex: 1,
  },

  songButton: {
    backgroundColor: COLORS.dark2,
    flex: 0.5,
    alignItems: "center",
    borderRadius: 5,
    height: 40,
    justifyContent: "center",
    marginBottom: 5,
    marginRight: 0,
  },

  name: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: "bold",
  },

  artist: {
    color: COLORS.text,
  },
});
