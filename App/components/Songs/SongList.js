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
import { uri } from "../../ip";

export default function SongList({ navigation, ...props }) {
  const [songArr, setSongArr] = useState(false);
  const [loading, setLoading] = useState(true);
  const [endReached, setEndReached] = useState(true);

  useEffect(() => {
    if (!songArr) {
      getSongs();
    }
  });

  const getSongs = async () => {
    fetch(uri + "/getSongsByGenre?genre=" + props.genre)
      .then((response) => response.json())
      .then((json) => {
        // randomly sorting array
        json.sort(() => Math.random() - 0.5);
        // displaying first 100 random songs
        setSongArr(json.slice(0, 100));
        setLoading(false);
      })
      .catch((error) => {
        console.error(error);
      });
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
          item == props.song && { borderColor: COLORS.theme, borderWidth: 1 },
        ]}
      >
        <Text numberOfLines={1} style={styles.name}>
          {item.song}
        </Text>
      </TouchableOpacity>
    );
  };

  return (
    <View style={styles.container}>
      {!loading && songArr && (
        <>
          <FlatList
            data={songArr}
            showsVerticalScrollIndicator={false}
            renderItem={renderItem}
            horizontal={false}
            numColumns={2}
            //initialNumToRender={20}
            //windowSize={20}
            onEndReached={() => {
              //setEndReached(true);
            }}
          />
        </>
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
    flexDirection: "column",
  },

  songButton: {
    backgroundColor: COLORS.dark2,
    flex: 0.5,
    alignItems: "center",
    borderRadius: 5,
    height: 40,
    justifyContent: "center",
    marginBottom: 5,
    //paddingHorizontal: 10,
    //marginRight: 0,
    marginHorizontal: 5,
  },

  name: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: "bold",
    paddingHorizontal: 10,
  },

  artist: {
    color: COLORS.text,
  },
});
