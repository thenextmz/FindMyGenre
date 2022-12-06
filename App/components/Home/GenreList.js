import {
  FlatList,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import React from "react";
import { COLORS } from "../../colors";

export default function GenreList({ navigation }) {
  const renderItem = ({ item, index }) => {
    return (
      <TouchableOpacity
        onPress={() => {
          navigation.navigate("Songs", {
            genre: item,
          });
        }}
        style={[
          styles.genreButton,
          index % 2 === 0 ? { marginRight: 2.5 } : { marginLeft: 2.5 },
        ]}
      >
        <Text style={styles.text}>{item}</Text>
      </TouchableOpacity>
    );
  };

  let dummyArray = ["Pop", "Rock", "Metal", "Indie", "Gospel", "..."];
  return (
    <View style={styles.container}>
      <FlatList
        data={dummyArray}
        renderItem={renderItem}
        horizontal={false}
        numColumns={2}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginHorizontal: 15,
    marginTop: 20,
    flexDirection: "row",
    justifyContent: "space-between",
  },

  genreButton: {
    backgroundColor: COLORS.dark2,
    flex: 0.5,
    alignItems: "center",
    borderRadius: 5,
    height: 40,
    justifyContent: "center",
    marginBottom: 5,
    marginRight: 0,
  },

  text: {
    //alignSelf: "center",
    color: COLORS.text,
  },
});
