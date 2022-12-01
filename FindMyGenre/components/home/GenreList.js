import {
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import React from "react";
import { genres, songs } from "../../data/data";
import { COLORS } from "../../theme/colors";

export default function GenreList({ navigation, ...props }) {
  return (
    <ScrollView style={styles.container}>
      <View style={styles.categoryView}>
        {genres.map((genre, index) => {
          return (
            <TouchableOpacity
              key={index}
              style={styles.button}
              onPress={() => {
                navigation.navigate("MusicView", {
                  genre: genre,
                });
              }}
            >
              <Text style={{ color: "white" }} key={index}>
                {genre}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    marginTop: 20,
    //marginHorizontal: 50,
  },

  button: {
    backgroundColor: COLORS.theme,
    paddingHorizontal: 15,
    paddingVertical: 10,
    borderRadius: 20,
    marginHorizontal: 10,
    marginBottom: 10,

    shadowColor: "#000",
    shadowOffset: { width: 1, height: 1 },
    shadowOpacity: 0.4,
    shadowRadius: 3,
    elevation: 5,
  },

  categoryView: {
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "center",
    marginHorizontal: 40,
    justifyContent: "space-between",
  },
});
