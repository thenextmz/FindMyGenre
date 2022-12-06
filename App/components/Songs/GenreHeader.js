import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import React from "react";
import { COLORS } from "../../colors";
import MaterialIcons from "react-native-vector-icons/MaterialIcons";

export default function GenreHeader({ navigation, ...props }) {
  return (
    <View style={styles.header}>
      <TouchableOpacity
        style={styles.button}
        activeOpacity={1}
        onPress={() => {
          navigation.goBack();
        }}
      >
        <MaterialIcons name={"arrow-back-ios"} size={25} color={COLORS.text} />
      </TouchableOpacity>
      <Text style={styles.text}>{props.genre}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  header: {
    //backgroundColor: "green",
    flexDirection: "row",
    marginHorizontal: 15,
    marginVertical: 20,
    alignItems: "center",
  },

  text: {
    color: COLORS.text,
    fontSize: 30,
    fontWeight: "bold",
  },
  button: {
    marginRight: 10,
  },
});
