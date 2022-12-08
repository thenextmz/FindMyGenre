import { StyleSheet, Text, View } from "react-native";
import React from "react";
import { COLORS } from "../../colors";

export default function Header() {
  return (
    <View style={styles.header}>
      <Text style={styles.text}>FindMyGenre</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  header: {
    //backgroundColor: "green",
    marginHorizontal: 15,
    marginVertical: 20,
  },

  text: {
    color: COLORS.text,
    fontSize: 30,
    fontWeight: "bold",
  },
});
