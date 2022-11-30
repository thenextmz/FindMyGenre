import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import React, { useState } from "react";
import { COLORS } from "../../theme/colors";

export default function ModeButtons(props) {
  return (
    <View style={styles.container}>
      <TouchableOpacity
        activeOpacity={1}
        style={[styles.button, props.mode && { backgroundColor: COLORS.theme }]}
        onPress={() => {
          props.setMode(true);
        }}
      >
        <Text style={[styles.buttonText]}>Find</Text>
      </TouchableOpacity>

      <TouchableOpacity
        activeOpacity={1}
        style={[
          styles.button,
          !props.mode && { backgroundColor: COLORS.theme },
        ]}
        onPress={() => {
          props.setMode(false);
        }}
      >
        <Text style={[styles.buttonText]}>Browse</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
  },

  button: {
    padding: 10,
    paddingHorizontal: 20,
    borderRadius: 10,
    marginTop: 10,
    backgroundColor: COLORS.theme_light,
    alignItems: "center",
    width: 100,
    marginHorizontal: 20,

    shadowColor: "#000",
    shadowOffset: { width: 1, height: 1 },
    shadowOpacity: 0.4,
    shadowRadius: 3,
    elevation: 5,
  },

  buttonText: {
    color: "white",
  },
});
