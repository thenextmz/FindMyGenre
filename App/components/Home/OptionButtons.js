import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import React, { useState } from "react";
import { COLORS } from "../../colors";
import { Audio } from "expo-av";

export default function OptionButtons({ navigation, ...props }) {
  async function stop() {
    await props.recording.stopAndUnloadAsync();
  }

  return (
    <View style={styles.container}>
      <TouchableOpacity
        activeOpacity={1}
        style={[
          styles.button,
          props.mode == 0 && { borderColor: COLORS.theme, borderWidth: 1 },
        ]}
        onPress={() => {
          if (props.mode !== 0) {
            props.setMode(0);
          }
        }}
      >
        <Text style={styles.text}>Listen</Text>
      </TouchableOpacity>

      <TouchableOpacity
        activeOpacity={1}
        style={[
          styles.button,
          props.mode == 1 && { borderColor: COLORS.theme, borderWidth: 1 },
        ]}
        onPress={() => {
          if (props.mode !== 1) {
            if (props.currentlyRecording) {
              props.setCurrentlyRecording(false);
              props.setRecording(false);
              stop();
            }
            props.setMode(1);
          }
        }}
      >
        <Text style={styles.text}>Genres</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    marginLeft: 15,
  },

  button: {
    backgroundColor: COLORS.dark2,
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
    marginRight: 20,
  },

  text: {
    color: COLORS.text,
    fontWeight: "400",
  },
});
