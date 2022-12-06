import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import React, { useState } from "react";
import { COLORS } from "../../colors";

export default function OptionButtons({ navigation, ...props }) {
  return (
    <View style={styles.container}>
      <TouchableOpacity
        activeOpacity={1}
        style={[
          styles.button,
          props.mode == 0 && { borderColor: COLORS.green, borderWidth: 1 },
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
          props.mode == 1 && { borderColor: COLORS.green, borderWidth: 1 },
        ]}
        onPress={() => {
          if (props.mode !== 1) {
            props.setMode(1);
          }
        }}
      >
        <Text style={styles.text}>Genres</Text>
      </TouchableOpacity>

      {/*<TouchableOpacity
        activeOpacity={1}
        style={[
          styles.button,
          props.mode == 2 && { borderColor: COLORS.green, borderWidth: 1 },
        ]}
        onPress={() => {
          if (props.mode !== 2) {
            props.setMode(2);
          }
        }}
      >
        <Text style={styles.}text}>Mood</Text>
      </TouchableOpacity>*/}
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
