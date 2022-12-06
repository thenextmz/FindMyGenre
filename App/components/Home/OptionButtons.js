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
          props.active && { borderColor: COLORS.green, borderWidth: 1 },
        ]}
        onPress={() => {
          if (!props.active) {
            props.setActive(!props.active);
          }
        }}
      >
        <Text style={styles.text}>Listen</Text>
      </TouchableOpacity>

      <TouchableOpacity
        activeOpacity={1}
        style={[
          styles.button,
          !props.active && { borderColor: COLORS.green, borderWidth: 1 },
        ]}
        onPress={() => {
          if (props.active) {
            props.setActive(!props.active);
          }
        }}
      >
        <Text style={styles.text}>Browse</Text>
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
