import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import React, { useState } from "react";
import FontAwesome5 from "react-native-vector-icons/FontAwesome5";
import { COLORS } from "../../theme/colors";

export default function ListenButton() {
  const [listening, setListening] = useState(false);
  return (
    <View style={styles.container}>
      <TouchableOpacity
        style={styles.circle}
        onPress={() => {
          setListening(!listening);
        }}
      >
        <FontAwesome5 name={"microphone"} size={35} color={"white"} />
      </TouchableOpacity>
      {listening && <Text style={styles.listening}>Now Listening...</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { alignItems: "center" },
  circle: {
    height: 100,
    width: 100,
    borderRadius: 1000,
    backgroundColor: COLORS.theme,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 30,

    shadowColor: "#000",
    shadowOffset: { width: 1, height: 1 },
    shadowOpacity: 0.4,
    shadowRadius: 5,
    elevation: 5,
  },

  listening: {
    marginTop: 10,
  },
});
