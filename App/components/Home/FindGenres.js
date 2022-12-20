import {
  StyleSheet,
  Text,
  View,
  ActivityIndicator,
  TouchableOpacity,
} from "react-native";
import React, { useState } from "react";
import { COLORS } from "../../colors";
import Entypo from "react-native-vector-icons/Entypo";
import AntDesign from "react-native-vector-icons/AntDesign";
import * as FileSystem from "expo-file-system";

export default function FindGenres({ navigation, ...props }) {
  const [backendAnswer, setBackendAnswer] = useState(false);
  const [loading, setLoading] = useState(false);

  async function getGenreFromPython() {
    try {
      console.log(props.recording.file);
      const response = await FileSystem.uploadAsync(
        "http://10.0.0.26:12345/uploadAudio",
        props.recording.file
      );
      console.log(response);
      new Promise((resolve) => setTimeout(resolve, 2000)).then(() => {
        setBackendAnswer(response.body);
        setLoading(false);
      });
    } catch (err) {
      console.error(err);
    }
  }

  return (
    <View style={styles.container}>
      {!loading && (
        <TouchableOpacity
          style={styles.startButton}
          activeOpacity={1}
          onPress={() => {
            setLoading(true);
            setBackendAnswer(false);
            getGenreFromPython();
          }}
        >
          <Entypo name={"music"} size={23} color={"white"} />
        </TouchableOpacity>
      )}

      {!backendAnswer && loading && (
        <ActivityIndicator size={23} color="white" />
      )}

      {backendAnswer && !loading && (
        <>
          <View style={styles.arrowView}>
            <AntDesign name={"arrowdown"} size={23} color={"white"} />
          </View>
          <TouchableOpacity
            style={styles.answerView}
            onPress={() => {
              navigation.navigate("Songs", {
                genre: backendAnswer,
              });
            }}
          >
            <Text style={styles.answerText}>{backendAnswer}</Text>
          </TouchableOpacity>
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginHorizontal: 10,
    marginVertical: 20,
  },

  startButton: {
    alignSelf: "center",
    backgroundColor: COLORS.dark2,
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 5,
  },

  answerView: {
    marginTop: 0,
    alignItems: "center",
    alignSelf: "center",
    backgroundColor: COLORS.dark2,
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 5,
    borderWidth: 1,
    borderColor: COLORS.theme,
  },

  arrowView: {
    marginTop: 0,
    alignItems: "center",
    alignSelf: "center",

    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 5,
  },

  answerText: {
    fontSize: 30,
    fontWeight: "bold",
    color: COLORS.text,
  },
});
