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
import { uri } from "../../ip";

export default function FindGenres({ navigation, ...props }) {
  const [backendAnswer, setBackendAnswer] = useState(false);
  const [loading, setLoading] = useState(false);

  async function getGenreFromPython() {
    try {
      console.log(props.recording.file);
      const response = await FileSystem.uploadAsync(
        uri + "/uploadAudio",
        props.recording.file
      );
      // console.log(JSON.parse(response.body));
      new Promise((resolve) => setTimeout(resolve, 2000)).then(() => {
        setBackendAnswer(JSON.parse(response.body));
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
          <View style={styles.multipleView}>
            <TouchableOpacity
              style={styles.answerView}
              onPress={() => {
                navigation.navigate("Songs", {
                  genre: backendAnswer[0],
                });
              }}
            >
              <Text style={styles.answerText}>{backendAnswer[0]}</Text>
              <Text style={styles.predictionAlgo}>GenreNeuralNetwork</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.answerView}
              onPress={() => {
                navigation.navigate("Songs", {
                  genre: backendAnswer[1],
                });
              }}
            >
              <Text style={styles.answerText}>{backendAnswer[1]}</Text>
              <Text style={styles.predictionAlgo}>GenreNeuralNetwork2D</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.answerView}
              onPress={() => {
                navigation.navigate("Songs", {
                  genre: backendAnswer[2],
                });
              }}
            >
              <Text style={styles.answerText}>{backendAnswer[2]}</Text>
              <Text style={styles.predictionAlgo}>
                GenreNeuralNetwork2DTransferLearned
              </Text>
            </TouchableOpacity>
          </View>
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
    marginBottom: 20,
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
    alignItems: "center",
    alignSelf: "center",
    paddingTop: 10,
    paddingBottom: 10,
    borderRadius: 5,
  },

  answerText: {
    fontSize: 30,
    fontWeight: "bold",
    color: COLORS.text,
  },

  predictionAlgo: {
    fontSize: 15,
    color: COLORS.text,
    alignSelf: "flex-start",
  },

  multipleView: {
    //flexDirection: "row",
  },
});
