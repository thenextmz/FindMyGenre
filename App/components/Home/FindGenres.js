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
import { Divider } from "@rneui/base";

export default function FindGenres({ navigation, ...props }) {
  const [backendAnswer, setBackendAnswer] = useState(false);
  const [loading, setLoading] = useState(false);
  const [nnMethod, setnnMethod] = useState(0);

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
            {/*<AntDesign name={"arrowdown"} size={23} color={"white"} />*/}
            <Divider
              color={"white"}
              width={1.2}
              style={{ marginVertical: 10, marginHorizontal: 10 }}
            />
          </View>
          <View style={styles.nnModeView}>
            <TouchableOpacity
              style={[
                styles.modeButton,
                nnMethod == 0 && { borderWidth: 1, borderColor: COLORS.theme },
              ]}
              activeOpacity={1}
              onPress={() => {
                setnnMethod(0);
              }}
            >
              <Text style={styles.predictionAlgo}>NN1</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.modeButton,
                nnMethod == 1 && { borderWidth: 1, borderColor: COLORS.theme },
              ]}
              activeOpacity={1}
              onPress={() => {
                setnnMethod(1);
              }}
            >
              <Text style={styles.predictionAlgo}>NN2</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.modeButton,
                nnMethod == 2 && { borderWidth: 1, borderColor: COLORS.theme },
              ]}
              activeOpacity={1}
              onPress={() => {
                setnnMethod(2);
              }}
            >
              <Text style={styles.predictionAlgo}>NN3</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.multipleView}>
            {nnMethod == 0 && (
              <TouchableOpacity
                style={styles.answerView}
                onPress={() => {
                  navigation.navigate("Songs", {
                    genre: backendAnswer[0],
                  });
                }}
              >
                <Text style={styles.answerText}>{backendAnswer[0]}</Text>
                <Text style={styles.predictionAlgo}>1D-NeuralNetwork</Text>
              </TouchableOpacity>
            )}

            {nnMethod == 1 && (
              <TouchableOpacity
                style={styles.answerView}
                onPress={() => {
                  navigation.navigate("Songs", {
                    genre: backendAnswer[1],
                  });
                }}
              >
                <Text style={styles.answerText}>{backendAnswer[1]}</Text>
                <Text style={styles.predictionAlgo}>2D-NeuralNetwork</Text>
              </TouchableOpacity>
            )}

            {nnMethod == 2 && (
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
                  2D-NeuralNetwork-Pretrained
                </Text>
              </TouchableOpacity>
            )}
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
    paddingTop: 10,
    //paddingBottom: 10,
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

  nnModeView: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 10,
    marginHorizontal: 20,
    marginBottom: 20,
  },

  modeButton: {
    backgroundColor: COLORS.dark2,
    padding: 20,
    borderRadius: 5,
  },
});
