import {
  StyleSheet,
  Text,
  View,
  ActivityIndicator,
  TouchableOpacity,
  FlatList,
  Linking,
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
      //console.log(props.recording.file);
      const response = await FileSystem.uploadAsync(
        uri + "/uploadAudio",
        props.recording.file
      );
      // console.log(JSON.parse(response.body));
      setBackendAnswer(JSON.parse(response.body));
      setLoading(false);
    } catch (err) {
      console.error(err);
    } finally {
      //console.log(backendAnswer);
    }
  }

  const renderItem = ({ item, index }) => {
    return (
      <TouchableOpacity
        style={[
          styles.songButton,
          index % 2 === 0 ? { marginRight: 2.5 } : { marginLeft: 2.5 },
          item == props.song && { borderColor: COLORS.theme, borderWidth: 1 },
        ]}
        onPress={() => {
          Linking.openURL(item.url);
        }}
        activeOpacity={1}
      >
        <Text numberOfLines={1} style={styles.name}>
          {item.artist}
        </Text>
        <Text numberOfLines={1} style={styles.name}>
          {item.song}
        </Text>
      </TouchableOpacity>
    );
  };

  return (
    <View style={styles.container}>
      {!loading && (
        <View style={styles.multiButton}>
          <TouchableOpacity
            style={styles.startButton}
            activeOpacity={1}
            onPress={() => {
              setLoading(true);
              setBackendAnswer(false);
              getGenreFromPython();
            }}
          >
            <Entypo name={"music"} size={30} color={"white"} />
          </TouchableOpacity>

          {backendAnswer && !loading && (
            <>
              <TouchableOpacity
                style={[
                  styles.modeButton,
                  nnMethod == 0 && {
                    borderWidth: 1,
                    borderColor: COLORS.theme,
                  },
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
                  nnMethod == 1 && {
                    borderWidth: 1,
                    borderColor: COLORS.theme,
                  },
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
                  nnMethod == 2 && {
                    borderWidth: 1,
                    borderColor: COLORS.theme,
                  },
                ]}
                activeOpacity={1}
                onPress={() => {
                  setnnMethod(2);
                }}
              >
                <Text style={styles.predictionAlgo}>NN3</Text>
              </TouchableOpacity>
            </>
          )}
        </View>
      )}

      {!backendAnswer && loading && (
        <View
          style={{
            justifyContent: "flex-start",
            flexDirection: "row",
            marginLeft: 20,
            marginTop: 15,
          }}
        >
          <ActivityIndicator size={23} color="white" />
        </View>
      )}

      {backendAnswer && !loading && (
        <View style={styles.test}>
          <View style={styles.arrowView}>
            {/*<AntDesign name={"arrowdown"} size={23} color={"white"} />*/}
            <Divider
              color={"white"}
              width={1.2}
              style={{ marginVertical: 10, marginHorizontal: 10 }}
            />
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
                activeOpacity={1}
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
                activeOpacity={1}
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
                activeOpacity={1}
              >
                <Text style={styles.answerText}>{backendAnswer[2]}</Text>
                <Text style={styles.predictionAlgo}>
                  2D-NeuralNetwork-Pretrained
                </Text>
              </TouchableOpacity>
            )}
            <Divider
              color={"white"}
              width={1.2}
              style={{ marginBottom: 20, marginHorizontal: 10 }}
            />
            <FlatList
              data={backendAnswer[3]}
              showsVerticalScrollIndicator={false}
              renderItem={renderItem}
              horizontal={false}
              numColumns={2}
              style={styles.flatlist}
              //initialNumToRender={20}
              //windowSize={20}
              onEndReached={() => {
                //setEndReached(true);
              }}
            />
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginHorizontal: 10,
    marginVertical: 20,
    flex: 1,
  },

  startButton: {
    alignSelf: "center",
    backgroundColor: COLORS.dark2,
    alignItems: "center",
    justifyContent: "center",
    height: 50,
    width: 50,
    borderRadius: 5,
  },

  answerView: {
    marginBottom: 10,
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
    borderRadius: 5,
  },

  answerText: {
    fontSize: 20,
    fontWeight: "bold",
    color: COLORS.text,
  },

  predictionAlgo: {
    fontSize: 15,
    color: COLORS.text,
    alignSelf: "center",
  },

  multipleView: {
    flex: 1,
  },

  nnModeView: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 10,
    marginHorizontal: 20,
  },

  modeButton: {
    backgroundColor: COLORS.dark2,
    width: 50,
    height: 50,
    borderRadius: 5,
    alignSelf: "center",
    justifyContent: "center",
  },

  multiButton: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginHorizontal: 10,
  },

  songButton: {
    backgroundColor: COLORS.dark2,
    flex: 0.5,
    alignItems: "center",
    borderRadius: 5,
    height: 40,
    justifyContent: "center",
    marginBottom: 5,
    marginHorizontal: 5,
  },

  name: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: "bold",
    paddingHorizontal: 10,
  },

  test: {
    flex: 1,
  },
});
