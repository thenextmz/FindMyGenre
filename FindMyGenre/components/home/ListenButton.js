import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import React, { useState } from "react";
import FontAwesome5 from "react-native-vector-icons/FontAwesome5";
import { COLORS } from "../../theme/colors";
import { Audio } from "expo-av";

export default function ListenButton(props) {
  const [listening, setListening] = useState(false);
  const [recording, setRecording] = React.useState();
  const [errorMessage, setErrorMessage] = React.useState("");

  async function record() {
    const permission = await Audio.requestPermissionsAsync();
    if (permission.status === "granted") {
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const { recording } = await Audio.Recording.createAsync(
        Audio.RECORDING_OPTIONS_PRESET_HIGH_QUALITY
      );
      setRecording(recording);
    } else {
      setErrorMessage("Please grant permission to app to access microphone");
    }
  }

  async function stop() {
    await recording.stopAndUnloadAsync();
    await Audio.setAudioModeAsync({
      allowsRecordingIOS: false,
      playsInSilentModeIOS: true,
    });

    const { sound } = await recording.createNewLoadedSoundAsync();
    setRecording({
      sound: sound,
      file: recording.getURI(),
    });
  }

  return (
    <View
      style={[
        styles.container,
        recording &&
          !listening && {
            flexDirection: "row",
            justifyContent: "space-between",
            marginHorizontal: 60,
          },
      ]}
    >
      {!listening && (
        <TouchableOpacity
          style={styles.circle}
          onPress={() => {
            setListening(!listening);
            record();
          }}
        >
          <FontAwesome5 name={"microphone"} size={35} color={"white"} />
        </TouchableOpacity>
      )}

      {listening && (
        <TouchableOpacity
          style={styles.circle}
          onPress={() => {
            setListening(!listening);
            stop();
          }}
        >
          <FontAwesome5 name={"stop"} size={35} color={"white"} />
        </TouchableOpacity>
      )}

      {listening && <Text style={styles.listening}>Now Listening...</Text>}
      {errorMessage && <Text style={styles.listening}>{message}</Text>}

      {recording && !listening && (
        <TouchableOpacity
          style={styles.playButton}
          onPress={() => {
            recording.sound.replayAsync();
          }}
        >
          <FontAwesome5 name={"play"} size={25} color={"white"} />
        </TouchableOpacity>
      )}
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

  playButton: {
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
});
