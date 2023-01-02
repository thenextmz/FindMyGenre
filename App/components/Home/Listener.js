import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import React, { useEffect, useState } from "react";
import { COLORS } from "../../colors";
import FontAwesome5 from "react-native-vector-icons/FontAwesome5";
import Entypo from "react-native-vector-icons/Entypo";
import AntDesign from "react-native-vector-icons/AntDesign";
import LoadingDots from "react-native-loading-dots";
import MaterialIcons from "react-native-vector-icons/MaterialIcons";
import { Audio } from "expo-av";

export default function Listener(props) {
  const [playingAudio, setPlayingAudio] = useState(false);

  async function record() {
    const permission = await Audio.requestPermissionsAsync();
    if (permission.status === "granted") {
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const { recording } = await Audio.Recording.createAsync(
          Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      props.setRecording(recording);
    } else {
      setErrorMessage("Allow microphone permissions!");
    }
  }

  async function stop(noRecording) {
    await props.recording.stopAndUnloadAsync();
    if (!noRecording) {
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        playsInSilentModeIOS: true,
      });

      const { sound, status } =
        await props.recording.createNewLoadedSoundAsync();
      sound.setOnPlaybackStatusUpdate(onPlaybackStatusUpdate);

      props.setRecording({
        sound: sound,
        file: props.recording.getURI(),
        status: status,
      });
    } else {
      props.setRecording(false);
    }
  }

  async function playAudio() {
    props.recording.sound.replayAsync();
    const result = await props.recording.sound.getStatusAsync();
  }

  const onPlaybackStatusUpdate = (status) => {
    setPlayingAudio(status.isPlaying);
  };

  return (
    <View>
      <View style={styles.recordContainer}>
        <TouchableOpacity
          style={styles.button}
          activeOpacity={1}
          onPress={() => {
            if (!props.currentlyRecording) {
              props.setCurrentlyRecording(true);
              record();
            }
          }}
        >
          <FontAwesome5
            name={"record-vinyl"}
            size={23}
            color={props.currentlyRecording ? COLORS.theme : "white"}
          />
          {!props.currentlyRecording && (
            <Text style={styles.text}>Press To Start Recording</Text>
          )}
        </TouchableOpacity>
        {props.currentlyRecording && (
          <>
            <Text style={styles.text}>Recording Audio</Text>
            <LoadingDots
              dots={3}
              size={8}
              bounceHeight={5}
              colors={[COLORS.theme, COLORS.theme, COLORS.theme]}
            />
            <View
              style={{
                flexDirection: "row",
                flex: 1,
                justifyContent: "flex-end",
              }}
            >
              <TouchableOpacity
                activeOpacity={1}
                style={{ marginRight: 10 }}
                onPress={() => {
                  props.setCurrentlyRecording(false);
                  stop(false);
                  //setAudio(true);
                }}
              >
                <Entypo name="save" size={23} color={COLORS.theme} />
              </TouchableOpacity>
              <TouchableOpacity
                activeOpacity={1}
                onPress={() => {
                  props.setCurrentlyRecording(!props.recording);
                  stop(true);
                }}
              >
                <AntDesign name="closesquare" size={23} color={COLORS.red} />
              </TouchableOpacity>
            </View>
          </>
        )}
      </View>

      {props.recording && !props.currentlyRecording && (
        <View style={styles.audioContainer}>
          <MaterialIcons name="audiotrack" size={23} color={COLORS.theme} />
          {!playingAudio ? (
            <Text style={styles.text}>
              Recorded Audio (
              {(props.recording?.status?.durationMillis / 1000).toFixed(1)}s)
            </Text>
          ) : (
            <View style={{ flexDirection: "row" }}>
              <Text style={styles.text}>Playing Audio</Text>
              <LoadingDots
                dots={3}
                size={8}
                bounceHeight={5}
                colors={[COLORS.theme, COLORS.theme, COLORS.theme]}
              />
            </View>
          )}
          <View style={styles.buttonView}>
            {!playingAudio && (
              <TouchableOpacity
                activeOpacity={1}
                style={{}}
                onPress={() => {
                  setPlayingAudio(!playingAudio);
                  playAudio();
                }}
              >
                <AntDesign name="playcircleo" size={23} color={COLORS.theme} />
              </TouchableOpacity>
            )}

            {playingAudio && (
              <TouchableOpacity
                activeOpacity={1}
                onPress={() => {
                  setPlayingAudio(!playingAudio);
                  props.recording.sound.stopAsync();
                }}
              >
                <AntDesign name="pause" size={23} color={COLORS.red} />
              </TouchableOpacity>
            )}
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  recordContainer: {
    backgroundColor: COLORS.dark2,
    marginHorizontal: 10,
    marginTop: 20,
    height: 40,
    borderRadius: 5,
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 10,
  },

  buttonView: {
    flexDirection: "row",
    flex: 1,
    justifyContent: "flex-end",
  },

  audioContainer: {
    backgroundColor: COLORS.dark2,
    marginHorizontal: 10,
    marginTop: 20,
    height: 40,
    borderRadius: 5,
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 10,
  },

  text: {
    color: COLORS.text,
    marginLeft: 10,
    marginRight: 10,
  },

  button: {
    flexDirection: "row",
    alignItems: "center",
  },
});
