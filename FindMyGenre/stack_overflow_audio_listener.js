import React from 'react';
import { Button, StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import { Audio } from 'expo-av';
import { MaterialCommunityIcons } from '@expo/vector-icons';

export default function AppVoice() {
  const [recording, setRecording] = React.useState();
  const [recordings, setRecordings] = React.useState([]);
  const [message, setMessage] = React.useState("");

  async function startRecording() {
    try {
        
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
        setMessage("Please grant permission to app to access microphone");
      }
    } catch (err) {
      console.error('Failed to start recording', err);
    }
  }

  async function stopRecording() {
    setRecording(undefined);
    await recording.stopAndUnloadAsync();
    await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        playsInSilentModeIOS: true,
        
      });

    let updatedRecordings = [...recordings];
    const { sound, status } = await recording.createNewLoadedSoundAsync();
    updatedRecordings.push({
      sound: sound,
      duration: getDurationFormatted(status.durationMillis),
      file: recording.getURI()
    });
    
    
    setRecordings(updatedRecordings);
  }

  function getDurationFormatted(millis) {
    const minutes = millis / 1000 / 60;
    const minutesDisplay = Math.floor(minutes);
    const seconds = Math.round((minutes - minutesDisplay) * 60);
    const secondsDisplay = seconds < 10 ? `0${seconds}` : seconds;
    return `${minutesDisplay}:${secondsDisplay}`;
  }

  function getRecordingLines() {
    
    return recordings.map((recordingLine, index) => {
      return (
        <View key={index} style={styles.row}>
          <Text style={styles.fill}>Recording {index + 1} - {recordingLine.duration}</Text>
          <Button style={styles.button} onPress={() => recordingLine.sound.replayAsync()} title="Play"></Button>
        </View>
      );
    });
  }

  return (


        <View style={styles.container}>
            <View style={styles.recorder}>
                <TouchableOpacity  style={{position:'absolute', left:10}}>
                    <MaterialCommunityIcons  name="microphone" size={24} color="black" />
                </TouchableOpacity>
                <TouchableOpacity onPress={recording ? stopRecording : startRecording} style={{position:'absolute', right:10}}>   
                    {recording ? <MaterialCommunityIcons  name="pause" size={28} color="black" /> : <MaterialCommunityIcons  name="record-circle-outline" size={28} color="red" />}
                    
                </TouchableOpacity>
            </View>
            <View style={{flex:1}}>
                {getRecordingLines()}
            </View>
        </View>
  );
}

const styles = StyleSheet.create({
  recorder: {

    width:300,
    backgroundColor:'white',
    height:50,
    borderRadius: 100,
    justifyContent:'center'
  },
  container:{
    flex:1,
    
    
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  fill: {
    flex: 1,
    margin: 16,
    color:'white'
  },
  button: {
    margin: 16
  }
});