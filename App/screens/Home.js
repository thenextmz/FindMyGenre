import { StyleSheet, Text, View, StatusBar } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import React, { useState } from "react";
import { COLORS } from "../colors";
import Header from "../components/Home/Header";
import OptionButtons from "../components/Home/OptionButtons";
import GenreList from "../components/Home/GenreList";
import Listener from "../components/Home/Listener";

export default function Home({ navigation }) {
  StatusBar.setBarStyle("light-content", true);
  const [mode, setMode] = useState(0);

  // TODO: not very elegant but idk if possible in another way with not
  // to much work
  const [currentlyRecording, setCurrentlyRecording] = useState(false);
  const [recording, setRecording] = useState(false);

  return (
    <SafeAreaView style={styles.safeArea}>
      <Header />
      <OptionButtons
        navigation={navigation}
        mode={mode}
        setMode={setMode}
        currentlyRecording={currentlyRecording}
        setCurrentlyRecording={setCurrentlyRecording}
        recording={recording}
        setRecording={setRecording}
      />
      {mode === 0 && (
        <Listener
          navigation={navigation}
          currentlyRecording={currentlyRecording}
          setCurrentlyRecording={setCurrentlyRecording}
          recording={recording}
          setRecording={setRecording}
        />
      )}
      {mode === 1 && <GenreList navigation={navigation} />}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: COLORS.dark,
  },
});
