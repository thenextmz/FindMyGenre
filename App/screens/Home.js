import { StyleSheet, Text, View, StatusBar } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import React, { useState } from "react";
import { COLORS } from "../colors";
import Header from "../components/Home/Header";
import OptionButtons from "../components/Home/OptionButtons";
import GenreList from "../components/Home/GenreList";

export default function Home({ navigation }) {
  StatusBar.setBarStyle("light-content", true);
  const [listen, setListen] = useState(true);
  return (
    <SafeAreaView style={styles.safeArea}>
      <Header />
      <OptionButtons
        navigation={navigation}
        active={listen}
        setActive={setListen}
      />

      {!listen && <GenreList navigation={navigation} />}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: COLORS.dark,
  },
});
