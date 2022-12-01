import React, { useEffect, useState } from "react";
import { COLORS } from "../theme/colors";
import { Divider } from "@rneui/base";
import { SafeAreaView } from "react-native-safe-area-context";
import { ActivityIndicator, StyleSheet, Text, View } from "react-native";
import Header from "../components/home/Header";
import ModeButtons from "../components/home/ModeButtons";
import ListenButton from "../components/home/ListenButton";
import GenreList from "../components/home/GenreList";

export default function Home({ navigation }) {
  const [mode, setMode] = useState(true);
  const [recording, setRecording] = React.useState(null);
  const [loading, setLoading] = useState(false);
  const [genres, setGenres] = useState(false);

  const findGenre = () => {
    // TODO: send recording to Classifier

    // fake 2 seconds here
    new Promise((resolve) => setTimeout(resolve, 2000)).then(() => {
      setLoading(false);
      setGenres(true);
    });
  };

  useEffect(() => {
    if (loading && recording) {
      console.log("hello");
      findGenre();
    }
  }, [loading, recording]);

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.body}>
        <Header navigation={navigation} />
        <Divider
          color={COLORS.theme}
          width={1.2}
          style={{ marginBottom: 10, marginHorizontal: 10 }}
        />
        <ModeButtons mode={mode} setMode={setMode} />
        {mode && (
          <ListenButton
            recording={recording}
            setRecording={setRecording}
            setLoading={setLoading}
          />
        )}

        {mode && !loading && recording && genres && (
          <GenreList navigation={navigation} />
        )}

        {!mode && <GenreList navigation={navigation} />}

        {loading && (
          <View style={{ flex: 1, marginTop: 20 }}>
            <ActivityIndicator size={40} color="black" />
          </View>
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: COLORS.theme_light, //COLORS.theme,
    flex: 1,
  },
  body: {
    backgroundColor: "white",
    flex: 1,
  },
});
