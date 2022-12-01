import { StyleSheet, Text, View } from "react-native";
import React, { useState } from "react";
import { SafeAreaView } from "react-native-safe-area-context";
import Header from "../components/home/Header";
import { COLORS } from "../theme/colors";
import ModeButtons from "../components/home/ModeButtons";
import { Divider } from "@rneui/base";
import ListenButton from "../components/home/ListenButton";

export default function Home({ navigation }) {
  const [mode, setMode] = useState(true);
  const [recording, setRecording] = React.useState(null);

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
          <ListenButton recording={recording} setRecording={setRecording} />
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
