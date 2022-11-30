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

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.body}>
        <Header navigation={navigation} />
        <Divider
          color={COLORS.dark}
          width={1.2}
          style={{ marginBottom: 10, marginHorizontal: 10 }}
        />
        <ModeButtons mode={mode} setMode={setMode} />
        {mode && <ListenButton />}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: "white", //COLORS.theme,
    flex: 1,
  },
  body: {
    backgroundColor: "white",
    flex: 1,
  },
});
