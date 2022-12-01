import React, { useEffect, useState } from "react";
import { COLORS } from "../theme/colors";
import { Divider } from "@rneui/base";
import { SafeAreaView } from "react-native-safe-area-context";
import { StyleSheet, Text, View } from "react-native";
import HeaderBack from "../components/musicView/HeaderBack";
import SongList from "../components/musicView/SongList";

export default function MusicView({ navigation, route }) {
  const { genre } = route.params;

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.body}>
        <HeaderBack navigation={navigation} />
        <Divider
          color={COLORS.theme}
          width={1.2}
          style={{ marginBottom: 10, marginHorizontal: 10 }}
        />
        <Text style={styles.header}>{genre}</Text>
        <SongList navigation={navigation} genre={genre} />
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

  header: {
    alignSelf: "center",
    fontSize: 30,
    fontWeight: "bold",
    color: COLORS.theme,
  },
});
