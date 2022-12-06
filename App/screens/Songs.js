import { StyleSheet, Text, View } from "react-native";
import React from "react";
import { COLORS } from "../colors";
import { SafeAreaView } from "react-native-safe-area-context";
import GenreHeader from "../components/Songs/GenreHeader";

export default function Songs({ navigation, route }) {
  const { genre } = route.params;

  return (
    <SafeAreaView style={styles.safeArea}>
      <GenreHeader genre={genre} navigation={navigation} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: COLORS.dark,
  },
});
