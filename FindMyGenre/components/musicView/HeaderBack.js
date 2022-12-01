import React from "react";
import { COLORS } from "../../theme/colors";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import MaterialIcons from "react-native-vector-icons/MaterialIcons";

export default function HeaderBack({ navigation }) {
  return (
    <View style={{ overflow: "hidden", paddingBottom: 5, marginBottom: 0 }}>
      <View style={styles.container}>
        <TouchableOpacity
          style={{ flex: 1 }}
          onPress={() => {
            navigation.goBack();
          }}
        >
          <MaterialIcons
            name={"arrow-back-ios"}
            size={20}
            color={COLORS.theme}
          />
        </TouchableOpacity>

        <TouchableOpacity
          style={{ flex: 1 }}
          onPress={() => {
            navigation.navigate("Root");
          }}
        >
          <Text style={styles.logo}>FindMyGenre</Text>
        </TouchableOpacity>

        <View style={{ flex: 1 }} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    // backgroundColor: COLORS.theme,
    alignItems: "center",
    justifyContent: "space-between",
    flexDirection: "row",
    paddingHorizontal: 15,
    paddingVertical: 10,
    paddingBottom: 10,
    borderBottomLeftRadius: 10,
    borderBottomRightRadius: 10,
  },

  logo: {
    fontSize: 18,
    fontWeight: "600",
    color: COLORS.theme,
  },
});
