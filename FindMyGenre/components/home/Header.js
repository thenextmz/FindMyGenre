import React from "react";
import { COLORS } from "../../theme/colors";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import FontAwesome5 from "react-native-vector-icons/FontAwesome5";

export default function Header({ navigation }) {
  return (
    <View style={{ overflow: "hidden", paddingBottom: 5, marginBottom: 0 }}>
      <View style={styles.container}>
        <TouchableOpacity
          style={{ flex: 1 }}
          onPress={() => {
            navigation.openDrawer();
          }}
        >
          <FontAwesome5 name={"bars"} size={20} color={COLORS.dark} />
        </TouchableOpacity>

        <View style={{ flex: 1 }}>
          <Text style={styles.logo}>FindMyGenre</Text>
        </View>

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
    color: COLORS.dark,
  },
});
