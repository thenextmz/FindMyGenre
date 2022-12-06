import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import React, { useState } from "react";
import { COLORS } from "../../colors";
import FontAwesome5 from "react-native-vector-icons/FontAwesome5";
import Entypo from "react-native-vector-icons/Entypo";
import AntDesign from "react-native-vector-icons/AntDesign";
import LoadingDots from "react-native-loading-dots";

export default function Listener() {
  const [recording, setRecording] = useState(false);
  return (
    <View>
      <View style={styles.recordContainer}>
        <TouchableOpacity
          style={styles.button}
          activeOpacity={1}
          onPress={() => {
            if (!recording) {
              setRecording(!recording);
            }
          }}
        >
          <FontAwesome5
            name={"record-vinyl"}
            size={23}
            color={recording ? COLORS.green : "white"}
          />
          {!recording && (
            <Text style={styles.text}>Press To Start Recording</Text>
          )}
        </TouchableOpacity>
        {recording && (
          <>
            <Text style={styles.text}>Recording Audio</Text>
            <LoadingDots
              dots={3}
              size={8}
              bounceHeight={5}
              colors={[COLORS.green, COLORS.green, COLORS.green]}
            />
            <View
              style={{
                flexDirection: "row",
                flex: 1,
                justifyContent: "flex-end",
              }}
            >
              <TouchableOpacity
                activeOpacity={1}
                style={{ marginRight: 10 }}
                onPress={() => {}}
              >
                <Entypo name="save" size={23} color={COLORS.green} />
              </TouchableOpacity>
              <TouchableOpacity
                activeOpacity={1}
                onPress={() => {
                  setRecording(!recording);
                }}
              >
                <AntDesign name="closesquare" size={23} color={COLORS.red} />
              </TouchableOpacity>
            </View>
          </>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  recordContainer: {
    backgroundColor: COLORS.dark2,
    marginHorizontal: 10,
    marginTop: 20,
    height: 40,
    borderRadius: 5,
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 10,
  },

  text: {
    color: COLORS.text,
    marginLeft: 10,
    marginRight: 10,
  },

  button: {
    flexDirection: "row",
    alignItems: "center",
  },
});
