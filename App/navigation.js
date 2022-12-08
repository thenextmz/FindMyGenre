import React from "react";
import Home from "./screens/Home";
import Songs from "./screens/Songs";
import { createNativeStackNavigator } from "@react-navigation/native-stack";

const Stack = createNativeStackNavigator();

const StackNavigator = () => {
  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      <Stack.Screen name="Home" component={Home} />
      <Stack.Screen name="Songs" component={Songs} />
    </Stack.Navigator>
  );
};

export { StackNavigator };
