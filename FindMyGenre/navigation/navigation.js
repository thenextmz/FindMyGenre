import React from "react";
import Home from "../screens/Home";
import { createDrawerNavigator } from "@react-navigation/drawer";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import Menue from "../screens/Menue";
import MusicView from "../screens/MusicView";

const Stack = createNativeStackNavigator();
const Drawer = createDrawerNavigator();

function Root() {
  return (
    <Drawer.Navigator
      useLegacyImplementation={true}
      initialRouteName="Home"
      drawerContent={(props) => <Menue {...props} />}
      screenOptions={{ headerShown: false }}
    >
      <Drawer.Screen name="Home" component={Home} />
    </Drawer.Navigator>
  );
}

const StackNavigator = () => {
  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      <Stack.Screen name="Root" component={Root} />
      <Stack.Screen name="Home" component={Home} />
      <Stack.Screen name="MusicView" component={MusicView} />
    </Stack.Navigator>
  );
};

export { StackNavigator };
