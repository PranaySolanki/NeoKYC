import React, { useState } from 'react';
import { StyleSheet, StatusBar } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  HomeScreen,
} from './src/screens';
import RegisterFaceScreen from './src/screens/RetinaFace+ArcFace/RegisterFaceScreen';
import FaceMatchingHubScreen from './src/screens/Face Matching/FaceMatchingHubScreen';
import VerifyFaceScreen from './src/screens/Face Matching/VerifyFaceScreen';
import { ModelProvider } from './src/assets/models/ModelContext';

export type ScreenName = 'Home' | 'FaceMatchingHub' | 'RegisterFace' | 'VerifyFace';

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<ScreenName>('Home');

  const navigateTo = (screen: string) => {
    setCurrentScreen(screen as ScreenName);
  };

  const renderCurrentScreen = () => {
    switch (currentScreen) {
      case 'Home':
        return <HomeScreen onNavigate={navigateTo} />;

      case 'FaceMatchingHub':
        return <FaceMatchingHubScreen onNavigate={navigateTo} />;

      case 'RegisterFace':
        return <RegisterFaceScreen onNavigate={navigateTo} />;

      case 'VerifyFace':
        return <VerifyFaceScreen onNavigate={navigateTo} />;

      default:
        return <HomeScreen onNavigate={navigateTo} />;
    }
  };

  return (
    <SafeAreaView style={styles.rootContainer}>
      <StatusBar barStyle="light-content" backgroundColor="#121212" />
      <ModelProvider>
        {renderCurrentScreen()}
      </ModelProvider>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  rootContainer: {
    flex: 1,
    backgroundColor: '#121212',
  },
});