import React, { useState } from 'react';
import { StyleSheet, SafeAreaView, StatusBar } from 'react-native';
import { ModelProvider } from './src/assets/models/ModelContext';
import {
  HomeScreen,
  DocumentScanScreen,
  FaceLivenessScreen,
  ResultScreen,
  FaceMatchingHubScreen,
  RegisterFaceScreen,
  VerifyFaceScreen,
} from './src/screens';

export type ScreenName =
  | 'Home'
  | 'DocumentScan'
  | 'FaceLiveness'
  | 'Result'
  | 'FaceMatchingHub'
  | 'RegisterFace'
  | 'VerifyFace';

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<ScreenName>('Home');
  const [routeParams, setRouteParams] = useState<Record<string, unknown>>({});

  const navigateTo = (screen: string, params?: Record<string, unknown>) => {
    if (params) {
      setRouteParams(params);
    }
    setCurrentScreen(screen as ScreenName);
  };

  const renderCurrentScreen = () => {
    switch (currentScreen) {
      // 1. Main Home Screen
      case 'Home':
        return <HomeScreen onNavigate={navigateTo} />;

      // 2. ID / Document Scanning Flow
      case 'DocumentScan':
        return <DocumentScanScreen onNavigate={navigateTo} />;

      // 3. Face Liveness Flow (Passes down any OCR/Document data from Step 2)
      case 'FaceLiveness':
        return <FaceLivenessScreen onNavigate={navigateTo} />;

      // 4. Verification / Results Screen
      // case 'Result':
      //   return <ResultScreen onNavigate={navigateTo} routeParams={routeParams} />;

      // 5. Offline Biometrics Hub
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