import React, { useState, useEffect } from 'react';
import { StyleSheet, SafeAreaView, StatusBar, BackHandler } from 'react-native';
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
import { HomeDash } from './src/screens/HomeScreen/home_dash';

export type ScreenName =
  | 'Home'
  | 'HomeDash'
  | 'DocumentScan'
  | 'FaceLiveness'
  | 'Result'
  | 'FaceMatchingHub'
  | 'RegisterFace'
  | 'VerifyFace';

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<ScreenName>('HomeDash');
  const [screenHistory, setScreenHistory] = useState<ScreenName[]>([]);
  const [routeParams, setRouteParams] = useState<Record<string, any>>({});
  const [flowMode, setFlowMode] = useState<'register' | 'verify'>('register');

  // Push new screens to history when navigating
  const navigateTo = (screen: string, params?: Record<string, any>) => {
    if (params) {
      setRouteParams(params);
    }

    if (screen === 'DocumentScan_Register') {
      setFlowMode('register');
      setScreenHistory((prev) => [...prev, currentScreen]);
      setCurrentScreen('DocumentScan');
      return;
    }

    if (screen === 'DocumentScan_Verify') {
      setFlowMode('verify');
      setScreenHistory((prev) => [...prev, currentScreen]);
      setCurrentScreen('DocumentScan');
      return;
    }

    setScreenHistory((prev) => [...prev, currentScreen]);
    setCurrentScreen(screen as ScreenName);
  };

  // Pop and return to previous screen
  const goBack = (): boolean => {
    if (screenHistory.length > 0) {
      const updatedHistory = [...screenHistory];
      const previousScreen = updatedHistory.pop()!;
      setScreenHistory(updatedHistory);
      setCurrentScreen(previousScreen);
      return true; // Tells Android: event handled, do NOT close app
    }
    return false; // On root screen: allow default exit behavior
  };

  // Bind Android hardware back press and side-swipe gestures
  useEffect(() => {
    const backAction = () => {
      return goBack();
    };

    const backHandler = BackHandler.addEventListener(
      'hardwareBackPress',
      backAction
    );

    return () => backHandler.remove();
  }, [screenHistory, currentScreen]);

  const renderCurrentScreen = () => {
    switch (currentScreen) {
      case 'HomeDash':
        return <HomeDash onNavigate={navigateTo} />;

      case 'Home':
        return <HomeScreen onNavigate={navigateTo} />;

      case 'DocumentScan':
        return (
          <DocumentScanScreen
            onNavigate={navigateTo}
            flowMode={flowMode}
          />
        );

      case 'FaceLiveness':
        return <FaceLivenessScreen onNavigate={navigateTo} />;

      case 'FaceMatchingHub':
        return <FaceMatchingHubScreen onNavigate={navigateTo} />;

      case 'RegisterFace':
        return (
          <RegisterFaceScreen
            onNavigate={navigateTo}
            applicantData={routeParams?.applicantData}
          />
        );

      case 'VerifyFace':
        return <VerifyFaceScreen onNavigate={navigateTo} />;

      default:
        return <HomeDash onNavigate={navigateTo} />;
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