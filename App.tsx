import React, { useState, useEffect } from 'react';
import { StyleSheet, SafeAreaView, StatusBar, BackHandler } from 'react-native';
import { ModelProvider } from './src/assets/models/ModelContext';
import {
  HomeScreen,
  DocumentScanScreen,
  FaceLivenessRPPG,
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
  | 'FaceLivenessRPPG'
  | 'Result'
  | 'FaceMatchingHub'
  | 'RegisterFace'
  | 'VerifyFace';

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<ScreenName>('HomeDash');
  const [screenHistory, setScreenHistory] = useState<ScreenName[]>([]);
  const [routeParams, setRouteParams] = useState<Record<string, any>>({});
  const [flowMode, setFlowMode] = useState<'register' | 'verify'>('register');

  const navigateTo = (screen: string, params?: Record<string, any>) => {
    if (params) setRouteParams(params);

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

  const goBack = (): boolean => {
    if (screenHistory.length > 0) {
      const updated = [...screenHistory];
      const prev = updated.pop()!;
      setScreenHistory(updated);
      setCurrentScreen(prev);
      return true;
    }
    return false;
  };

  useEffect(() => {
    const backAction = () => goBack();
    const handler = BackHandler.addEventListener('hardwareBackPress', backAction);
    return () => handler.remove();
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

      case 'FaceLivenessRPPG':
        return (
          <FaceLivenessRPPG
            onNavigate={navigateTo}
            applicantData={routeParams?.applicantData}
            idSha256={routeParams?.idSha256}
          />
        );

      case 'Result':
        return (
          <ResultScreen
            onNavigate={navigateTo}
            applicantData={routeParams?.applicantData}
            idSha256={routeParams?.idSha256}
            liveVector={routeParams?.liveVector}
            livenessResult={routeParams?.livenessResult}
            pulseBpm={routeParams?.pulseBpm}
          />
        );

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
      <StatusBar barStyle="light-content" backgroundColor="#0F172A" />
      <ModelProvider>
        {renderCurrentScreen()}
      </ModelProvider>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  rootContainer: {
    flex: 1,
    backgroundColor: '#0F172A',
  },
});