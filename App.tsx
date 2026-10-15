import React, { useState } from 'react';
import { StyleSheet, View, StatusBar, TouchableOpacity, Text } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import {
  HomeScreen,
  DocumentScanScreen,
  FaceLivenessScreen,
  ResultScreen
} from './src/screens';
import { RppgResult } from './src/services/rppgService';

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<string>('Home');
  const [livenessResult, setLivenessResult] = useState<RppgResult | null>(null);

  const renderScreen = () => {
    switch (currentScreen) {
      case 'DocumentScan':
        return <DocumentScanScreen onNavigate={setCurrentScreen} />;
      case 'FaceLiveness':
        return (
          <FaceLivenessScreen
            onNavigate={setCurrentScreen}
            onComplete={setLivenessResult}
          />
        );
      case 'Result':
        return (
          <ResultScreen
            livenessResult={livenessResult}
            onNavigate={setCurrentScreen}
          />
        );
      case 'Home':
      default:
        return <HomeScreen onNavigate={setCurrentScreen} />;
    }
  };

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" />

        {/* Render Back Button on secondary screens */}
        {currentScreen !== 'Home' && (
          <TouchableOpacity
            style={styles.backButton}
            onPress={() => setCurrentScreen('Home')}
          >
            <Text style={styles.backButtonText}>← Back to Home</Text>
          </TouchableOpacity>
        )}

        <View style={styles.content}>
          {renderScreen()}
        </View>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F172A',
  },
  content: {
    flex: 1,
  },
  backButton: {
    position: 'absolute',
    top: 50,
    left: 20,
    zIndex: 10,
    backgroundColor: '#1E293B',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#334155',
  },
  backButtonText: {
    color: '#38BDF8',
    fontSize: 14,
    fontWeight: '600',
  },
});