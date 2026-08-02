import React from 'react';
import { StyleSheet, Text, View, StatusBar } from 'react-native';

export default function App() {
  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      <Text style={styles.brandTitle}>NeoKYC</Text>
      <Text style={styles.tagline}>
        Next-Generation Digital Identity Verification for Deepfake and Fraud Detection using Machine Learning
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F172A',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
  },
  brandTitle: {
    color: '#38BDF8',
    fontSize: 32,
    fontWeight: '800',
    letterSpacing: 1.5,
    marginBottom: 12,
  },
  tagline: {
    color: '#94A3B8',
    fontSize: 13,
    fontWeight: '500',
    textAlign: 'center',
    lineHeight: 20,
  },
});