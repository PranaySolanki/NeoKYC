import React from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';

interface HomeScreenProps {
  onNavigate: (screenName: string) => void;
}

export const HomeScreen: React.FC<HomeScreenProps> = ({ onNavigate }) => {
  return (
    <View style={styles.container}>
      <Text style={styles.brandTitle}>NeoKYC</Text>
      <Text style={styles.tagline}>
        Next-Generation Digital Identity Verification for Deepfake and Fraud Detection using Machine Learning
      </Text>

      <View style={styles.navContainer}>
        <Text style={styles.sectionHeader}>Test Navigation</Text>

        <TouchableOpacity 
          style={styles.button} 
          onPress={() => onNavigate('DocumentScan')}
        >
          <Text style={styles.buttonText}>1. Document Scan Screen</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={[styles.button, styles.secondaryButton]} 
          onPress={() => onNavigate('FaceLiveness')}
        >
          <Text style={styles.buttonText}>2. Liveness Check Screen</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={[styles.button, styles.accentButton]} 
          onPress={() => onNavigate('Result')}
        >
          <Text style={styles.buttonText}>3. Verification Result Screen</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

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
    marginBottom: 8,
  },
  tagline: {
    color: '#94A3B8',
    fontSize: 13,
    fontWeight: '500',
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 40,
  },
  navContainer: {
    width: '100%',
    gap: 12,
  },
  sectionHeader: {
    color: '#64748B',
    fontSize: 12,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 4,
    textAlign: 'center',
  },
  button: {
    backgroundColor: '#0EA5E9',
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderRadius: 12,
    alignItems: 'center',
  },
  secondaryButton: {
    backgroundColor: '#0284C7',
  },
  accentButton: {
    backgroundColor: '#0D9488',
  },
  buttonText: {
    color: '#FFFFFF',
    fontWeight: '700',
    fontSize: 15,
  },
});