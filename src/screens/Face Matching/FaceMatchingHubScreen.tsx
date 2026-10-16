import React, { useState, useEffect } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, Alert } from 'react-native';
import { getAllFaceRecords, clearAllFaceRecords } from '../../utils/storage';

interface FaceMatchingHubScreenProps {
  onNavigate?: (screenName: string) => void;
}

export default function FaceMatchingHubScreen({ onNavigate }: FaceMatchingHubScreenProps) {
  const [recordCount, setRecordCount] = useState<number>(0);

  const refreshCount = async () => {
    const list = await getAllFaceRecords();
    setRecordCount(list.length);
  };

  useEffect(() => {
    refreshCount();
  }, []);

  const handleClearFaces = () => {
    if (recordCount === 0) {
      Alert.alert('Empty', 'No registered faces to delete.');
      return;
    }

    Alert.alert(
      'Clear Stored Faces',
      `Are you sure you want to delete all ${recordCount} registered faces?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete All',
          style: 'destructive',
          onPress: async () => {
            await clearAllFaceRecords();
            setRecordCount(0);
            Alert.alert('Cleared', 'All stored face records have been deleted.');
          },
        },
      ]
    );
  };

  return (
    <View style={styles.container}>
      <Text style={styles.headerTitle}>Face Matching</Text>
      <Text style={styles.subtitle}>Registered Identities: {recordCount}</Text>

      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={styles.cardButton}
          onPress={() => onNavigate?.('RegisterFace')}
        >
          <Text style={styles.buttonTitle}>Register Face</Text>
          <Text style={styles.buttonDesc}>Scan face, detect landmarks, and store 512-D embedding</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.cardButton, styles.verifyButton]}
          onPress={() => {
            if (recordCount === 0) {
              Alert.alert('No Records', 'Please register at least one face before running verification.');
              return;
            }
            onNavigate?.('VerifyFace');
          }}
        >
          <Text style={styles.buttonTitle}>Verify Face</Text>
          <Text style={styles.buttonDesc}>Scan live face and match against registered local database</Text>
        </TouchableOpacity>

        {/* Clear Data Button */}
        <TouchableOpacity style={styles.clearButton} onPress={handleClearFaces}>
          <Text style={styles.clearButtonText}>Clear All Registered Faces</Text>
        </TouchableOpacity>
      </View>

      <TouchableOpacity style={styles.backButton} onPress={() => onNavigate?.('Home')}>
        <Text style={styles.backButtonText}>← Back to Home</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#121212',
    padding: 24,
    justifyContent: 'center',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 6,
  },
  subtitle: {
    fontSize: 14,
    color: '#0A84FF',
    textAlign: 'center',
    marginBottom: 32,
  },
  buttonContainer: {
    gap: 16,
  },
  cardButton: {
    backgroundColor: '#1C1C1E',
    borderWidth: 1,
    borderColor: '#2C2C2E',
    borderRadius: 16,
    padding: 20,
  },
  verifyButton: {
    borderColor: '#0A84FF',
  },
  buttonTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#FFFFFF',
    marginBottom: 6,
  },
  buttonDesc: {
    fontSize: 13,
    color: '#A1A1A6',
    lineHeight: 18,
  },
  clearButton: {
    marginTop: 10,
    backgroundColor: 'rgba(255, 69, 58, 0.12)',
    borderWidth: 1,
    borderColor: '#FF453A',
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: 'center',
  },
  clearButtonText: {
    color: '#FF453A',
    fontWeight: '600',
    fontSize: 14,
  },
  backButton: {
    marginTop: 28,
    alignItems: 'center',
    padding: 10,
  },
  backButtonText: {
    color: '#8E8E93',
    fontSize: 15,
  },
});