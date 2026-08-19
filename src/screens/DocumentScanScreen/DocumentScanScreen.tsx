import { recognizeText } from '@infinitered/react-native-mlkit-text-recognition';
import * as ImagePicker from 'expo-image-picker';
import React, { useState } from 'react';
import {
  Alert,
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';

export const DocumentScanScreen = () => {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [rawText, setRawText] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [isRecognizing, setIsRecognizing] = useState(false);

  const chooseDocument = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permission.granted) {
      Alert.alert(
        'Photo access needed',
        'Please allow photo access so you can select an identity document.'
      );
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      quality: 1,
    });

    if (result.canceled || !result.assets?.[0]) {
      return;
    }

    setImageUri(result.assets[0].uri);
    setRawText('');
    setErrorMessage('');
  };

  const extractRawText = async () => {
    if (!imageUri) {
      Alert.alert('Choose a document', 'Select an image before extracting text.');
      return;
    }

    setIsRecognizing(true);
    setErrorMessage('');

    try {
      const result = await recognizeText(imageUri);
      setRawText(result.text || 'No readable text was found in this image.');
    } catch {
      setRawText('');
      setErrorMessage('Text extraction failed. Please try a clearer document image.');
    } finally {
      setIsRecognizing(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Document Scan</Text>

      <Text style={styles.subtitle}>
        Select an ID document, then extract its unformatted text.
      </Text>

      <Pressable style={styles.secondaryButton} onPress={chooseDocument}>
        <Text style={styles.buttonText}>Choose Document Image</Text>
      </Pressable>

      {imageUri ? (
        <Image source={{ uri: imageUri }} style={styles.preview} resizeMode="contain" />
      ) : (
        <View style={styles.emptyPreview}>
          <Text style={styles.emptyPreviewText}>No document selected</Text>
        </View>
      )}

      <Pressable
        style={[styles.primaryButton, (!imageUri || isRecognizing) && styles.disabledButton]}
        onPress={extractRawText}
        disabled={!imageUri || isRecognizing}
      >
        <Text style={styles.buttonText}>
          {isRecognizing ? 'Extracting text…' : 'Extract Raw Text'}
        </Text>
      </Pressable>

      <Text style={styles.resultLabel}>Extracted Raw Text</Text>

      <View style={styles.resultBox}>
        <Text style={rawText ? styles.rawText : styles.placeholderText}>
          {rawText || 'Your OCR result will appear here.'}
        </Text>
      </View>

      {!!errorMessage && <Text style={styles.errorText}>{errorMessage}</Text>}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    backgroundColor: '#0F172A',
    paddingHorizontal: 24,
    paddingTop: 116,
    paddingBottom: 32,
  },
  title: {
    color: '#F8FAFC',
    fontSize: 22,
    fontWeight: '700',
    marginBottom: 8,
  },
  subtitle: {
    color: '#94A3B8',
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 24,
  },
  primaryButton: {
    alignItems: 'center',
    backgroundColor: '#0EA5E9',
    borderRadius: 12,
    marginBottom: 24,
    paddingHorizontal: 20,
    paddingVertical: 14,
  },
  secondaryButton: {
    alignItems: 'center',
    backgroundColor: '#1E293B',
    borderColor: '#334155',
    borderRadius: 12,
    borderWidth: 1,
    marginBottom: 16,
    paddingHorizontal: 20,
    paddingVertical: 14,
  },
  disabledButton: {
    backgroundColor: '#475569',
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '700',
  },
  preview: {
    backgroundColor: '#020617',
    borderColor: '#334155',
    borderRadius: 12,
    borderWidth: 1,
    height: 180,
    marginBottom: 16,
    width: '100%',
  },
  emptyPreview: {
    alignItems: 'center',
    backgroundColor: '#020617',
    borderColor: '#334155',
    borderRadius: 12,
    borderStyle: 'dashed',
    borderWidth: 1,
    height: 180,
    justifyContent: 'center',
    marginBottom: 16,
  },
  emptyPreviewText: {
    color: '#64748B',
    fontSize: 14,
  },
  resultLabel: {
    color: '#CBD5E1',
    fontSize: 14,
    fontWeight: '700',
    marginBottom: 8,
  },
  resultBox: {
    backgroundColor: '#020617',
    borderColor: '#334155',
    borderRadius: 12,
    borderWidth: 1,
    minHeight: 160,
    padding: 16,
  },
  rawText: {
    color: '#E2E8F0',
    fontSize: 14,
    lineHeight: 21,
  },
  placeholderText: {
    color: '#64748B',
    fontSize: 14,
    fontStyle: 'italic',
  },
  errorText: {
    color: '#FCA5A5',
    fontSize: 13,
    lineHeight: 19,
    marginTop: 12,
  },
});