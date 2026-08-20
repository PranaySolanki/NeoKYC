import React, { useState, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  Dimensions,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { recognizeText } from '@infinitered/react-native-mlkit-text-recognition';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface ExtractedFields {
  name: string | null;
  dob: string | null;
  gender: string | null;
  id_detected: boolean;
}

export const DocumentScanScreen = () => {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<any>(null);

  const [isTorchOn, setIsTorchOn] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [rawText, setRawText] = useState('');
  const [parsedFields, setParsedFields] = useState<ExtractedFields | null>(null);
  const [statusMessage, setStatusMessage] = useState('Position ID card inside the frame and capture.');

  // Clean candidate name string
  const sanitizeName = (str: string): string => {
    return str
      .replace(/[^a-zA-Z\s]/g, '')
      .replace(/\s+/g, ' ')
      .trim();
  };

  // High-Precision ML Kit Spatial & Semantic Parser
  const parseMLKitResult = (ocrResult: any) => {
    const fullText = ocrResult.text || '';
    setRawText(fullText);

    const blocks = ocrResult.blocks || [];
    const allLines: string[] = [];

    blocks.forEach((b: any) => {
      if (b.lines) {
        b.lines.forEach((l: any) => allLines.push(l.text.trim()));
      } else if (b.text) {
        allLines.push(b.text.trim());
      }
    });

    const normalizedFull = fullText.replace(/[^a-zA-Z0-9\s/:\-]/g, ' ');

    // 1. Detect 12-digit ID Number pattern
    const idPattern = /\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b/;
    const id_detected = idPattern.test(normalizedFull);

    // 2. Extract Gender
    const genderMatch = normalizedFull.match(/\b(MALE|FEMALE|TRANSGENDER|Male|Female|Transgender)\b/i);
    const gender = genderMatch ? genderMatch[1].toUpperCase() : null;

    // 3. Extract DOB / YOB
    const dobRegex = /(?:DOB|Date of Birth|Year of Birth)[:\s\-]*([0-9]{2}[/\-][0-9]{2}[/\-][0-9]{4}|[0-9]{4})/i;
    const directDateRegex = /\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b/;

    const dobMatch = fullText.match(dobRegex) || fullText.match(directDateRegex);
    let dob: string | null = null;
    if (dobMatch) {
      dob = dobMatch[1] || dobMatch[0];
    }

    // 4. Extract Name (Line directly preceding the DOB line)
    const blacklist = [
      'GOVERNMENT', 'INDIA', 'AUTHORITY', 'UNIQUE', 'IDENTIFICATION',
      'ENROLMENT', 'MALE', 'FEMALE', 'DOB', 'DATE', 'BIRTH', 'YEAR',
      'CARD', 'ADDRESS', 'FATHER', 'HUSBAND', 'HELP', 'WWW', 'UIDAI'
    ];

    let anchorIndex = -1;
    for (let i = 0; i < allLines.length; i++) {
      const upper = allLines[i].toUpperCase();
      if (
        upper.includes('DOB') ||
        upper.includes('DATE OF BIRTH') ||
        upper.includes('YEAR OF BIRTH') ||
        /\d{2}[/\-]\d{2}[/\-]\d{4}/.test(upper)
      ) {
        anchorIndex = i;
        break;
      }
    }

    let name: string | null = null;

    if (anchorIndex > 0) {
      for (let j = anchorIndex - 1; j >= Math.max(0, anchorIndex - 3); j--) {
        const candidate = sanitizeName(allLines[j]);
        const upperCandidate = candidate.toUpperCase();
        const isBlacklisted = blacklist.some((term) => upperCandidate.includes(term));
        const words = candidate.split(' ').filter((w) => w.length >= 2);

        if (!isBlacklisted && words.length >= 2 && words.length <= 4) {
          name = words.map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');
          break;
        }
      }
    }

    if (!name) {
      for (const line of allLines) {
        const candidate = sanitizeName(line);
        const upperCandidate = candidate.toUpperCase();
        const isBlacklisted = blacklist.some((term) => upperCandidate.includes(term));
        const words = candidate.split(' ').filter((w) => w.length >= 2);

        if (!isBlacklisted && words.length >= 2 && words.length <= 4) {
          name = words.map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');
          break;
        }
      }
    }

    setParsedFields({ name, dob, gender, id_detected });
  };

  const takePhotoAndScan = async () => {
    if (!cameraRef.current || isProcessing) return;

    try {
      setIsProcessing(true);
      setStatusMessage('Running native on-device Google ML Kit...');

      const photo = await cameraRef.current.takePictureAsync({
        quality: 1.0,
        skipProcessing: true,
      });

      if (photo?.uri) {
        // Native C++ Google ML Kit call directly on the device
        const result = await recognizeText(photo.uri);
        parseMLKitResult(result);
        setStatusMessage('Document parsed successfully!');
      } else {
        setStatusMessage('Camera capture failed.');
      }
    } catch (err: any) {
      setStatusMessage('ML Kit Error: ' + (err.message || 'Processing failed'));
    } finally {
      setIsProcessing(false);
    }
  };

  if (!permission?.granted) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.permText}>Camera permission is required.</Text>
        <TouchableOpacity style={styles.primaryButton} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <ScrollView contentContainerStyle={styles.scrollContainer}>
      <Text style={styles.title}>Document Scan</Text>
      <Text style={styles.subtitle}>{statusMessage}</Text>

      {/* Live Viewfinder */}
      <View style={styles.cameraContainer}>
        <CameraView
          ref={cameraRef}
          style={StyleSheet.absoluteFillObject}
          facing="back"
          enableTorch={isTorchOn}
        />
        <View style={styles.darkMask} />

        {/* Flash Button */}
        <TouchableOpacity
          style={[styles.flashButton, isTorchOn && styles.flashButtonActive]}
          onPress={() => setIsTorchOn((prev) => !prev)}
        >
          <Text style={styles.flashIcon}>{isTorchOn ? '⚡ Flash ON' : '💡 Flash OFF'}</Text>
        </TouchableOpacity>

        {/* Framing Guide */}
        <View style={styles.guideBox}>
          <View style={[styles.corner, styles.topLeft]} />
          <View style={[styles.topRight, styles.corner]} />
          <View style={[styles.bottomLeft, styles.corner]} />
          <View style={[styles.bottomRight, styles.corner]} />
          <Text style={styles.guideText}>ALIGN ID CARD HERE</Text>
        </View>
      </View>

      {/* Capture Button */}
      <TouchableOpacity
        style={[styles.primaryButton, isProcessing && styles.disabledButton]}
        onPress={takePhotoAndScan}
        disabled={isProcessing}
      >
        {isProcessing ? (
          <ActivityIndicator color="#FFFFFF" />
        ) : (
          <Text style={styles.buttonText}>📸 Capture & Run ML Kit OCR</Text>
        )}
      </TouchableOpacity>

      {/* Parsed Identity Fields */}
      {parsedFields && (
        <View style={styles.labelsCard}>
          <Text style={styles.cardHeader}>PARSED IDENTITY FIELDS</Text>

          <View style={styles.fieldRow}>
            <Text style={styles.fieldKey}>Name:</Text>
            <Text style={styles.fieldVal}>{parsedFields.name || 'Not detected'}</Text>
          </View>

          <View style={styles.fieldRow}>
            <Text style={styles.fieldKey}>DOB / Year:</Text>
            <Text style={styles.fieldVal}>{parsedFields.dob || 'Not detected'}</Text>
          </View>

          <View style={styles.fieldRow}>
            <Text style={styles.fieldKey}>Gender:</Text>
            <Text style={styles.fieldVal}>{parsedFields.gender || 'Not detected'}</Text>
          </View>

          <View style={styles.fieldRow}>
            <Text style={styles.fieldKey}>ID Pattern:</Text>
            <Text style={[styles.fieldVal, { color: parsedFields.id_detected ? '#38BDF8' : '#94A3B8' }]}>
              {parsedFields.id_detected ? '12-Digit Format Verified' : 'Not detected'}
            </Text>
          </View>
        </View>
      )}

      {/* Raw Output Preview */}
      <Text style={styles.resultLabel}>ML Kit OCR Output</Text>
      <View style={styles.resultBox}>
        <Text style={rawText ? styles.rawText : styles.placeholderText}>
          {rawText || 'OCR output will appear here after capture.'}
        </Text>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  scrollContainer: {
    flexGrow: 1,
    backgroundColor: '#0F172A',
    paddingHorizontal: 20,
    paddingTop: 110,
    paddingBottom: 40,
  },
  centerContainer: {
    flex: 1,
    backgroundColor: '#0F172A',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
  },
  permText: { color: '#94A3B8', fontSize: 16, textAlign: 'center', marginBottom: 20 },
  title: { color: '#F8FAFC', fontSize: 22, fontWeight: '700', marginBottom: 6 },
  subtitle: { color: '#94A3B8', fontSize: 13, lineHeight: 18, marginBottom: 16 },
  cameraContainer: {
    width: '100%',
    height: 280,
    borderRadius: 16,
    overflow: 'hidden',
    backgroundColor: '#020617',
    borderWidth: 1,
    borderColor: '#334155',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
    position: 'relative',
  },
  darkMask: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(2, 6, 23, 0.45)' },
  flashButton: {
    position: 'absolute',
    top: 12,
    right: 12,
    backgroundColor: 'rgba(15, 23, 42, 0.8)',
    borderColor: '#334155',
    borderWidth: 1,
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 6,
    zIndex: 10,
  },
  flashButtonActive: { backgroundColor: '#0284C7', borderColor: '#38BDF8' },
  flashIcon: { color: '#F8FAFC', fontSize: 12, fontWeight: '700' },
  guideBox: {
    width: SCREEN_WIDTH * 0.88,
    height: SCREEN_WIDTH * 0.88 * 0.63,
    borderWidth: 1.5,
    borderColor: '#38BDF8',
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'transparent',
  },
  corner: { position: 'absolute', width: 20, height: 20, borderColor: '#0EA5E9' },
  topLeft: { top: -2, left: -2, borderTopWidth: 3, borderLeftWidth: 3 },
  topRight: { top: -2, right: -2, borderTopWidth: 3, borderRightWidth: 3 },
  bottomLeft: { bottom: -2, left: -2, borderBottomWidth: 3, borderLeftWidth: 3 },
  bottomRight: { bottom: -2, right: -2, borderBottomWidth: 3, borderRightWidth: 3 },
  guideText: { color: 'rgba(248, 250, 252, 0.75)', fontSize: 10, fontWeight: '800', letterSpacing: 1.2 },
  primaryButton: {
    alignItems: 'center',
    backgroundColor: '#0EA5E9',
    borderRadius: 12,
    marginBottom: 20,
    paddingVertical: 14,
  },
  disabledButton: { backgroundColor: '#475569' },
  buttonText: { color: '#FFFFFF', fontSize: 15, fontWeight: '700' },
  labelsCard: {
    backgroundColor: '#1E293B',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#38BDF8',
  },
  cardHeader: { color: '#38BDF8', fontSize: 12, fontWeight: '800', letterSpacing: 1, marginBottom: 12 },
  fieldRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 8 },
  fieldKey: { color: '#94A3B8', fontSize: 14, fontWeight: '600' },
  fieldVal: { color: '#F8FAFC', fontSize: 14, fontWeight: '700' },
  resultLabel: { color: '#CBD5E1', fontSize: 14, fontWeight: '700', marginBottom: 8 },
  resultBox: {
    backgroundColor: '#020617',
    borderColor: '#334155',
    borderRadius: 12,
    borderWidth: 1,
    minHeight: 100,
    padding: 16,
  },
  rawText: { color: '#E2E8F0', fontSize: 14, lineHeight: 21 },
  placeholderText: { color: '#64748B', fontSize: 14, fontStyle: 'italic' },
});