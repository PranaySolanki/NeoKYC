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
  id_number: string | null;
}

interface BoundingBox {
  top?: number;
  left?: number;
  width?: number;
  height?: number;
  bottom?: number;
  right?: number;
  y?: number;
  x?: number;
}

interface SpatialLine {
  text: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

export const DocumentScanScreen = () => {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<any>(null);

  const [isTorchOn, setIsTorchOn] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [rawText, setRawText] = useState<string>('');
  const [parsedFields, setParsedFields] = useState<ExtractedFields | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>(
    'Position ID card inside the frame and capture.'
  );

  const cleanString = (str: string): string => {
    return str
      .replace(/[^a-zA-Z\s]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  };

  const parseMLKitResult = (ocrResult: any) => {
    const fullText: string = ocrResult.text || '';
    setRawText(fullText);

    // 1. Flatten all blocks and lines with spatial coordinates
    const spatialLines: SpatialLine[] = [];

    (ocrResult.blocks || []).forEach((b: any) => {
      if (b.lines && Array.isArray(b.lines)) {
        b.lines.forEach((l: any) => {
          const t = l.text?.trim();
          if (t && t.length > 1) {
            const frame: BoundingBox = l.frame || l.boundingBox || b.frame || b.boundingBox || {};
            const y = frame.top ?? frame.y ?? 0;
            const x = frame.left ?? frame.x ?? 0;
            const width = frame.width ?? 0;
            const height = frame.height ?? 0;

            spatialLines.push({ text: t, x, y, width, height });
          }
        });
      } else if (b.text) {
        const frame: BoundingBox = b.frame || b.boundingBox || {};
        spatialLines.push({
          text: b.text.trim(),
          x: frame.left ?? frame.x ?? 0,
          y: frame.top ?? frame.y ?? 0,
          width: frame.width ?? 0,
          height: frame.height ?? 0,
        });
      }
    });

    // 2. Sort all lines in 2D reading order (top-to-bottom)
    spatialLines.sort((a, b) => {
      if (Math.abs(a.y - b.y) > 15) {
        return a.y - b.y;
      }
      return a.x - b.x;
    });

    const normalizedFull: string = fullText.replace(/[^a-zA-Z0-9\s/:\-]/g, ' ');

    // 3. Extract 12-digit Card ID Number and strip all spaces
    const idMatch =
      normalizedFull.match(/\b\d{4}\s\d{4}\s\d{4}\b/) ||
      normalizedFull.match(/\b\d{12}\b/);
    const id_number: string | null = idMatch ? idMatch[0].replace(/\s+/g, '') : null;

    // 4. Extract Gender
    const genderMatch = normalizedFull.match(
      /\b(MALE|FEMALE|TRANSGENDER|Male|Female|Transgender)\b/i
    );
    const gender: string | null = genderMatch ? genderMatch[1].toUpperCase() : null;

    // 5. Extract Date of Birth & Spatial Anchor
    const dobRegex =
      /(?:DOB|Date of Birth|Year of Birth|DOB\/Date)[:\s\-]*([0-9]{2}[/\-][0-9]{2}[/\-][0-9]{4}|[0-9]{4})/i;
    const dateRegex = /\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b/;

    const dobMatch = fullText.match(dobRegex) || fullText.match(dateRegex);
    let dob: string | null = null;
    if (dobMatch) {
      dob = dobMatch[1] || dobMatch[0];
    }

    let dobSpatialAnchor: SpatialLine | null = null;
    let dobIndexInSorted = -1;

    for (let i = 0; i < spatialLines.length; i++) {
      const upper = spatialLines[i].text.toUpperCase();
      if (
        upper.includes('DOB') ||
        upper.includes('DATE OF BIRTH') ||
        upper.includes('YEAR OF BIRTH') ||
        dateRegex.test(upper)
      ) {
        dobSpatialAnchor = spatialLines[i];
        dobIndexInSorted = i;
        break;
      }
    }

    // 6. Name Extraction Filtering
    const blacklist: string[] = [
      'GOVERNMENT', 'INDIA', 'AUTHORITY', 'UNIQUE', 'IDENTIFICATION',
      'ENROLMENT', 'MALE', 'FEMALE', 'DOB', 'DATE', 'BIRTH', 'YEAR',
      'CARD', 'ADDRESS', 'FATHER', 'HUSBAND', 'HELP', 'WWW', 'UIDAI',
      'VID', 'DOWNLOAD', 'STATE', 'ROAD', 'NAGAR', 'MARG', 'LANE',
      'DIST', 'MAHARASHTRA', 'MUMBAI', 'DELHI', 'GUJARAT', 'COPILOT',
      'NEAR', 'BLOCK', 'FLAT', 'STREET', 'ENROLLMENT', 'AADHAAR', 'AADHAR'
    ];

    const isValidCandidate = (rawStr: string): boolean => {
      const cleaned = cleanString(rawStr);
      const upper = cleaned.toUpperCase();
      const words = cleaned.split(' ').filter((w) => w.length >= 2);

      const isBlacklisted = blacklist.some((term) => upper.includes(term));
      return !isBlacklisted && words.length >= 2 && words.length <= 4;
    };

    let name: string | null = null;

    // Strategy 1: Closest text situated directly above DOB in 2D space
    if (dobSpatialAnchor && dobSpatialAnchor.y > 0) {
      const candidatesAbove = spatialLines.filter((line) => {
        const isAbove = line.y < dobSpatialAnchor!.y;
        const isWithinRange = dobSpatialAnchor!.y - line.y < 350;
        return isAbove && isWithinRange && isValidCandidate(line.text);
      });

      if (candidatesAbove.length > 0) {
        const closest = candidatesAbove.sort((a, b) => b.y - a.y)[0];
        name = cleanString(closest.text);
      }
    }

    // Strategy 2: Sorted line reverse traversal
    if (!name && dobIndexInSorted > 0) {
      for (let j = dobIndexInSorted - 1; j >= Math.max(0, dobIndexInSorted - 4); j--) {
        if (isValidCandidate(spatialLines[j].text)) {
          name = cleanString(spatialLines[j].text);
          break;
        }
      }
    }

    // Strategy 3: Global fallback
    if (!name) {
      for (const line of spatialLines) {
        if (isValidCandidate(line.text)) {
          name = cleanString(line.text);
          break;
        }
      }
    }

    setParsedFields({ name, dob, gender, id_number });
  };

  const takePhotoAndScan = async () => {
    if (!cameraRef.current || isProcessing) return;

    try {
      setIsProcessing(true);
      setStatusMessage('Analyzing card via 2D Spatial Google ML Kit...');

      const photo = await cameraRef.current.takePictureAsync({
        quality: 1.0,
        skipProcessing: true,
      });

      if (photo?.uri) {
        const result = await recognizeText(photo.uri);
        parseMLKitResult(result);
        setStatusMessage('Document parsed successfully!');
      } else {
        setStatusMessage('Camera capture failed.');
      }
    } catch (err: any) {
      setStatusMessage('Scan Error: ' + (err.message || 'Processing failed'));
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

        {/* Flash Toggle */}
        <TouchableOpacity
          style={[styles.flashButton, isTorchOn && styles.flashButtonActive]}
          onPress={() => setIsTorchOn((prev) => !prev)}
        >
          <Text style={styles.flashIcon}>{isTorchOn ? '⚡ Flash ON' : '💡 Flash OFF'}</Text>
        </TouchableOpacity>

        {/* Alignment Window */}
        <View style={styles.guideBox}>
          <View style={[styles.corner, styles.topLeft]} />
          <View style={[styles.topRight, styles.corner]} />
          <View style={[styles.bottomLeft, styles.corner]} />
          <View style={[styles.bottomRight, styles.corner]} />
          <Text style={styles.guideText}>ALIGN ID CARD HERE</Text>
        </View>
      </View>

      {/* Capture Action */}
      <TouchableOpacity
        style={[styles.primaryButton, isProcessing && styles.disabledButton]}
        onPress={takePhotoAndScan}
        disabled={isProcessing}
      >
        {isProcessing ? (
          <ActivityIndicator color="#FFFFFF" />
        ) : (
          <Text style={styles.buttonText}>📸 Capture & Extract</Text>
        )}
      </TouchableOpacity>

      {/* Structured KYC Identity Record */}
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
            <Text style={styles.fieldKey}>Card ID Number:</Text>
            <Text style={[styles.fieldVal, styles.highlightVal]}>
              {parsedFields.id_number || 'Not detected'}
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
  permText: {
    color: '#94A3B8',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 20,
  },
  title: {
    color: '#F8FAFC',
    fontSize: 22,
    fontWeight: '700',
    marginBottom: 6,
  },
  subtitle: {
    color: '#94A3B8',
    fontSize: 13,
    lineHeight: 18,
    marginBottom: 16,
  },
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
  darkMask: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(2, 6, 23, 0.45)',
  },
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
  flashButtonActive: {
    backgroundColor: '#0284C7',
    borderColor: '#38BDF8',
  },
  flashIcon: {
    color: '#F8FAFC',
    fontSize: 12,
    fontWeight: '700',
  },
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
  corner: {
    position: 'absolute',
    width: 20,
    height: 20,
    borderColor: '#0EA5E9',
  },
  topLeft: { top: -2, left: -2, borderTopWidth: 3, borderLeftWidth: 3 },
  topRight: { top: -2, right: -2, borderTopWidth: 3, borderRightWidth: 3 },
  bottomLeft: { bottom: -2, left: -2, borderBottomWidth: 3, borderLeftWidth: 3 },
  bottomRight: { bottom: -2, right: -2, borderBottomWidth: 3, borderRightWidth: 3 },
  guideText: {
    color: 'rgba(248, 250, 252, 0.75)',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1.2,
  },
  primaryButton: {
    alignItems: 'center',
    backgroundColor: '#0EA5E9',
    borderRadius: 12,
    marginBottom: 20,
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
  labelsCard: {
    backgroundColor: '#1E293B',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#38BDF8',
  },
  cardHeader: {
    color: '#38BDF8',
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 1,
    marginBottom: 12,
  },
  fieldRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  fieldKey: {
    color: '#94A3B8',
    fontSize: 14,
    fontWeight: '600',
  },
  fieldVal: {
    color: '#F8FAFC',
    fontSize: 14,
    fontWeight: '700',
  },
  highlightVal: {
    color: '#38BDF8',
    letterSpacing: 0.8,
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
    minHeight: 100,
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
});