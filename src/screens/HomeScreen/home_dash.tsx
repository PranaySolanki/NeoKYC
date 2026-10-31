import React from 'react';
import { StyleSheet, Text, View, TouchableOpacity, StatusBar } from 'react-native';

interface HomeDashProps {
  onNavigate: (screenName: string, data?: any) => void;
}

export const HomeDash: React.FC<HomeDashProps> = ({ onNavigate }) => {
  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />

      {/* Header with Quick Link Button to HomeScreen */}
      <View style={styles.topBar}>
        <TouchableOpacity
          style={styles.homeLinkBtn}
          onPress={() => onNavigate('Home')}
          activeOpacity={0.7}
        >
          <Text style={styles.homeLinkText}>⚙️ Test Home</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.header}>
        <Text style={styles.brandTitle}>NeoKYC</Text>
        <Text style={styles.subtitle}>On-Device Biometric & Document Verification</Text>
      </View>

      <View style={styles.actionContainer}>
        {/* Register Button */}
        <TouchableOpacity
          style={styles.primaryButton}
          onPress={() => onNavigate('DocumentScan_Register')}
          activeOpacity={0.8}
        >
          <View style={styles.iconCircle}>
            <Text style={styles.iconText}>👤+</Text>
          </View>
          <View style={styles.btnTextContainer}>
            <Text style={styles.buttonTitle}>Register User</Text>
            <Text style={styles.buttonDesc}>Scan card, detect face, and enroll biometric template</Text>
          </View>
        </TouchableOpacity>

        {/* Verify Button */}
        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={() => onNavigate('DocumentScan_Verify')}
          activeOpacity={0.8}
        >
          <View style={[styles.iconCircle, styles.verifyCircle]}>
            <Text style={styles.iconText}>🛡️</Text>
          </View>
          <View style={styles.btnTextContainer}>
            <Text style={styles.buttonTitle}>Verify User</Text>
            <Text style={styles.buttonDesc}>Authenticate live face against enrolled identities</Text>
          </View>
        </TouchableOpacity>
      </View>

      <View style={styles.footer}>
        <Text style={styles.footerText}>Secure Zero-Trust Identity Pipeline</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F172A',
    justifyContent: 'space-between',
    paddingHorizontal: 24,
    paddingTop: 60,
    paddingBottom: 40,
  },
  topBar: {
    width: '100%',
    flexDirection: 'row',
    justifyContent: 'flex-end',
    marginBottom: 8,
  },
  homeLinkBtn: {
    backgroundColor: 'rgba(30, 41, 59, 0.9)',
    borderColor: '#334155',
    borderWidth: 1,
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 8,
  },
  homeLinkText: {
    color: '#38BDF8',
    fontSize: 12,
    fontWeight: '700',
  },
  header: {
    alignItems: 'center',
  },
  brandTitle: {
    color: '#38BDF8',
    fontSize: 34,
    fontWeight: '800',
    letterSpacing: 1.2,
    marginBottom: 8,
  },
  subtitle: {
    color: '#94A3B8',
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 20,
  },
  actionContainer: {
    width: '100%',
    gap: 20,
  },
  primaryButton: {
    backgroundColor: '#1E293B',
    borderColor: '#0284C7',
    borderWidth: 1.5,
    borderRadius: 16,
    padding: 20,
    flexDirection: 'row',
    alignItems: 'center',
  },
  secondaryButton: {
    backgroundColor: '#1E293B',
    borderColor: '#334155',
    borderWidth: 1.5,
    borderRadius: 16,
    padding: 20,
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconCircle: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(2, 132, 199, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  verifyCircle: {
    backgroundColor: 'rgba(52, 211, 153, 0.2)',
  },
  iconText: {
    fontSize: 22,
  },
  btnTextContainer: {
    flex: 1,
  },
  buttonTitle: {
    color: '#F8FAFC',
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 4,
  },
  buttonDesc: {
    color: '#94A3B8',
    fontSize: 12,
    lineHeight: 16,
  },
  footer: {
    alignItems: 'center',
  },
  footerText: {
    color: '#475569',
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
});