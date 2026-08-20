import React, { useState } from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import {
  WelcomeScreen,
  DocumentCaptureScreen,
  FaceLivenessScreen,
  ResultScreen,
  HomeScreen,
} from '../screens';

const MOCK_MODE = true;

export type RootStackParamList = {
  Welcome: undefined;
  DocumentCapture: undefined;
  FaceLiveness: { documentUri: string };
  Result: { verified: boolean; confidence: number; reason?: string };
  Home: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

async function runMockVerification(): Promise<{ verified: boolean; confidence: number; reason?: string }> {
  await new Promise((resolve) => setTimeout(resolve, 1200));
  return { verified: true, confidence: 0.93 };
}

export default function AppNavigator() {
  const [activity, setActivity] = useState<
    { id: string; title: string; timestamp: string; status: 'verified' | 'rejected' | 'pending' }[]
  >([]);
  const [verifiedAt, setVerifiedAt] = useState<string | undefined>(undefined);

  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      <Stack.Screen name="Welcome">
        {({ navigation }) => (
          <WelcomeScreen onGetStarted={() => navigation.navigate('DocumentCapture')} />
        )}
      </Stack.Screen>

      <Stack.Screen name="DocumentCapture">
        {({ navigation }) => (
          <DocumentCaptureScreen
            onCancel={() => navigation.goBack()}
            onCaptured={(uri) => navigation.navigate('FaceLiveness', { documentUri: uri })}
          />
        )}
      </Stack.Screen>

      <Stack.Screen name="FaceLiveness">
        {({ navigation, route }) => (
          <FaceLivenessScreen
            documentUri={route.params.documentUri}
            onCancel={() => navigation.goBack()}
            onComplete={async () => {
              const outcome = MOCK_MODE
                ? await runMockVerification()
                : await runMockVerification();
              navigation.replace('Result', outcome);
            }}
          />
        )}
      </Stack.Screen>

      <Stack.Screen name="Result">
        {({ navigation, route }) => (
          <ResultScreen
            verified={route.params.verified}
            confidence={route.params.confidence}
            reason={route.params.reason}
            onRetry={() => navigation.navigate('DocumentCapture')}
            onContinue={() => {
              if (route.params.verified) {
                setVerifiedAt('Just now');
                setActivity((prev) => [
                  {
                    id: String(Date.now()),
                    title: 'Identity Verification',
                    timestamp: 'Just now',
                    status: 'verified',
                  },
                  ...prev,
                ]);
              }
              navigation.navigate('Home');
            }}
          />
        )}
      </Stack.Screen>

      <Stack.Screen name="Home">
        {({ navigation }) => (
          <HomeScreen
            userName="Employee"
            verifiedAt={verifiedAt}
            activity={activity}
            onStartVerification={() => navigation.navigate('DocumentCapture')}
            onViewCertificate={() => {}}
          />
        )}
      </Stack.Screen>
    </Stack.Navigator>
  );
}