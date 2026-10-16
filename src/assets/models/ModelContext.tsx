import React, { createContext, useContext, ReactNode, useMemo, useEffect } from 'react';
import { View, Text, ActivityIndicator, StyleSheet } from 'react-native';
import {
  useTensorflowModel,
  TensorflowModel,
  TensorflowModelDelegate,
} from 'react-native-fast-tflite';

// 1. Static delegates constant declared OUTSIDE the component.
// This fulfills the required 2nd parameter and prevents React re-render reload loops.
const DEFAULT_DELEGATES: TensorflowModelDelegate[] = [];

interface ModelContextType {
  retinaModel: TensorflowModel | null;
  arcfaceModel: TensorflowModel | null;
  isReady: boolean;
}

const ModelContext = createContext<ModelContextType>({
  retinaModel: null,
  arcfaceModel: null,
  isReady: false,
});

export const ModelProvider = ({ children }: { children: ReactNode }) => {
  // Pre-load RetinaFace into memory once
  const retinaPlugin = useTensorflowModel(
    require('./RetinaFace/retinaface_mb025_320.tflite'),
    DEFAULT_DELEGATES
  );

  // Pre-load ArcFace into memory once
  const arcfacePlugin = useTensorflowModel(
    require('./ArcFace/arcface_mobile_112.tflite'),
    DEFAULT_DELEGATES
  );

  const isReady =
    retinaPlugin.state === 'loaded' &&
    arcfacePlugin.state === 'loaded' &&
    retinaPlugin.model != null &&
    arcfacePlugin.model != null;

  // Log tensor shape details to the console once the models load
  useEffect(() => {
    if (isReady && arcfacePlugin.model && retinaPlugin.model) {
      console.log('--- Vision AI Models Loaded ---');
      console.log('RetinaFace Inputs:', retinaPlugin.model.inputs);
      console.log('ArcFace Inputs:', arcfacePlugin.model.inputs);
    }
  }, [isReady, arcfacePlugin.model, retinaPlugin.model]);

  const value = useMemo(
    () => ({
      retinaModel: retinaPlugin.model ?? null,
      arcfaceModel: arcfacePlugin.model ?? null,
      isReady,
    }),
    [retinaPlugin.model, arcfacePlugin.model, isReady]
  );

  // Splash loader: shows only once while reading models from disk into RAM/NPU
  if (!isReady) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#0A84FF" />
        <Text style={styles.loadingTitle}>NeoKYC Vision Engine</Text>
        <Text style={styles.loadingSub}>
          Loading RetinaFace & ArcFace into memory...
        </Text>
      </View>
    );
  }

  return <ModelContext.Provider value={value}>{children}</ModelContext.Provider>;
};

export const useModels = () => {
  const context = useContext(ModelContext);
  if (!context) {
    throw new Error('useModels must be used within a ModelProvider');
  }
  return context;
};

const styles = StyleSheet.create({
  loadingContainer: {
    flex: 1,
    backgroundColor: '#121212',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingTitle: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: '700',
    marginTop: 20,
  },
  loadingSub: {
    color: '#8E8E93',
    fontSize: 13,
    marginTop: 8,
  },
});