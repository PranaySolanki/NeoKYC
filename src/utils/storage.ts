import AsyncStorage from '@react-native-async-storage/async-storage';
import { computeCosineSimilarity } from '../screens/RetinaFace+ArcFace/utils/ArcFace/faceMatcher';

export interface UserFaceRecord {
  id: string;
  name: string;
  embedding: number[];
  createdAt: string;
}

const STORAGE_KEY = '@local_registered_faces';

export async function saveFaceRecord(name: string, embedding: Float32Array): Promise<UserFaceRecord> {
  const existingRaw = await AsyncStorage.getItem(STORAGE_KEY);
  const records: UserFaceRecord[] = existingRaw ? JSON.parse(existingRaw) : [];

  const newRecord: UserFaceRecord = {
    id: Date.now().toString(),
    name: name.trim(),
    embedding: Array.from(embedding),
    createdAt: new Date().toISOString(),
  };

  records.push(newRecord);
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(records));
  return newRecord;
}

export async function getAllFaceRecords(): Promise<UserFaceRecord[]> {
  const existingRaw = await AsyncStorage.getItem(STORAGE_KEY);
  return existingRaw ? JSON.parse(existingRaw) : [];
}

export async function clearAllFaceRecords(): Promise<void> {
  await AsyncStorage.removeItem(STORAGE_KEY);
}

/**
 * 1:N Identification with Cosine Similarity and Diagnostic Logs
 */
export async function matchLiveEmbedding(
  liveEmbedding: Float32Array,
  threshold = 0.68
): Promise<{ matchedUser: string | null; similarity: number }> {
  const records = await getAllFaceRecords();
  if (records.length === 0) {
    console.log('No registered records found in storage.');
    return { matchedUser: null, similarity: 0 };
  }

  let bestScore = -1;
  let bestName: string | null = null;

  console.log('--- Biometric Verification Diagnostic ---');
  for (const record of records) {
    const storedVec = new Float32Array(record.embedding);
    const score = computeCosineSimilarity(liveEmbedding, storedVec);

    // === DIAGNOSTIC LOGS ===
    console.log(`Checking against registered user: ${record.name}`);
    console.log('Live embedding snippet:', liveEmbedding.slice(0, 5));
    console.log('Stored embedding snippet:', storedVec.slice(0, 5));
    console.log('Cosine Score:', score);
    console.log('-----------------------------------------');

    if (score > bestScore) {
      bestScore = score;
      bestName = record.name;
    }
  }

  const isMatch = bestScore >= threshold;
  console.log(`Best Candidate: ${bestName} with Score: ${bestScore} (Threshold: ${threshold}) -> Match: ${isMatch}`);

  return {
    matchedUser: isMatch ? bestName : null,
    similarity: Math.max(0, bestScore),
  };
}