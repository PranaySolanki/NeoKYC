/**
 * Computes exact Cosine Similarity bounded strictly between -1.0 and 1.0
 */
export function computeCosineSimilarity(embA: Float32Array, embB: Float32Array): number {
  if (embA.length !== 512 || embB.length !== 512) {
    throw new Error(`Embedding size mismatch: ${embA.length} vs ${embB.length}`);
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < 512; i++) {
    const a = embA[i];
    const b = embB[i];
    dotProduct += a * b;
    normA += a * a;
    normB += b * b;
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) return 0;

  const similarity = dotProduct / (normA * normB);
  // Clamp between -1.0 and 1.0 to prevent floating-point overflows
  return Math.max(-1.0, Math.min(1.0, similarity));
}

/**
 * ArcFace mobile threshold:
 * 0.60 (60%) is standard for mobile cameras to reject different age/gender family members.
 */
export function isSamePerson(embA: Float32Array, embB: Float32Array, threshold = 0.60): boolean {
  return computeCosineSimilarity(embA, embB) >= threshold;
}