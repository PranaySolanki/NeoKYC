// src/utils/spatialParser.ts

export interface OCRWord {
  text: string;
  bbox: { x0: number; y0: number; x1: number; y1: number };
}

export interface StructuredKYCDocument {
  name: string | null;
  dob: string | null;
  gender: string | null;
  idNumber: string | null;
  isIdDetected: boolean;
  cardFaceRegion: { originX: number; originY: number; width: number; height: number } | null;
}

export const parseSpatialDocument = (
  words: OCRWord[],
  imageWidth: number,
  imageHeight: number
): StructuredKYCDocument => {
  const result: StructuredKYCDocument = {
    name: null,
    dob: null,
    gender: null,
    idNumber: null,
    isIdDetected: false,
    cardFaceRegion: null,
  };

  const blacklist = [
    'GOVERNMENT', 'INDIA', 'AUTHORITY', 'UNIQUE', 'IDENTIFICATION',
    'ENROLMENT', 'MALE', 'FEMALE', 'DOB', 'DATE', 'BIRTH', 'CARD',
    'FATHER', 'HUSBAND', 'ADDRESS'
  ];

  // 1. Locate Anchor: Date of Birth Token
  let dobWord: OCRWord | null = null;
  for (const w of words) {
    const text = w.text.replace(/[^a-zA-Z0-9/:\-]/g, '');
    const isDate = /\b\d{2}[/\-]\d{2}[/\-]\d{4}\b/.test(text) || /\b(19|20)\d{2}\b/.test(text);
    const isDobAnchor = /DOB|Birth|Year/i.test(w.text);

    if (isDate || isDobAnchor) {
      dobWord = w;
      const dateMatch = text.match(/\b\d{2}[/\-]\d{2}[/\-]\d{4}\b/) || text.match(/\b(19|20)\d{2}\b/);
      if (dateMatch && !result.dob) {
        result.dob = dateMatch[0];
      }
    }

    // Extract Gender
    if (/^(MALE|FEMALE|TRANSGENDER)$/i.test(text)) {
      result.gender = text.toUpperCase();
    }
  }

  // 2. Spatial Name Extraction: Tokens located directly above DOB
  if (dobWord) {
    const nameCandidates = words.filter((w) => {
      const isAboveDob = w.bbox.y1 <= dobWord!.bbox.y0 && w.bbox.y0 >= dobWord!.bbox.y0 - (imageHeight * 0.25);
      const isRightOfPhoto = w.bbox.x0 >= imageWidth * 0.20;
      const isCapitalized = /^[A-Z][a-z]+$/.test(w.text) || (/^[A-Z\s]+$/.test(w.text) && w.text.length > 2);
      const isNotBlacklisted = !blacklist.some((term) => w.text.toUpperCase().includes(term));

      return isAboveDob && isRightOfPhoto && isCapitalized && isNotBlacklisted;
    });

    if (nameCandidates.length > 0) {
      result.name = nameCandidates.map((c) => c.text).join(' ');
    }
  }

  // 3. 12-Digit Identity Pattern Detection
  const fullText = words.map((w) => w.text).join(' ');
  const idMatch = fullText.match(/\b\d{4}\s\d{4}\s\d{4}\b/) || fullText.match(/\b\d{12}\b/);
  if (idMatch) {
    result.idNumber = idMatch[0];
    result.isIdDetected = true;
  }

  // 4. Estimate ID Card Face Anchor Coordinates (Left-hand quadrant)
  result.cardFaceRegion = {
    originX: Math.floor(imageWidth * 0.05),
    originY: Math.floor(imageHeight * 0.20),
    width: Math.floor(imageWidth * 0.35),
    height: Math.floor(imageHeight * 0.55),
  };

  return result;
};