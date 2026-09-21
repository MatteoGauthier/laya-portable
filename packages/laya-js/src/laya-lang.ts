// Dependency-free language/script detection for routing between Laya checkpoints.
// Faithful port of upstream/laya/laya/lang.py: script is the primary signal,
// Latin language guess is best-effort stopword/diacritic heuristic.
//
// Routing only needs one decision: is this English Latin text, or something
// the English checkpoint cannot read? Pass explicit model/lang when known.
export interface ScriptProfile {
  [script: string]: number;
}

export interface AnalyseResult {
  script: string;
  script_profile: ScriptProfile;
  language: string | null;
  is_english: boolean;
  non_latin_fraction: number;
}

// Unicode blocks the English (ModernBERT-large, 50k BPE) checkpoint cannot read.
// Mirrors _SCRIPT_RANGES in lang.py.
const SCRIPT_RANGES: ReadonlyArray<readonly [string, ReadonlyArray<readonly [number, number]>]> = [
  [
    'greek',
    [
      [0x0370, 0x03ff],
      [0x1f00, 0x1fff],
    ],
  ],
  [
    'cyrillic',
    [
      [0x0400, 0x052f],
      [0x2de0, 0x2dff],
      [0xa640, 0xa69f],
    ],
  ],
  ['hebrew', [[0x0590, 0x05ff]]],
  [
    'arabic',
    [
      [0x0600, 0x06ff],
      [0x0750, 0x077f],
      [0x08a0, 0x08ff],
      [0xfb50, 0xfdff],
      [0xfe70, 0xfeff],
    ],
  ],
  [
    'devanagari',
    [
      [0x0900, 0x097f],
      [0xa8e0, 0xa8ff],
    ],
  ],
  ['bengali', [[0x0980, 0x09ff]]],
  ['gurmukhi', [[0x0a00, 0x0a7f]]],
  ['gujarati', [[0x0a80, 0x0aff]]],
  ['oriya', [[0x0b00, 0x0b7f]]],
  ['tamil', [[0x0b80, 0x0bff]]],
  ['telugu', [[0x0c00, 0x0c7f]]],
  ['kannada', [[0x0c80, 0x0cff]]],
  ['malayalam', [[0x0d00, 0x0d7f]]],
  ['sinhala', [[0x0d80, 0x0dff]]],
  ['thai', [[0x0e00, 0x0e7f]]],
  ['lao', [[0x0e80, 0x0eff]]],
  ['tibetan', [[0x0f00, 0x0fff]]],
  ['myanmar', [[0x1000, 0x109f]]],
  ['georgian', [[0x10a0, 0x10ff]]],
  ['ethiopic', [[0x1200, 0x137f]]],
  ['khmer', [[0x1780, 0x17ff]]],
  [
    'hangul',
    [
      [0x1100, 0x11ff],
      [0x3130, 0x318f],
      [0xac00, 0xd7af],
    ],
  ],
  [
    'kana',
    [
      [0x3040, 0x309f],
      [0x30a0, 0x30ff],
      [0x31f0, 0x31ff],
    ],
  ],
  [
    'han',
    [
      [0x3400, 0x4dbf],
      [0x4e00, 0x9fff],
      [0xf900, 0xfaff],
    ],
  ],
];

const STOP: Record<string, ReadonlySet<string>> = {
  en: new Set(
    'the and is are was were to of in for with that this it you have has not but on at be as from will can would there their what which please we i'.split(
      ' ',
    ),
  ),
  fr: new Set(
    'le la les des une est pour dans que qui avec sur pas plus nous vous être cette mais sont ont aux ce'.split(' '),
  ),
  de: new Set(
    'der die das und ist ein eine den dem nicht mit für auf von zu sich auch werden wurde haben sind oder aber'.split(
      ' ',
    ),
  ),
  es: new Set('el los las que por con para una es se del como pero son está este esta todo más muy hay sus'.split(' ')),
  pt: new Set('os as que em um uma para com não é se do da dos das mas são está este esta muito pelo pela'.split(' ')),
  it: new Set(
    'il lo gli che di per con non è si del della sono questo questa anche come più sono nella alla'.split(' '),
  ),
  nl: new Set('het een van is op te dat niet met voor zijn aan door maar ook worden deze naar wordt'.split(' ')),
};

const NON_EN_DIACRITICS = new Set('àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœđłşţğıåäö');
const WORD_RE = /[^\W\d_]+/gu;
const LETTER_RE = /\p{L}/u;

function isAlpha(ch: string): boolean {
  return LETTER_RE.test(ch);
}

function scriptOf(cp: number): string | null {
  if (cp < 0x0250 || (0x1e00 <= cp && cp <= 0x1eff)) return 'latin';
  for (const [name, ranges] of SCRIPT_RANGES) {
    for (const [lo, hi] of ranges) {
      if (lo <= cp && cp <= hi) return name;
    }
  }
  return null;
}

/** Collect string leaves of a state (str/dict/list), so detection sees content. Keys ignored. */
export function iterText(state: unknown, depth = 0): string[] {
  if (depth > 6 || state === null || state === undefined) return [];
  if (typeof state === 'string') return [state];
  if (Array.isArray(state)) {
    const out: string[] = [];
    for (const v of state) out.push(...iterText(v, depth + 1));
    return out;
  }
  if (typeof state === 'object') {
    const out: string[] = [];
    for (const v of Object.values(state as Record<string, unknown>)) out.push(...iterText(v, depth + 1));
    return out;
  }
  return [];
}

/** Flatten a state into detection text (keys ignored: usually English). */
export function stateText(state: unknown, maxChars = 4000): string {
  return iterText(state).join(' ').slice(0, maxChars);
}

/** Dominant script: 'latin', 'han', 'devanagari', ... or 'unknown' if no letters. */
export function detectScript(text: string): string {
  const counts = new Map<string, number>();
  for (const ch of text) {
    if (!isAlpha(ch)) continue;
    const cp = ch.codePointAt(0) ?? 0;
    const s = scriptOf(cp);
    if (s === null) continue;
    counts.set(s, (counts.get(s) ?? 0) + 1);
  }
  if (counts.size === 0) return 'unknown';
  let best = 'unknown';
  let bestN = -1;
  for (const [k, v] of counts) {
    if (v > bestN) {
      bestN = v;
      best = k;
    }
  }
  return best;
}

/** Fraction of alphabetic chars per detected script. */
export function scriptProfile(text: string): ScriptProfile {
  const counts = new Map<string, number>();
  for (const ch of text) {
    if (!isAlpha(ch)) continue;
    const cp = ch.codePointAt(0) ?? 0;
    const s = scriptOf(cp);
    if (s === null) continue;
    counts.set(s, (counts.get(s) ?? 0) + 1);
  }
  let total = 0;
  for (const v of counts.values()) total += v;
  if (!total) return {};
  const out: ScriptProfile = {};
  for (const [k, v] of counts) {
    if (v) out[k] = v / total;
  }
  return out;
}

/** Best-effort language code for Latin text, or null when undecided. */
export function guessLatinLanguage(text: string): string | null {
  const words = (text.match(WORD_RE) ?? []).map((w) => w.toLowerCase());
  if (words.length < 4) return null;
  const scores: Record<string, number> = {};
  for (const [lg, sw] of Object.entries(STOP)) {
    let n = 0;
    for (const w of words) if (sw.has(w)) n += 1;
    scores[lg] = n;
  }
  const lowered = text.toLowerCase();
  let diac = 0;
  for (const ch of lowered) if (NON_EN_DIACRITICS.has(ch)) diac += 1;
  const diacRate = diac / Math.max(1, lowered.length);
  const en = scores['en'] ?? 0;
  let bestLg: string | null = null;
  let best = 0;
  for (const [lg, s] of Object.entries(scores)) {
    if (lg === 'en') continue;
    if (s > best) {
      best = s;
      bestLg = lg;
    }
  }
  if (best === 0 && diacRate < 0.02) return en ? 'en' : null;
  if (bestLg && best >= Math.max(2, en + 2)) return bestLg;
  if (diacRate >= 0.04 && bestLg && best >= en) return bestLg;
  return en ? 'en' : null;
}

/** Full detection result for a state. */
export function analyse(state: unknown): AnalyseResult {
  const text = stateText(state);
  const prof = scriptProfile(text);
  const script = detectScript(text);
  const nonLatin = prof && Object.keys(prof).length ? Math.round((1.0 - (prof['latin'] ?? 0.0)) * 10000) / 10000 : 0.0;
  if (script === 'unknown') {
    return { script: 'unknown', script_profile: prof, language: null, is_english: true, non_latin_fraction: 0.0 };
  }
  if (script !== 'latin') {
    return { script, script_profile: prof, language: null, is_english: false, non_latin_fraction: nonLatin };
  }
  const lang = guessLatinLanguage(text);
  return {
    script: 'latin',
    script_profile: prof,
    language: lang,
    is_english: lang === null || lang === 'en',
    non_latin_fraction: nonLatin,
  };
}

/** True when the English checkpoint can be expected to read this state. */
export function isEnglish(state: unknown): boolean {
  return analyse(state).is_english;
}
