/**
 * Contract segmentation — split a raw, multi-paragraph Japanese lease agreement
 * (契約書全文) into individual clauses that `evaluateClause` can score one at a time.
 *
 * WHY THIS IS A SEPARATE STEP, AND WHAT IT DELIBERATELY DOES NOT DO
 * ----------------------------------------------------------------
 * The rule engine reasons about ONE 特約 clause. A pasted contract is many clauses in
 * one blob, so something has to cut it up first. That cutting is pure text handling —
 * it makes no legal judgment, assigns no verdict, and never decides that a clause is
 * good or bad. It only answers "where does one clause end and the next begin", so that
 * a wrong guess here can at worst mis-group text; it can never manufacture a finding
 * against a landlord. The legal caution lives entirely in `rules.ts`.
 *
 * The splitter tries the most reliable boundary it can find and stops at the first
 * layer that actually divides the text:
 *   1. Article headers   — 第○条 / 第○条の○ at the start of a line. The strongest
 *      signal, and the one real leases use. Mid-sentence cross-references like
 *      「本契約第3条に定める」 are NOT boundaries, so a header only counts at line start.
 *   2. Item markers      — 1. / （1） / ① / 一、 / ・ / ○ at the start of a line.
 *   3. Blank-line paragraphs.
 *   4. Sentence groups    — 。-terminated runs, as a last resort.
 *
 * Nothing here is tuned against the golden set; it is mechanical and unit-tested in
 * tests/unit/segment.test.ts.
 */

export interface ClauseSegment {
  /** 1-based position in the contract, in document order. */
  index: number;
  /** Article/section label when one was detected (e.g. "第12条", "第8条の2", "①"); else null. */
  label: string | null;
  /** Clause text, trimmed and length-bounded, ready to feed `evaluateClause`. */
  text: string;
}

export interface SegmentOptions {
  /** Hard cap on segments returned, so a pathological paste cannot fan out without bound. Default 200. */
  maxSegments?: number;
  /**
   * Max characters per segment. Longer runs are split on sentence (。) boundaries.
   * Defaults to 4000 — the same cap `evaluateRequestSchema` enforces on clause_text,
   * so every segment this returns is guaranteed to be acceptable to the engine.
   */
  maxCharsPerSegment?: number;
  /** Fragments shorter than this (ignoring whitespace) are dropped as noise. Default 8. */
  minChars?: number;
}

const DEFAULTS = { maxSegments: 200, maxCharsPerSegment: 4000, minChars: 8 } as const;

/** 第○条 / 第○条の○ — arabic, full-width or kanji numerals. */
const ARTICLE_NUM = "[0-9０-９一二三四五六七八九十百千]+";
const ARTICLE_HEADER = new RegExp(`第\\s*${ARTICLE_NUM}\\s*条(?:の\\s*${ARTICLE_NUM})?`);
/** Same, but anchored to the start of a line (optionally behind spaces / an opening bracket). */
const ARTICLE_AT_LINE_START = new RegExp(`(?:^|\\n)[\\s　]*[「『（(]?(${ARTICLE_HEADER.source})`, "g");

/** A line that opens a numbered or bulleted item. */
const ITEM_MARKER = new RegExp(
  "^[\\s　]*(" +
    [
      `第${ARTICLE_NUM}項`, // 第○項
      "[0-9０-９]+\\s*[.．、)）]", // 1.  １．  2)  ３）
      "[（(]\\s*[0-9０-９]+\\s*[)）]", // (1) （２）
      "[①-⑳]", // circled numbers
      "[一二三四五六七八九十]+\\s*[、.．)）]", // 一、 二．
      "[ア-ンｱ-ﾝ]\\s*[、.．)）]", // katakana enumerators ア、 イ．
      "[・○●◯◆▪▶►\\-－―]", // bullets
    ].join("|") +
    ")",
);

function toHalfWidthDigits(s: string): string {
  return s.replace(/[０-９]/g, (d) => String.fromCharCode(d.charCodeAt(0) - 0xff10 + 0x30));
}

/** Collapse a matched header to a canonical label: strip internal whitespace. */
function canonicalLabel(raw: string): string {
  return raw.replace(/[\s　]+/g, "");
}

function meaningfulLength(text: string): number {
  return text.replace(/[\s　]/g, "").length;
}

/**
 * Split a run that exceeds `maxChars` into pieces at sentence (。) boundaries, keeping
 * each piece under the cap. A single sentence longer than the cap is hard-truncated —
 * the engine would reject it otherwise, and a clause that long is already pathological.
 */
function enforceMaxChars(text: string, maxChars: number): string[] {
  if (text.length <= maxChars) return [text];
  const sentences = text.split(/(?<=。)/); // keep the delimiter
  const out: string[] = [];
  let buf = "";
  for (const s of sentences) {
    if (s.length > maxChars) {
      if (buf) { out.push(buf); buf = ""; }
      for (let i = 0; i < s.length; i += maxChars) out.push(s.slice(i, i + maxChars));
      continue;
    }
    if (buf.length + s.length > maxChars) { out.push(buf); buf = s; }
    else buf += s;
  }
  if (buf) out.push(buf);
  return out;
}

interface RawSegment { label: string | null; text: string; }

/** Layer 1: split on 第○条 headers at line start. */
function splitByArticle(text: string): RawSegment[] | null {
  const boundaries: { start: number; label: string }[] = [];
  for (const m of text.matchAll(ARTICLE_AT_LINE_START)) {
    // m.index points at the (possible) leading newline; the header itself is group 1.
    const headerIdx = m.index + m[0].length - m[1].length;
    boundaries.push({ start: headerIdx, label: canonicalLabel(m[1]) });
  }
  if (boundaries.length < 2) return null;
  const segs: RawSegment[] = [];
  for (let i = 0; i < boundaries.length; i++) {
    const from = boundaries[i].start;
    const to = i + 1 < boundaries.length ? boundaries[i + 1].start : text.length;
    segs.push({ label: boundaries[i].label, text: text.slice(from, to).trim() });
  }
  return segs;
}

/** Layer 2: group lines under item markers (1. / （1） / ① / ・ ...). */
function splitByItemMarker(text: string): RawSegment[] | null {
  const lines = text.split("\n");
  const segs: RawSegment[] = [];
  let current: { label: string | null; lines: string[] } | null = null;
  for (const line of lines) {
    const m = line.match(ITEM_MARKER);
    if (m) {
      if (current) segs.push({ label: current.label, text: current.lines.join("\n").trim() });
      current = { label: m[1].trim(), lines: [line] };
    } else if (current) {
      current.lines.push(line);
    } else if (line.trim().length > 0) {
      // Leading text before the first marker becomes its own labelless segment.
      current = { label: null, lines: [line] };
    }
  }
  if (current) segs.push({ label: current.label, text: current.lines.join("\n").trim() });
  return segs.length >= 2 ? segs : null;
}

/** Layer 3: blank-line-separated paragraphs. */
function splitByParagraph(text: string): RawSegment[] | null {
  const paras = text.split(/\n[\s　]*\n/).map((p) => p.trim()).filter((p) => p.length > 0);
  return paras.length >= 2 ? paras.map((p) => ({ label: null, text: p })) : null;
}

/** Layer 4: sentence groups — always yields something for non-empty text. */
function splitBySentence(text: string, maxChars: number): RawSegment[] {
  const sentences = text.split(/(?<=。)/).map((s) => s.trim()).filter((s) => s.length > 0);
  if (sentences.length === 0) return [{ label: null, text: text.trim() }];
  const segs: RawSegment[] = [];
  let buf = "";
  for (const s of sentences) {
    if (buf && buf.length + s.length > maxChars) { segs.push({ label: null, text: buf }); buf = s; }
    else buf = buf ? `${buf}${s}` : s;
  }
  if (buf) segs.push({ label: null, text: buf });
  return segs;
}

/**
 * Segment a raw contract into clauses. Always returns at least one segment for any
 * text with meaningful content; returns [] for empty or whitespace-only input.
 */
export function segmentContract(raw: string, opts: SegmentOptions = {}): ClauseSegment[] {
  const maxSegments = opts.maxSegments ?? DEFAULTS.maxSegments;
  const maxChars = opts.maxCharsPerSegment ?? DEFAULTS.maxCharsPerSegment;
  const minChars = opts.minChars ?? DEFAULTS.minChars;

  const text = raw.replace(/\r\n?/g, "\n").trim();
  if (meaningfulLength(text) === 0) return [];

  const rawSegs =
    splitByArticle(text) ??
    splitByItemMarker(text) ??
    splitByParagraph(text) ??
    splitBySentence(text, maxChars);

  const out: ClauseSegment[] = [];
  for (const seg of rawSegs) {
    const trimmed = seg.text.trim();
    if (meaningfulLength(trimmed) < minChars) continue;
    for (const piece of enforceMaxChars(trimmed, maxChars)) {
      if (meaningfulLength(piece) < minChars) continue;
      out.push({ index: out.length + 1, label: seg.label, text: piece });
      if (out.length >= maxSegments) return out;
    }
  }
  // If filtering removed everything (e.g. one very short clause), fall back to the
  // whole text as a single segment rather than returning nothing.
  if (out.length === 0) return [{ index: 1, label: null, text }];
  return out;
}

/**
 * Best-effort extraction of the largest yen figure stated in a clause. Used only to
 * surface a rough, clearly-labelled "amount named in this clause" in the batch
 * summary — it is NOT a computed liability and never feeds the verdict. Returns null
 * when the clause names no figure. Handles full-width digits, thousands separators,
 * and a trailing 万 (× 10,000).
 */
export function extractStatedAmountJpy(text: string): number | null {
  const re = /([0-9０-９][0-9０-９,，]*)\s*(万)?\s*円/g;
  let max: number | null = null;
  for (const m of text.matchAll(re)) {
    const digits = toHalfWidthDigits(m[1]).replace(/[,，]/g, "");
    if (!/^\d+$/.test(digits)) continue;
    let value = Number.parseInt(digits, 10);
    if (m[2]) value *= 10000; // 「○○万円」
    if (Number.isFinite(value)) max = max === null ? value : Math.max(max, value);
  }
  return max;
}
