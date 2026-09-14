/**
 * Batch contract audit — segment a whole lease, score every clause through
 * `evaluateClause`, and roll the results up into a Red/Yellow/Green summary.
 *
 * This module adds NO new legal reasoning. Each clause is scored by the exact same
 * engine the single-clause endpoint uses, with the same provisional `advisory` and
 * the same `reviewRequired` discipline. The only new thing here is arithmetic over
 * the results (counts, a rough flagged-amount total) and a health tier that is a pure
 * function of the verdict the engine already returned.
 */

import { z } from "zod";
import { PLACEMENTS, evaluateClause, type ClauseEvaluation } from "./rules.ts";
import { VERDICTS, type Verdict } from "./taxonomy.ts";
import { segmentContract, extractStatedAmountJpy } from "./segment.ts";

export const HEALTH_TIERS = ["green", "yellow", "red"] as const;
export type HealthTier = (typeof HEALTH_TIERS)[number];

/**
 * Verdict → traffic light. Deliberately conservative, and keyed off the DECISIVE
 * verdict rather than off `reviewRequired`:
 *
 *   green  — `enforceable`. The clause stands as written; nothing to dispute.
 *   red    — `unenforceable`. The strongest finding: the clause likely cannot stand.
 *   yellow — partial invalidity (`reducible`, `severable`) or `needs_review`.
 *
 * Confidence gate: when the classifier was not confident which pattern it matched, the
 * chosen verdict rests on a shaky premise, so the clause is held at yellow rather than
 * asserted green or red. This is why the batch view never over-claims from a mis-read.
 *
 * Note on `reviewRequired`: `deriveVerdict` only returns a decisive verdict once the
 * prongs that DECIDE it are settled — an "unknown" in a non-decisive prong (e.g. P3 on
 * a clause already void on P4) still leaves `reviewRequired` true but does not soften
 * the verdict. So a red/green tier here always reflects a settled verdict; the unknown
 * prongs, missing facts and provisional advisory travel with each clause for the
 * drill-down. `needs_review` is the engine's own "cannot decide", and maps to yellow.
 */
export function clauseHealth(verdict: Verdict, confident: boolean): HealthTier {
  if (!confident) return "yellow";
  if (verdict === "enforceable") return "green";
  if (verdict === "unenforceable") return "red";
  return "yellow"; // reducible, severable, needs_review
}

export interface BatchClauseResult {
  /** 1-based position in the contract. */
  index: number;
  /** Article/section label when detected (e.g. "第12条"); else null. */
  label: string | null;
  clause_text: string;
  health: HealthTier;
  /** Largest yen figure named in the clause, if any. Provisional; see summary basis. */
  stated_amount_jpy: number | null;
  /** Full engine output, unchanged, so the UI can drill into prongs, remedy and missing facts. */
  evaluation: ClauseEvaluation;
}

export interface BatchSummary {
  total_clauses: number;
  by_health: Record<HealthTier, number>;
  by_verdict: Record<Verdict, number>;
  /** Clauses the engine could not decide outright. */
  review_required: number;
  /**
   * Sum of the largest yen figure named in each non-green clause. A rough indicator of
   * money in play, NOT a computed liability — most clauses carry no figure, and a
   * figure appearing in a clause does not mean the whole sum is recoverable or
   * disputable. Read `flagged_amount_basis` before showing this to anyone.
   */
  flagged_amount_jpy: number;
  flagged_amount_basis: string;
}

export interface BatchEvaluation {
  summary: BatchSummary;
  clauses: BatchClauseResult[];
  /** Same provisional advisory the single-clause engine returns. */
  advisory: string;
  /** Batch-level caution shown once at the top of the dashboard. */
  disclaimer: string;
}

export const batchEvaluateRequestSchema = z.object({
  contract_text: z.string().min(1).max(200_000),
  /**
   * Where this document sits in the paperwork, applied to every clause in it. A pasted
   * 契約書全文 is usually the signed lease body; leaving it "unknown" makes P2 unknown
   * and routes clauses to review rather than guessing they were agreed.
   */
  placement: z.enum(PLACEMENTS).default("unknown"),
});
export type BatchEvaluateRequest = z.infer<typeof batchEvaluateRequestSchema>;
export type BatchEvaluateInput = z.input<typeof batchEvaluateRequestSchema>;

const DISCLAIMER =
  "この一括診断は暫定版です。契約書全文を機械的に条項へ分割し、各条項を単体エンジンで評価したものです。" +
  "各判定は P1・P2・P4 をヒューリスティックで近似しており（算定されるのは P3 のみ）、" +
  "根拠となる判例・ガイドラインの引用は一次資料による検証が未了です（docs/citation-audit-checklist.md 参照）。" +
  "法的助言ではなく、入居者に対する結論として提示できるものではありません。";

export function batchEvaluateContract(raw: BatchEvaluateInput): BatchEvaluation {
  const input = batchEvaluateRequestSchema.parse(raw);
  const segments = segmentContract(input.contract_text);

  const clauses: BatchClauseResult[] = segments.map((seg) => {
    const evaluation = evaluateClause({ clause_text: seg.text, placement: input.placement });
    return {
      index: seg.index,
      label: seg.label,
      clause_text: seg.text,
      health: clauseHealth(evaluation.verdict, evaluation.classification.confident),
      stated_amount_jpy: extractStatedAmountJpy(seg.text),
      evaluation,
    };
  });

  const by_health: Record<HealthTier, number> = { green: 0, yellow: 0, red: 0 };
  const by_verdict = Object.fromEntries(VERDICTS.map((v) => [v, 0])) as Record<Verdict, number>;
  let review_required = 0;
  let flagged_amount_jpy = 0;
  for (const c of clauses) {
    by_health[c.health] += 1;
    by_verdict[c.evaluation.verdict] += 1;
    if (c.evaluation.reviewRequired) review_required += 1;
    if (c.health !== "green" && c.stated_amount_jpy !== null) flagged_amount_jpy += c.stated_amount_jpy;
  }

  const advisory = clauses[0]?.evaluation.advisory ??
    "Provisional output. Not legal advice, and not fit to be shown to a tenant as a conclusion.";

  return {
    summary: {
      total_clauses: clauses.length,
      by_health,
      by_verdict,
      review_required,
      flagged_amount_jpy,
      flagged_amount_basis:
        "赤・黄の条項に記載された最大の金額（円）の単純合計です。実際の負担額・返還請求可能額ではなく、" +
        "金額の記載がない条項は含まれません。争点の規模感を示す暫定的な目安としてのみご覧ください。",
    },
    clauses,
    advisory,
    disclaimer: DISCLAIMER,
  };
}
