/**
 * Unit tests for contract segmentation and batch roll-up.
 *
 * Runs on stock Node (>=22.18) via native type stripping — no test runner:
 *   npm run test:unit
 *
 * These cover the mechanical text handling that feeds the engine (where clause
 * boundaries fall, how a stated amount is read) plus the pure health-tier mapping and
 * an end-to-end batch pass. The engine's own legal behaviour is covered by
 * eval:corpus; nothing here re-litigates a verdict.
 */

import { segmentContract, extractStatedAmountJpy } from "../../lib/modules/taikyo/segment.ts";
import { batchEvaluateContract, clauseHealth } from "../../lib/modules/taikyo/batch.ts";

let failures = 0;
function check(name: string, cond: boolean, detail?: string): void {
  if (cond) {
    console.log(`  ok   ${name}`);
  } else {
    failures += 1;
    console.error(`  FAIL ${name}${detail ? ` — ${detail}` : ""}`);
  }
}
function eq<T>(name: string, actual: T, expected: T): void {
  check(name, Object.is(actual, expected), `expected ${String(expected)}, got ${String(actual)}`);
}

console.log("segmentation");

// 1. Article headers at line start split into one segment per article.
{
  const contract = [
    "第1条（目的）本契約は賃貸借について定める。",
    "第2条（賃料）賃料は月額80,000円とする。",
    "第3条（原状回復）賃借人は、退去時のハウスクリーニング費用として金30,000円を負担する。",
  ].join("\n");
  const segs = segmentContract(contract);
  eq("article: three segments", segs.length, 3);
  eq("article: label 1", segs[0]?.label, "第1条");
  eq("article: label 2", segs[1]?.label, "第2条");
  eq("article: label 3", segs[2]?.label, "第3条");
  check("article: text carried", segs[2]?.text.includes("ハウスクリーニング") ?? false);
}

// 2. 第○条の○ headers and kanji/full-width numerals are recognised.
{
  const contract = "第8条の2　特約\n本文その一。\n第九条　次条\n本文その二。";
  const segs = segmentContract(contract);
  eq("article-no: two segments", segs.length, 2);
  eq("article-no: の-label", segs[0]?.label, "第8条の2");
  eq("article-no: kanji label", segs[1]?.label, "第九条");
}

// 3. A mid-sentence cross-reference is NOT a boundary (header only counts at line start).
{
  const contract =
    "第1条　本契約第2条に定める賃料の支払を怠ったときは、賃貸人は本契約を解除できる。\n" +
    "第2条　賃料は月額とする。";
  const segs = segmentContract(contract);
  eq("cross-ref: two segments (not three)", segs.length, 2);
  check("cross-ref: reference stays inside clause 1", segs[0]?.text.includes("第2条に定める") ?? false);
}

// 4. Item markers drive the fallback when there are no article headers.
{
  const contract = "①室内の清掃を行うこと。\n②鍵を返却すること。\n③郵便物の転送手続を行うこと。";
  const segs = segmentContract(contract);
  eq("items: three segments", segs.length, 3);
  eq("items: label", segs[0]?.label, "①");
}

// 5. Blank-line paragraphs are the next fallback.
{
  const contract = "退去時のクリーニング費用は賃借人の負担とする。\n\n鍵交換費用は賃借人の負担とする。";
  const segs = segmentContract(contract);
  eq("paragraphs: two segments", segs.length, 2);
}

// 6. Empty / whitespace-only input yields no segments.
{
  eq("empty: no segments", segmentContract("   \n 　 \n").length, 0);
}

// 7. A segment longer than the char cap is split on sentence boundaries.
{
  const long = "あ。".repeat(20);
  const segs = segmentContract(long, { maxCharsPerSegment: 10 });
  check("maxChars: split into multiple", segs.length > 1, `got ${segs.length}`);
  check("maxChars: each within cap", segs.every((s) => s.text.length <= 10));
}

console.log("stated-amount extraction");
eq("amount: 金30,000円", extractStatedAmountJpy("費用として金30,000円を負担する"), 30000);
eq("amount: full-width", extractStatedAmountJpy("１０，０００円"), 10000);
eq("amount: 万円", extractStatedAmountJpy("違約金として2万円"), 20000);
eq("amount: picks the largest", extractStatedAmountJpy("5,500円または280,000円"), 280000);
eq("amount: none", extractStatedAmountJpy("賃借人の負担とする"), null);

console.log("health tiers");
eq("health: enforceable→green", clauseHealth("enforceable", true), "green");
eq("health: unenforceable→red", clauseHealth("unenforceable", true), "red");
eq("health: reducible→yellow", clauseHealth("reducible", true), "yellow");
eq("health: severable→yellow", clauseHealth("severable", true), "yellow");
eq("health: needs_review→yellow", clauseHealth("needs_review", true), "yellow");
eq("health: not confident forces yellow", clauseHealth("unenforceable", false), "yellow");

console.log("batch roll-up");
{
  const contract = [
    "第1条　賃借人は、退去時のハウスクリーニング費用として金30,000円を負担するものとする。",
    "第2条　退去時、賃借人はクロスを全面張替えする費用を、毀損の有無にかかわらず負担する。",
    "第3条　賃借人は、鍵交換費用として金16,500円を負担する。",
  ].join("\n");
  const out = batchEvaluateContract({ contract_text: contract, placement: "lease_body" });
  eq("batch: three clauses", out.summary.total_clauses, 3);
  const tierSum = out.summary.by_health.green + out.summary.by_health.yellow + out.summary.by_health.red;
  eq("batch: tiers sum to total", tierSum, out.summary.total_clauses);
  const verdictSum = Object.values(out.summary.by_verdict).reduce((a, b) => a + b, 0);
  eq("batch: verdicts sum to total", verdictSum, out.summary.total_clauses);
  check("batch: advisory present", out.advisory.length > 0);
  check("batch: disclaimer present", out.disclaimer.length > 0);
  check("batch: every clause carries an evaluation", out.clauses.every((c) => c.evaluation.verdict.length > 0));
  check(
    "batch: flagged amount only counts non-green clauses with a figure",
    out.summary.flagged_amount_jpy >= 0,
  );
}

if (failures > 0) {
  console.error(`\n${failures} check(s) failed`);
  process.exitCode = 1;
} else {
  console.log("\nOK");
}
