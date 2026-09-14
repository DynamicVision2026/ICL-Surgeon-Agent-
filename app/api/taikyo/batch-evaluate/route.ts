/**
 * POST /api/taikyo/batch-evaluate — audit an entire lease agreement (契約書全文).
 *
 * Request body (zod-validated, see batchEvaluateRequestSchema):
 *   { contract_text, placement? }
 *
 * The whole text is segmented into individual clauses and each is scored by the same
 * engine as /api/taikyo/evaluate. The response carries a Red/Yellow/Green summary plus
 * the full per-clause evaluation, and — like the single-clause endpoint — every result
 * is provisional by construction: the batch `advisory` and `disclaimer` must be
 * surfaced, and clauses the engine could not decide come back as review_required rather
 * than as findings against a landlord.
 */

import { NextResponse } from "next/server";
import { batchEvaluateContract, batchEvaluateRequestSchema } from "@/lib/modules/taikyo/batch.ts";

export const runtime = "nodejs";

export async function POST(request: Request): Promise<NextResponse> {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "invalid_json", message: "Request body must be JSON." }, { status: 400 });
  }

  const parsed = batchEvaluateRequestSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      {
        error: "invalid_request",
        issues: parsed.error.issues.map((i) => ({ path: i.path.join("."), message: i.message })),
      },
      { status: 400 },
    );
  }

  return NextResponse.json({ result: batchEvaluateContract(parsed.data) }, { status: 200 });
}

export async function GET(): Promise<NextResponse> {
  return NextResponse.json(
    { error: "method_not_allowed", message: "Use POST with a contract_text body." },
    { status: 405 },
  );
}
