"use client";

/**
 * V10 batch audit — paste an entire lease agreement (契約書全文), press 診断, and read a
 * Red/Yellow/Green breakdown of every clause with a drill-down into each one.
 *
 * This screen presents the SAME provisional engine output as the single-clause flow,
 * only many clauses at once. It leans on two disciplines the engine hands it:
 *   - every clause carries `reviewRequired`, the unknown prongs and the missing facts,
 *     so a red/green tier is never shown without the caveats that qualify it;
 *   - the batch `disclaimer` and `advisory` are shown prominently and unconditionally.
 * Nothing here is a conclusion a tenant should act on without a human.
 */

import { useCallback, useMemo, useState } from "react";
import {
  EvaluateError,
  batchEvaluateContract,
  type BatchClauseResult,
  type BatchEvaluation,
  type ClauseEvaluation,
  type HealthTier,
  type Placement,
} from "@/lib/shared/taikyo-client.ts";
import styles from "./batch.module.css";

const PRONG_LABELS = {
  P1: "P1 明確性",
  P2: "P2 所在",
  P3: "P3 相当性",
  P4: "P4 621条",
} as const;

const TIERS: Record<HealthTier, { labelJa: string; blurbJa: string }> = {
  red: { labelJa: "無効の可能性", blurbJa: "特約自体が民法621条の原則を覆せず、費用が賃貸人負担に戻る可能性が高い条項。" },
  yellow: { labelJa: "要確認・一部無効", blurbJa: "減額・一部無効の可能性、または判断に情報が不足している条項。人による確認が必要。" },
  green: { labelJa: "有効の可能性", blurbJa: "記載どおり有効に成立している可能性が高く、争点にならない条項。" },
};
const TIER_ORDER: readonly HealthTier[] = ["red", "yellow", "green"];

/** Placement applies to the whole pasted document. */
const PLACEMENT_OPTIONS: { value: Placement; labelJa: string }[] = [
  { value: "lease_body", labelJa: "賃貸借契約書の本体" },
  { value: "signed_rider", labelJa: "個別に署名した特約書面" },
  { value: "explanatory_document", labelJa: "重要事項説明書" },
  { value: "house_rules", labelJa: "入居のしおり・管理規約など" },
  { value: "unknown", labelJa: "わからない" },
];

const YEN = new Intl.NumberFormat("ja-JP");

function prongCell(value: boolean | "unknown") {
  if (value === true) return <span className={styles.yes}>満たす</span>;
  if (value === false) return <span className={styles.no}>満たさない</span>;
  return <span className={styles.unk}>不明</span>;
}

function ClauseCard({ clause }: { clause: BatchClauseResult }) {
  const [open, setOpen] = useState(false);
  const ev: ClauseEvaluation = clause.evaluation;
  return (
    <div className={`${styles.clause} ${styles[`edge_${clause.health}`]}`}>
      <button className={styles.clauseHead} onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <span className={`${styles.dot} ${styles[`dot_${clause.health}`]}`} aria-hidden />
        <span className={styles.clauseLabel}>{clause.label ?? `条項 ${clause.index}`}</span>
        <span className={styles.clauseVerdict}>{ev.remedy.labelJa}</span>
        {clause.stated_amount_jpy !== null && (
          <span className={styles.clauseAmount}>¥{YEN.format(clause.stated_amount_jpy)}</span>
        )}
        <span className={styles.chevron}>{open ? "−" : "+"}</span>
      </button>
      <p className={styles.clausePreview}>{clause.clause_text}</p>

      {open && (
        <div className={styles.drill}>
          <p className={styles.remedy}>{ev.remedy.tenantMessageJa}</p>
          <p className={styles.remedyEn}>{ev.remedy.tenantMessageEn}</p>
          <p className={styles.meta}>
            分類：{ev.code ?? "判定不能"}
            {ev.code && !ev.classification.confident && "（確信度が低いため要確認）"}
          </p>

          <table className={styles.table}>
            <thead>
              <tr><th>要件</th><th>判定</th><th>判定理由（開発用・英語）</th></tr>
            </thead>
            <tbody>
              {(["P1", "P2", "P3", "P4"] as const).map((p) => (
                <tr key={p}>
                  <th scope="row">{PRONG_LABELS[p]}</th>
                  <td>{prongCell(ev.prongs[p])}</td>
                  <td>{ev.reasons[p]}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {ev.band && ev.band.measured !== null && (
            <p className={styles.meta}>
              相当性バンド：測定値 {ev.band.measured.toFixed(2)} ／ 目安 {ev.band.supportedMax} 以下・
              上限 {ev.band.elevatedMax} → {ev.band.level}
            </p>
          )}

          {ev.missingFacts.length > 0 && (
            <div className={styles.remediation}>
              <h4 className={styles.remediationTitle}>次に確認すること（この確認で判定が確定します）</h4>
              <ol>
                {ev.missingFacts.map((f) => (
                  <li key={f.id}>
                    {f.questionJa}
                    <span className={styles.remediationWhy}>
                      （{f.unblocks.map((p) => PRONG_LABELS[p]).join("・")}）
                    </span>
                    {f.helpJa && <p className={styles.remediationHelp}>{f.helpJa}</p>}
                  </li>
                ))}
              </ol>
            </div>
          )}

          {ev.authorities.length > 0 && (
            <p className={styles.meta}>根拠：{ev.authorities.join(" ／ ")}</p>
          )}
        </div>
      )}
    </div>
  );
}

export default function BatchAuditView() {
  const [contractText, setContractText] = useState("");
  const [placement, setPlacement] = useState<Placement>("lease_body");
  const [result, setResult] = useState<BatchEvaluation | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<HealthTier | "all">("all");

  const analyze = useCallback(async () => {
    const text = contractText.trim();
    if (text.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      const evaluation = await batchEvaluateContract({ contract_text: text, placement });
      setResult(evaluation);
      setFilter("all");
    } catch (e) {
      setError(e instanceof EvaluateError ? e.message : "診断に失敗しました。時間をおいて再度お試しください。");
    } finally {
      setBusy(false);
    }
  }, [contractText, placement]);

  const shown = useMemo(
    () => (result ? result.clauses.filter((c) => filter === "all" || c.health === filter) : []),
    [result, filter],
  );

  const s = result?.summary;
  const validCount = s?.by_health.green ?? 0;
  const flaggedCount = s ? s.total_clauses - validCount : 0;

  return (
    <div className={styles.shell}>
      <h1 className={styles.h1}>原状回復 特約 一括診断</h1>
      <p className={styles.sub}>
        賃貸借契約書の全文を貼り付けると、条項ごとに分割し、民法621条および国土交通省ガイドラインに照らして
        一括で判定します。契約全体の Red / Yellow / Green の内訳と、各争点の詳細を確認できます。
      </p>

      <div className={styles.card}>
        <label className={styles.label} htmlFor="contract">契約書の全文を貼り付けてください</label>
        <textarea
          id="contract"
          className={styles.textarea}
          value={contractText}
          onChange={(e) => setContractText(e.target.value)}
          placeholder={"例：\n第1条（目的）…\n第2条（賃料）…\n第3条（原状回復）賃借人は、退去時のハウスクリーニング費用として金30,000円を負担する。…"}
        />
        <div className={styles.controls}>
          <label className={styles.inlineLabel}>
            この書面の種類
            <select
              className={styles.select}
              value={placement}
              onChange={(e) => setPlacement(e.target.value as Placement)}
            >
              {PLACEMENT_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.labelJa}</option>
              ))}
            </select>
          </label>
          <button className={styles.btn} disabled={contractText.trim().length === 0 || busy} onClick={() => void analyze()}>
            {busy ? "診断中…" : "契約書を診断する"}
          </button>
        </div>
        <p className={styles.help}>
          書面の種類は全条項の「所在（P2）」の判定に使われます。契約書本体か個別の署名特約であれば契約上の合意として扱われ、
          しおり・管理規約のみに記載された条項は契約上の負担とはみなされません。
        </p>
        {error && <p className={styles.error}>{error}</p>}
      </div>

      {result && s && (
        <>
          <div className={styles.disclaimer}>{result.disclaimer}</div>

          <div className={styles.card}>
            <h2 className={styles.sectionTitle}>診断サマリー</h2>
            <div className={styles.statRow}>
              <div className={styles.stat}>
                <span className={styles.statNum}>{s.total_clauses}</span>
                <span className={styles.statLabel}>抽出された条項</span>
              </div>
              <div className={styles.stat}>
                <span className={`${styles.statNum} ${styles.tGreen}`}>{validCount}</span>
                <span className={styles.statLabel}>有効の可能性</span>
              </div>
              <div className={styles.stat}>
                <span className={`${styles.statNum} ${styles.tRed}`}>{flaggedCount}</span>
                <span className={styles.statLabel}>争点あり（要確認・無効）</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statNum}>¥{YEN.format(s.flagged_amount_jpy)}</span>
                <span className={styles.statLabel}>争点条項に記載の金額</span>
              </div>
            </div>

            <div className={styles.gauge} role="img" aria-label={`赤 ${s.by_health.red}件、黄 ${s.by_health.yellow}件、緑 ${s.by_health.green}件`}>
              {TIER_ORDER.map((t) =>
                s.by_health[t] > 0 ? (
                  <div
                    key={t}
                    className={`${styles.gaugeSeg} ${styles[`seg_${t}`]}`}
                    style={{ flexGrow: s.by_health[t] }}
                  >
                    {s.by_health[t]}
                  </div>
                ) : null,
              )}
            </div>

            <div className={styles.legend}>
              {TIER_ORDER.map((t) => (
                <button
                  key={t}
                  className={`${styles.legendItem} ${filter === t ? styles.legendOn : ""}`}
                  onClick={() => setFilter((f) => (f === t ? "all" : t))}
                >
                  <span className={`${styles.dot} ${styles[`dot_${t}`]}`} aria-hidden />
                  <span className={styles.legendLabel}>{TIERS[t].labelJa}</span>
                  <span className={styles.legendCount}>{s.by_health[t]}</span>
                  <span className={styles.legendBlurb}>{TIERS[t].blurbJa}</span>
                </button>
              ))}
            </div>
            <p className={styles.help}>{s.flagged_amount_basis}</p>
          </div>

          <div className={styles.card}>
            <div className={styles.listHead}>
              <h2 className={styles.sectionTitle}>条項ごとの判定</h2>
              <div className={styles.filters}>
                <button className={`${styles.filterBtn} ${filter === "all" ? styles.filterOn : ""}`} onClick={() => setFilter("all")}>
                  すべて（{s.total_clauses}）
                </button>
                {TIER_ORDER.map((t) => (
                  <button
                    key={t}
                    className={`${styles.filterBtn} ${filter === t ? styles.filterOn : ""}`}
                    onClick={() => setFilter(t)}
                  >
                    {TIERS[t].labelJa}（{s.by_health[t]}）
                  </button>
                ))}
              </div>
            </div>

            {shown.length === 0 ? (
              <p className={styles.help}>該当する条項はありません。</p>
            ) : (
              shown.map((c) => <ClauseCard key={c.index} clause={c} />)
            )}
          </div>

          <p className={styles.advisory}>{result.advisory}</p>
        </>
      )}
    </div>
  );
}
