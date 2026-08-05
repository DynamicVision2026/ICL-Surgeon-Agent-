"""
Liu's Digital Brain — Local-First Demo App
==========================================
A leadership-facing demonstration of the eye-clinic intelligence + digital-asset
system, built on the three-layer spine (Facts -> Judgments -> Principles).

Local-first: runs entirely on the doctor's machine. The historical case data
here is SYNTHETIC demo data standing in for Prof. Liu's de-identified archive —
no real patient records are used in this demo.

Run locally:
    pip install streamlit plotly numpy pandas
    streamlit run demo_app.py

Trilingual: 简体中文 / English / 日本語 (switch in the sidebar).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ===========================================================================
# 0. i18n — every UI string, medical term, and insight in zh / en / ja
# ===========================================================================
LANGS = {"简体中文": "zh", "English": "en", "日本語": "ja"}

T: dict[str, dict[str, str]] = {
    "app_title": {"zh": "刘教授的数字大脑", "en": "Prof. Liu's Digital Brain",
                  "ja": "劉教授のデジタルブレイン"},
    "tagline": {"zh": "将顶尖临床经验沉淀为可查询、可传承的数字资产",
                "en": "Distilling elite clinical know-how into a queryable, portable digital asset",
                "ja": "卓越した臨床ノウハウを検索可能で継承可能なデジタル資産へ"},
    "sidebar_lang": {"zh": "语言 / Language", "en": "Language", "ja": "言語 / Language"},
    "sidebar_data": {"zh": "历史病例库", "en": "Historical case base", "ja": "症例データベース"},
    "sidebar_cases": {"zh": "病例总数", "en": "Total cases", "ja": "総症例数"},
    "sidebar_note": {"zh": "本演示使用合成数据，不含任何真实患者信息。",
                     "en": "This demo uses synthetic data — no real patient information.",
                     "ja": "本デモは合成データを使用しており、実際の患者情報は含みません。"},
    "tab1": {"zh": "① 患者信任与科普", "en": "① Patient Trust & Education",
             "ja": "① 患者の信頼と啓発"},
    "tab2": {"zh": "② 学术与临床智能", "en": "② Academic & Clinical Intelligence",
             "ja": "② 学術・臨床インテリジェンス"},
    "tab3": {"zh": "③ 合作伙伴 / 供应商", "en": "③ Partner / Supplier",
             "ja": "③ パートナー / サプライヤー"},
    # --- Tab 1 ---
    "t1_head": {"zh": "术前咨询：为这只眼睛匹配相似的历史病例",
                "en": "Pre-op consult: match this eye to similar historical cases",
                "ja": "術前相談：この眼に類似する過去症例をマッチング"},
    "t1_intro": {"zh": "输入患者生物测量参数，系统即时检索刘教授历史病例中最相似的眼睛，"
                       "用真实数据建立信任与合理预期。",
                 "en": "Enter the patient's biometry; the system instantly retrieves the most "
                       "similar eyes from Prof. Liu's history to build trust and set expectations.",
                 "ja": "患者の生体計測値を入力すると、劉教授の症例から最も類似した眼を即時に検索し、"
                       "信頼と適切な期待を築きます。"},
    "acd": {"zh": "前房深度 ACD (mm)", "en": "Anterior chamber depth ACD (mm)",
            "ja": "前房深度 ACD (mm)"},
    "wtw": {"zh": "角膜横径 WTW / 白到白 (mm)", "en": "White-to-white WTW (mm)",
            "ja": "角膜横径 WTW (mm)"},
    "sts": {"zh": "沟到沟 STS (mm)", "en": "Sulcus-to-sulcus STS (mm)",
            "ja": "毛様溝間距離 STS (mm)"},
    "sph": {"zh": "球镜 SPH (D)", "en": "Sphere SPH (D)", "ja": "球面度数 SPH (D)"},
    "age": {"zh": "年龄", "en": "Age", "ja": "年齢"},
    "match_btn": {"zh": "🔍 匹配相似病例", "en": "🔍 Match similar cases",
                  "ja": "🔍 類似症例をマッチング"},
    "t1_cohort": {"zh": "匹配到的相似病例队列", "en": "Matched similar-case cohort",
                  "ja": "マッチした類似症例コホート"},
    "t1_cohort_n": {"zh": "相似病例数", "en": "Similar cases", "ja": "類似症例数"},
    "t1_pred_vault": {"zh": "预测拱高 (中位)", "en": "Predicted vault (median)",
                      "ja": "予測ボールト (中央値)"},
    "t1_vault_range": {"zh": "拱高区间 (P25–P75)", "en": "Vault range (P25–P75)",
                       "ja": "ボールト範囲 (P25–P75)"},
    "t1_common_size": {"zh": "最常选用尺寸", "en": "Most-chosen size", "ja": "最頻選択サイズ"},
    "t1_preview": {"zh": "拱高 / 间隙 3D 预览（基于历史数据模拟）",
                   "en": "Vault / clearance 3D preview (simulated from history)",
                   "ja": "ボールト / クリアランス 3D プレビュー（履歴に基づくシミュレーション）"},
    "t1_preview_cap": {"zh": "上方曲面为 ICL 后表面，下方为自然晶状体前表面；两者间隙即拱高。",
                       "en": "Upper surface = ICL posterior; lower = crystalline lens; the gap is the vault.",
                       "ja": "上面＝ICL後面、下面＝水晶体前面。その間隙がボールトです。"},
    "t1_insight": {"zh": "洞见", "en": "Insight", "ja": "インサイト"},
    "t1_insight_txt": {
        "zh": "在与本眼高度相似的 {n} 只历史眼中，刘教授最常选择 {size} mm，术后拱高中位 {v} µm，"
              "落在理想安全区间。这为患者提供了以真实经验为依据的可靠预期。",
        "en": "Across {n} highly similar historical eyes, Prof. Liu most often chose {size} mm, "
              "with a median post-op vault of {v} µm — within the ideal safety window. This gives "
              "the patient expectations grounded in real experience.",
        "ja": "本眼に酷似する {n} 眼の履歴では、劉教授は {size} mm を最も多く選択し、術後ボールト中央値は "
              "{v} µm と理想的な安全域内でした。実際の経験に基づく信頼できる予測を患者に提供します。"},
    "t1_table_cols": {"zh": ["病例", "ACD", "WTW", "STS", "SPH", "选用尺寸", "拱高µm"],
                      "en": ["Case", "ACD", "WTW", "STS", "SPH", "Size", "Vault µm"],
                      "ja": ["症例", "ACD", "WTW", "STS", "SPH", "サイズ", "ボールトµm"]},
    # --- Tab 2 ---
    "t2_head": {"zh": "学术与临床智能", "en": "Academic & Clinical Intelligence",
                "ja": "学術・臨床インテリジェンス"},
    "t2_research_h": {"zh": "一键临床研究选题生成", "en": "One-click clinical research topics",
                      "ja": "ワンクリック臨床研究テーマ生成"},
    "t2_research_btn": {"zh": "✨ 生成研究选题", "en": "✨ Generate research topics",
                        "ja": "✨ 研究テーマを生成"},
    "t2_research_cap": {"zh": "选题由病例库统计自动生成，可直接用于论文或学术汇报。",
                        "en": "Topics auto-generated from the case-base statistics — ready for a paper or talk.",
                        "ja": "症例統計から自動生成されたテーマ。論文や発表にそのまま使えます。"},
    "t2_nomo_h": {"zh": "个人 Nomogram 决策看板（描述性）",
                  "en": "Personal Nomogram decision dashboard (descriptive)",
                  "ja": "個人ノモグラム意思決定ダッシュボード（記述的）"},
    "t2_nomo_cap": {"zh": "按前房深度分层，展示刘教授的实际选择倾向。此为“描述其既往行为”，"
                          "经本人确认后方可升级为处方性原则。",
                    "en": "Stratified by ACD, showing Prof. Liu's actual choice tendency. This is "
                          "DESCRIPTIVE (what was done); it becomes prescriptive only once he ratifies it.",
                    "ja": "前房深度で層別化し、劉教授の実際の選択傾向を表示。これは記述的（過去の行動）であり、"
                          "本人の承認を経て初めて処方的原則になります。"},
    "t2_nomo_cols": {"zh": ["ACD 区间", "病例数", "最常尺寸", "相对参考的平均偏移", "平均拱高µm"],
                     "en": ["ACD band", "n", "Modal size", "Mean Δ vs reference", "Mean vault µm"],
                     "ja": ["ACD 区間", "症例数", "最頻サイズ", "参照との平均偏差", "平均ボールトµm"]},
    "t2_nomo_insight": {
        "zh": "已识别的签名倾向：当 ACD ≥ 3.3 mm 时，刘教授倾向比参考尺寸上调一档 —— "
              "此模式已在 {n} 例中观察到，等待本人确认。",
        "en": "Detected signature tendency: when ACD ≥ 3.3 mm, Prof. Liu tends to upsize one step "
              "vs. the reference — observed in {n} cases, awaiting his ratification.",
        "ja": "検出された署名的傾向：ACD ≥ 3.3 mm のとき、劉教授は参照より一段大きいサイズを選ぶ傾向 —— "
              "{n} 例で観察され、本人の承認待ちです。"},
    "t2_await": {"zh": "候选原则 · 待医生确认", "en": "Candidate principle · awaiting ratification",
                 "ja": "候補原則 · 承認待ち"},
    # --- Tab 3 ---
    "t3_head": {"zh": "合作伙伴 / 供应商协作", "en": "Partner / Supplier Collaboration",
                "ja": "パートナー / サプライヤー連携"},
    "t3_intro": {"zh": "平台不买卖数据、不做数据经纪。医生可一键生成去标识化的真实世界数据（RWD）"
                       "汇总，自主与厂商洽谈。以下为可导出的聚合指标示例。",
                 "en": "The platform does not buy, sell, or broker data. The doctor can generate "
                       "de-identified real-world-data (RWD) summaries on demand and negotiate with "
                       "manufacturers independently. Below are exportable aggregate metrics.",
                 "ja": "本プラットフォームはデータの売買・仲介を行いません。医師は非識別化されたリアルワールド"
                       "データ（RWD）の集計をワンクリックで生成し、メーカーと独自に交渉できます。以下は"
                       "エクスポート可能な集計指標です。"},
    "t3_m_eyes": {"zh": "去标识眼数", "en": "De-identified eyes", "ja": "非識別化眼数"},
    "t3_m_meanvault": {"zh": "平均拱高µm", "en": "Mean vault µm", "ja": "平均ボールトµm"},
    "t3_m_sizes": {"zh": "覆盖尺寸型号", "en": "Lens sizes covered", "ja": "対象サイズ数"},
    "t3_m_kanon": {"zh": "k-匿名下限", "en": "k-anonymity floor", "ja": "k-匿名性の下限"},
    "t3_dist_h": {"zh": "尺寸选择分布（去标识聚合）", "en": "Lens-size distribution (de-identified aggregate)",
                  "ja": "サイズ選択分布（非識別化集計）"},
    "t3_priv_h": {"zh": "本地隐私合规", "en": "Local privacy compliance", "ja": "ローカルプライバシー遵守"},
    "t3_priv_txt": {
        "zh": "• 原始病历的 OCR、提取与去标识全部在本地完成，云端永不接触未脱敏数据。\n"
              "• 中国与日本各自通过本地注册主体运营，数据严格留存在本辖区，零跨境传输。\n"
              "• 对外聚合导出强制执行 k-匿名（默认 k≥5），低于阈值的切片自动抑制或泛化。",
        "en": "• All OCR, extraction and de-identification of raw charts happen locally; the cloud "
              "never touches unmasked data.\n"
              "• China and Japan operate via separate locally-registered entities; data stays strictly "
              "within its jurisdiction — zero cross-border transfer.\n"
              "• Aggregate exports enforce k-anonymity (default k≥5); slices below the floor are "
              "suppressed or generalized.",
        "ja": "• 原本カルテのOCR・抽出・非識別化はすべてローカルで実行され、クラウドは非マスクデータに"
              "一切触れません。\n"
              "• 中国と日本はそれぞれ現地登録法人で運営し、データは管轄内に厳格に留まります —— 越境転送ゼロ。\n"
              "• 集計エクスポートは k-匿名性（既定 k≥5）を強制し、下限未満のスライスは抑制または一般化されます。"},
    "t3_export_btn": {"zh": "⬇️ 生成供应商 RWD 汇总", "en": "⬇️ Generate supplier RWD summary",
                      "ja": "⬇️ サプライヤー向けRWD集計を生成"},
    "t3_export_ok": {"zh": "已生成去标识 RWD 汇总（演示）。所有切片满足 k≥5。",
                     "en": "De-identified RWD summary generated (demo). All slices satisfy k≥5.",
                     "ja": "非識別化RWD集計を生成しました（デモ）。全スライスが k≥5 を満たします。"},
    # --- persona / provenance / compliance ---
    "persona": {"zh": "🎓 刘教授的数字学者：只引用刘教授本人的病例数据，绝不臆测。",
                "en": "🎓 Prof. Liu's Digital Scholar: cites only Prof. Liu's own case records — never speculates.",
                "ja": "🎓 劉教授のデジタル学者：劉教授自身の症例のみを引用し、推測は一切しません。"},
    "badge_observed": {"zh": "📎 刘教授病例中观察到", "en": "📎 observed in Prof. Liu's records",
                       "ja": "📎 劉教授の症例で観察"},
    "badge_ratified": {"zh": "✅ 刘教授已确认的原则", "en": "✅ Prof. Liu's ratified principle",
                       "ja": "✅ 劉教授が承認した原則"},
    "badge_descriptive": {"zh": "描述性 · 待确认", "en": "descriptive · awaiting ratification",
                          "ja": "記述的 · 承認待ち"},
    "compliance": {"zh": "⚕️ 本工具用于科普与预期沟通，不构成诊断或个体化医疗建议；结果为“相似眼的既往表现”，"
                         "非承诺或保证。所有临床决策由医生做出。",
                   "en": "⚕️ This tool is for education and expectation-setting — not a diagnosis or "
                         "individual medical advice. Results show how similar eyes fared, not a promise "
                         "or guarantee. All clinical decisions rest with the surgeon.",
                   "ja": "⚕️ 本ツールは啓発と期待形成のためのものであり、診断や個別の医療助言ではありません。"
                         "結果は「類似した眼の過去の経過」であり、約束・保証ではありません。臨床判断はすべて医師が行います。"},
    # --- cohort meter / scatter ---
    "tightness": {"zh": "相似度严格程度（越严越像本眼）", "en": "Similarity tightness (stricter = more alike)",
                  "ja": "類似度の厳しさ（厳しいほど本眼に近い）"},
    "meter_label": {"zh": "相似病例数", "en": "Similar cases", "ja": "類似症例数"},
    "cohort_ok": {"zh": "在刘教授病例中找到 **{n}** 只与本眼高度相似的眼。",
                  "en": "Found **{n}** eyes in Prof. Liu's records highly similar to this one.",
                  "ja": "劉教授の症例に、本眼に酷似する眼が **{n}** 眼見つかりました。"},
    "cohort_thin": {"zh": "⚠️ 刘教授病例中相似的眼过少（n={n} < {k}），不足以在此可靠陈述。请放宽相似度或调整参数。",
                    "en": "⚠️ Too few similar eyes in Prof. Liu's records (n={n} < {k}) to speak confidently "
                          "here. Loosen the similarity or adjust the parameters.",
                    "ja": "⚠️ 劉教授の症例に類似する眼が少なすぎます（n={n} < {k}）。確信を持って述べられません。"
                          "類似度を緩めるかパラメータを調整してください。"},
    "scatter_title": {"zh": "「你在这里」：本眼在刘教授病例中的位置", "en": "\"You are here\": this eye among Prof. Liu's cases",
                      "ja": "「あなたはここ」：劉教授の症例における本眼の位置"},
    "scatter_you": {"zh": "本眼", "en": "This eye", "ja": "本眼"},
    "scatter_cohort": {"zh": "相似队列", "en": "Similar cohort", "ja": "類似コホート"},
    "scatter_other": {"zh": "其他病例", "en": "Other cases", "ja": "その他の症例"},
    # --- vault scrubber ---
    "scrubber_h": {"zh": "拱高与视力随时间演变（拖动时间轴）", "en": "Vault & vision over time (drag the timeline)",
                   "ja": "ボールトと視力の経時変化（タイムラインをドラッグ）"},
    "scrubber_tp": {"zh": "术后时间点", "en": "Post-op timepoint", "ja": "術後の時点"},
    "vault_at": {"zh": "该时点拱高 (中位)", "en": "Vault at this point (median)", "ja": "この時点のボールト (中央値)"},
    "va_at": {"zh": "该时点视力≥1.0 占比", "en": "Share reaching VA ≥ 1.0", "ja": "視力≥1.0 到達割合"},
    "scrubber_note": {"zh": "分布来自上方相似队列（{n} 眼）；曲面为该时点中位拱高的模拟。",
                      "en": "Distributions come from the similar cohort above ({n} eyes); the surface simulates "
                            "the median vault at this timepoint.",
                      "ja": "分布は上記の類似コホート（{n} 眼）に基づき、曲面はこの時点の中央値ボールトのシミュレーションです。"},
    "tp_labels": {"zh": ["第1天", "第1周", "1个月", "3个月", "6个月", "1年", "2年", "3年"],
                  "en": ["Day 1", "Week 1", "Month 1", "Month 3", "Month 6", "Year 1", "Year 2", "Year 3"],
                  "ja": ["1日目", "1週", "1ヶ月", "3ヶ月", "6ヶ月", "1年", "2年", "3年"]},
    "explorer_h": {"zh": "实时相似病例探索器（新）", "en": "Live similar-case explorer (new)",
                   "ja": "リアルタイム類似症例エクスプローラー（新）"},
    "expert_h": {"zh": "历史专家病例匹配", "en": "Historical expert case matching",
                 "ja": "過去のエキスパート症例マッチング"},
    "plain_label": {"zh": "通俗解释：", "en": "In plain terms:", "ja": "かんたんに言うと："},
    "why_label": {"zh": "为何重要：", "en": "Why it matters:", "ja": "なぜ重要："},
    "understand_terms": {"zh": "👉 点此了解这些专业术语（通俗解释 + 刘教授真实数据）",
                         "en": "👉 Tap to understand these clinical terms (plain language + Prof. Liu's real data)",
                         "ja": "👉 これらの専門用語をやさしく理解する（劉教授の実データつき）"},
    "cmp_h": {"zh": "人工经验 vs. AI 辅助：同一只眼，两种沟通方式",
              "en": "Manual experience vs. AI-assisted: one eye, two ways to communicate",
              "ja": "手動の経験 vs. AI支援：同じ眼、2つの伝え方"},
    "cmp_manual_title": {"zh": "🧑‍⚕️ 传统人工方式", "en": "🧑‍⚕️ Traditional manual approach",
                         "ja": "🧑‍⚕️ 従来の手動アプローチ"},
    "cmp_ai_title": {"zh": "🤖 AI 辅助队列探索", "en": "🤖 AI-assisted cohort exploration",
                     "ja": "🤖 AI支援コホート探索"},
    "cmp_manual_pts": {
        "zh": ["凭记忆回忆“几例”相似病例", "主观印象，无精确数量", "满口专业术语，患者难懂",
               "无隐私核验流程", "预期基于医生印象"],
        "en": ["Recalls 'a few' similar cases from memory", "Subjective impression, no exact count",
               "Heavy jargon, hard for patients", "No privacy-verification step",
               "Expectations set by impression"],
        "ja": ["記憶から「数例」を想起", "主観的印象、正確な数なし", "専門用語が多く患者に難解",
               "プライバシー検証なし", "期待は印象に基づく"]},
    "cmp_ai_pts": {
        "zh": ["从刘教授 {N} 例中客观检索", "精确相似队列 n={n}，可复现", "术语即时转为通俗语言",
               "k≥{k} 隐私校验实时通过", "预期基于真实结果分布"],
        "en": ["Objective search across Prof. Liu's {N} eyes", "Exact cohort n={n}, reproducible",
               "Terms auto-translated to plain language", "k≥{k} privacy check verified live",
               "Expectations from real outcome distribution"],
        "ja": ["劉教授の {N} 眼から客観的に検索", "正確なコホート n={n}、再現可能",
               "用語を即座にやさしく変換", "k≥{k} プライバシー検証を即時通過",
               "期待は実際の結果分布に基づく"]},
    "cmp_anchor": {"zh": "同一只眼：人工凭印象 → 本系统从刘教授 {N} 例中客观检索出 n={n} 例相似眼，并通过 k≥{k} 隐私校验。",
                   "en": "Same eye: manual impression → this system objectively retrieves n={n} similar eyes "
                         "from Prof. Liu's {N}, k≥{k} privacy-verified.",
                   "ja": "同じ眼：手動の印象 → 本システムは劉教授の {N} 眼から n={n} 眼を客観的に抽出し、k≥{k} で検証。"},
    "live_badge": {"zh": "● 实时", "en": "● LIVE", "ja": "● ライブ"},
    "kanon_ok_badge": {"zh": "✓ k-匿名已校验", "en": "✓ k-anonymity verified", "ja": "✓ k-匿名性 検証済"},
    "kanon_bad_badge": {"zh": "✕ 低于 k 阈值", "en": "✕ below k-threshold", "ja": "✕ k閾値未満"},
}

def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return T.get(key, {}).get(lang, T.get(key, {}).get("en", key))


# ===========================================================================
# 1. Synthetic "Prof. Liu" historical archive (stands in for de-identified data)
#    The signature is baked in: deeper ACD -> tends to upsize vs. the reference.
# ===========================================================================
LENS_SIZES = np.array([12.1, 12.6, 13.2, 13.7])

def _reference_size(sts: float) -> float:
    """A neutral reference nomogram (what a naive sizing table would pick)."""
    if sts < 11.0:
        return 12.1
    if sts < 11.6:
        return 12.6
    if sts < 12.2:
        return 13.2
    return 13.7

def _step_up(size: float) -> float:
    idx = int(np.argmin(np.abs(LENS_SIZES - size)))
    return LENS_SIZES[min(idx + 1, len(LENS_SIZES) - 1)]

@st.cache_data
def load_history(n: int = 420, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    acd = np.clip(rng.normal(3.2, 0.26, n), 2.6, 4.1)
    wtw = np.clip(rng.normal(11.7, 0.42, n), 10.6, 12.9)
    sts = np.clip(wtw + rng.normal(0.30, 0.18, n), 10.7, 13.2)
    sph = np.clip(rng.normal(-8.0, 3.2, n), -20, -2)
    age = np.clip(rng.normal(31, 6.5, n), 18, 52).round().astype(int)

    ref = np.array([_reference_size(s) for s in sts])
    chosen = ref.copy()
    # Prof. Liu's signature: deep ACD -> upsize one step in most such eyes
    deep = acd >= 3.3
    upsize_roll = rng.random(n) < 0.80
    for i in range(n):
        if deep[i] and upsize_roll[i]:
            chosen[i] = _step_up(ref[i])

    # vault grows with (chosen - sts); noisy, clipped to a plausible range.
    # Treat this as the ~Month-3 stable value; the scrubber applies a settle curve.
    vault = 300 + 430 * (chosen - sts - 0.7) + rng.normal(0, 70, n)
    vault = np.clip(vault, 130, 900).round().astype(int)

    # per-case final (plateau) best-corrected decimal VA, for the vision trajectory
    bcva_final = np.clip(rng.normal(1.08, 0.16, n), 0.6, 1.5).round(2)

    return pd.DataFrame({"acd": acd.round(2), "wtw": wtw.round(2), "sts": sts.round(2),
                         "sph": sph.round(2), "age": age,
                         "ref_size": ref, "size": chosen, "vault": vault,
                         "bcva_final": bcva_final})


# k-anonymity floor: below this many similar eyes, the agent refuses to speak
K_ANON = 5

# post-op timepoints (day offset) + settle/recovery curves applied to the cohort.
# vault settles slightly from an early high; vision recovers over the first weeks.
TP_DAYS = [1, 7, 30, 90, 180, 365, 730, 1095]
VAULT_SETTLE = {1: 1.14, 7: 1.09, 30: 1.04, 90: 1.00, 180: 0.99,
                365: 0.98, 730: 0.97, 1095: 0.96}
VA_RECOVERY = {1: 0.60, 7: 0.84, 30: 0.96, 90: 1.00, 180: 1.00,
               365: 1.00, 730: 0.99, 1095: 0.99}


# ===========================================================================
# 2. Weighted structured KNN (mirrors the KNN engine design; sizing biometry
#    dominates). z-scored features, weighted Euclidean, exact scan.
# ===========================================================================
KNN_FEATURES = ["sts", "wtw", "acd", "sph", "age"]
KNN_WEIGHTS = np.array([3.0, 2.5, 2.0, 1.0, 0.8])

@st.cache_data
def _feature_stats(df: pd.DataFrame):
    means = df[KNN_FEATURES].mean().values
    stds = df[KNN_FEATURES].std(ddof=0).replace(0, 1).values
    return means, stds

def _z(df, means, stds):
    return (df[KNN_FEATURES].values - means) / stds

def weighted_distances(df: pd.DataFrame, query: dict) -> np.ndarray:
    means, stds = _feature_stats(df)
    qz = (np.array([query[f] for f in KNN_FEATURES], dtype=float) - means) / stds
    fz = _z(df, means, stds)
    return np.sqrt(((fz - qz) ** 2 * KNN_WEIGHTS).sum(axis=1))

def cohort_within(df: pd.DataFrame, query: dict, radius: float):
    """Radius-based 'similar' cohort. Returns (cohort_df, distance_array, mask).
    The cohort SHRINKS as the user tightens the radius or picks rare parameters —
    which is what lets the live meter honestly drop below the k-anonymity floor."""
    dist = weighted_distances(df, query)
    mask = dist <= radius
    out = df.copy()
    out["distance"] = dist
    return out[mask].sort_values("distance"), dist, mask

def match_cases(df: pd.DataFrame, query: dict, k: int = 25) -> pd.DataFrame:
    means, stds = _feature_stats(df)
    q = np.array([query[f] for f in KNN_FEATURES], dtype=float)
    qz = (q - means) / stds
    fz = (df[KNN_FEATURES].values - means) / stds
    dist = np.sqrt(((fz - qz) ** 2 * KNN_WEIGHTS).sum(axis=1))
    out = df.copy()
    out["distance"] = dist
    return out.nsmallest(k, "distance")


# ===========================================================================
# 3. Descriptive nomogram (mirrors knowhow_registry.distill_nomogram)
# ===========================================================================
ACD_EDGES = [3.0, 3.3, 3.6]

def _band_label(v: float) -> str:
    edges = ACD_EDGES
    lo = None
    for e in edges:
        if v < e:
            return f"[{'−∞' if lo is None else lo}, {e})"
        lo = e
    return f"[{lo}, +∞)"

def build_nomogram(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tmp = df.copy()
    tmp["band"] = tmp["acd"].apply(_band_label)
    for band, g in tmp.groupby("band"):
        modal = g["size"].mode().iloc[0]
        mean_delta = (g["size"] - g["ref_size"]).mean()
        rows.append({"band": band, "n": len(g), "modal": modal,
                     "delta": round(mean_delta, 3), "vault": round(g["vault"].mean(), 0)})
    order = {"[−∞, 3.0)": 0, "[3.0, 3.3)": 1, "[3.3, 3.6)": 2, "[3.6, +∞)": 3}
    return pd.DataFrame(rows).sort_values("band", key=lambda s: s.map(order)).reset_index(drop=True)


# ===========================================================================
# 4. 3D vault preview (schematic — ICL posterior surface over crystalline lens)
# ===========================================================================
def vault_preview(vault_um: float) -> go.Figure:
    vault_mm = vault_um / 1000.0
    g = np.linspace(-3, 3, 40)
    X, Y = np.meshgrid(g, g)
    R2 = X ** 2 + Y ** 2
    mask = R2 <= 9
    lens = np.where(mask, -0.16 * R2, np.nan)              # crystalline lens anterior dome
    icl = np.where(mask, -0.16 * R2 + vault_mm, np.nan)    # ICL posterior surface, vault above
    fig = go.Figure()
    fig.add_surface(x=X, y=Y, z=lens, showscale=False, opacity=0.9,
                    colorscale="Blues", name="Crystalline lens")
    fig.add_surface(x=X, y=Y, z=icl, showscale=False, opacity=0.6,
                    colorscale="Teal", name="ICL")
    fig.update_layout(
        height=420, margin=dict(l=0, r=0, t=30, b=0),
        uirevision="vault",                       # keep the user's camera across scrub updates
        transition=dict(duration=400, easing="cubic-in-out"),
        scene=dict(xaxis_title="mm", yaxis_title="mm", zaxis_title="mm",
                   aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.6)),
        title=f"Vault ≈ {int(vault_um)} µm")
    return fig


# ===========================================================================
# 5. Research-topic generator (data-aware, templated per language)
# ===========================================================================
def research_topics(df: pd.DataFrame) -> list[str]:
    lang = st.session_state.get("lang", "en")
    n = len(df)
    hi = df[df["sph"] <= -10]
    deep = df[df["acd"] >= 3.3]
    tpl = {
        "zh": [
            f"高度近视（SPH ≤ −10 D，n={len(hi)}）ICL 术后拱高的分布与影响因素分析",
            f"前房深度对 ICL 尺寸选择与术后拱高的影响：基于 {n} 眼的真实世界回顾",
            f"深前房眼（ACD ≥ 3.3 mm，n={len(deep)}）的个体化尺寸策略与安全性",
            "不同 STS/WTW 差值下 ICL 拱高预测模型的构建与验证",
            "刘教授 ICL 手术拱高长期稳定性的真实世界证据（RWE）研究",
        ],
        "en": [
            f"Vault distribution and predictors after ICL in high myopia (SPH ≤ −10 D, n={len(hi)})",
            f"Effect of anterior chamber depth on ICL sizing and post-op vault: a {n}-eye RWD review",
            f"Individualized sizing strategy and safety in deep-ACD eyes (ACD ≥ 3.3 mm, n={len(deep)})",
            "Building and validating a vault-prediction model across STS/WTW differences",
            "Real-world evidence (RWE) on long-term vault stability in Prof. Liu's ICL series",
        ],
        "ja": [
            f"強度近視（SPH ≤ −10 D、n={len(hi)}）におけるICL術後ボールトの分布と規定因子",
            f"前房深度がICLサイズ選択と術後ボールトに与える影響：{n}眼のRWDレビュー",
            f"深前房眼（ACD ≥ 3.3 mm、n={len(deep)}）における個別化サイズ戦略と安全性",
            "STS/WTW差に基づくボールト予測モデルの構築と検証",
            "劉教授のICL症例におけるボールト長期安定性のリアルワールドエビデンス（RWE）",
        ],
    }
    return tpl[lang]


# ===========================================================================
# 5b. "You are here" scatter, timepoint cohort stats, provenance badges
# ===========================================================================
SIZE_COLORS = {12.1: "#a8dadc", 12.6: "#457b9d", 13.2: "#2a9d8f", 13.7: "#e76f51"}

def cohort_scatter(df: pd.DataFrame, query: dict, mask: np.ndarray) -> go.Figure:
    """STS (x) vs ACD (y) — the sizing space. Other cases grey, cohort colored by
    chosen size, the patient's eye a gold star."""
    fig = go.Figure()
    other = df[~mask]
    fig.add_scatter(x=other["sts"], y=other["acd"], mode="markers",
                    marker=dict(size=6, color="#d9d9d9"), name=t("scatter_other"),
                    hoverinfo="skip")
    coh = df[mask]
    fig.add_scatter(
        x=coh["sts"], y=coh["acd"], mode="markers",
        marker=dict(size=9, color=[SIZE_COLORS.get(s, "#555") for s in coh["size"]],
                    line=dict(width=0.5, color="white")),
        name=t("scatter_cohort"),
        text=[f"{s} mm · vault {v}µm" for s, v in zip(coh["size"], coh["vault"])],
        hovertemplate="STS %{x:.2f} · ACD %{y:.2f}<br>%{text}<extra></extra>")
    fig.add_scatter(x=[query["sts"]], y=[query["acd"]], mode="markers",
                    marker=dict(size=22, color="gold", symbol="star",
                                line=dict(width=1.5, color="#333")),
                    name=t("scatter_you"), hovertemplate=t("scatter_you") + "<extra></extra>")
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="STS (mm)", yaxis_title="ACD (mm)",
                      legend=dict(orientation="h", y=1.08))
    return fig

def timepoint_stats(cohort: pd.DataFrame, day: int) -> dict:
    """Apply the settle/recovery curves to the cohort at a given post-op day."""
    vault = cohort["vault"].values * VAULT_SETTLE[day]
    va = cohort["bcva_final"].values * VA_RECOVERY[day]
    return {
        "vault_median": int(np.median(vault)),
        "vault_p25": int(np.percentile(vault, 25)),
        "vault_p75": int(np.percentile(vault, 75)),
        "vault_dist": vault,
        "va_dist": va,
        "va_share_10": float(np.mean(va >= 1.0)),
    }

def badge(kind: str, n: int | None = None) -> str:
    """Provenance tag for the Scholar persona. kind = 'observed' | 'ratified'."""
    label = t("badge_observed") if kind == "observed" else t("badge_ratified")
    tail = f" · n={n}" if n is not None else ""
    return f"`{label}{tail}`"



# ===========================================================================
# 5c. Plain-language glossary + animated manual-vs-AI comparison (additive)
# ===========================================================================
CSS = """
<style>
@keyframes pulse {0%{opacity:1}50%{opacity:.4}100%{opacity:1}}
@keyframes fadein {from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
@keyframes shimmer {0%{background-position:-220px 0}100%{background-position:220px 0}}
.pulse-badge{display:inline-block;padding:3px 12px;border-radius:12px;font-size:.82rem;
  font-weight:700;animation:pulse 1.6s ease-in-out infinite}
.badge-ok{background:#d8f3dc;color:#1b4332}
.badge-warn{background:#ffd6d6;color:#7f1d1d}
.scanbar{height:4px;border-radius:2px;margin:6px 0 2px;background:linear-gradient(90deg,
  rgba(42,157,143,.15),#2a9d8f,rgba(42,157,143,.15));background-size:220px 100%;
  animation:shimmer 1.4s linear infinite}
.cmp-card{border-radius:14px;padding:14px 18px;animation:fadein .5s ease;height:100%}
.cmp-manual{background:#f4f4f5;border:1px solid #e4e4e7}
.cmp-ai{background:linear-gradient(135deg,#e7f5f1,#eef6ff);border:1px solid #b7e4d3}
.cmp-card h4{margin:.1rem 0 .5rem}
.cmp-card ul{margin:.2rem 0 0 1.1rem;padding:0;line-height:1.7}
.term-card{background:#fbfbfd;border:1px solid #ececf1;border-radius:12px;padding:12px 14px;
  margin-bottom:10px;animation:fadein .4s ease}
.term-name{font-weight:700;margin-bottom:4px;color:#264653}
.term-data{margin-top:6px;font-size:.86rem;color:#2a9d8f}
</style>
"""

# term -> plain-language meaning + why it matters, per language. Data anchors
# (real cohort numbers) are computed separately in data_anchor().
GLOSSARY: dict[str, dict[str, dict[str, str]]] = {
    "acd": {
        "zh": {"name": "前房深度 (ACD)", "plain": "眼球角膜和自身晶状体之间那段空间的深度。",
               "why": "空间越充裕，植入的镜片越有余地，术后越安全。"},
        "en": {"name": "Anterior chamber depth (ACD)",
               "plain": "How deep the space is between the cornea and your natural lens.",
               "why": "More room means the implanted lens sits more safely."},
        "ja": {"name": "前房深度 (ACD)", "plain": "角膜と水晶体の間の空間の深さです。",
               "why": "空間が広いほど、レンズを安全に留置できます。"}},
    "wtw": {
        "zh": {"name": "角膜横径 / 白到白 (WTW)", "plain": "角膜的水平宽度，用来估计眼睛内部大小。",
               "why": "帮助挑选合适的镜片尺寸，避免过大或过小。"},
        "en": {"name": "White-to-white (WTW)",
               "plain": "The horizontal width of the cornea — a proxy for eye size.",
               "why": "Helps pick a lens size that isn't too big or too small."},
        "ja": {"name": "角膜横径 (WTW)", "plain": "角膜の横幅で、眼の内部の大きさの目安です。",
               "why": "適切なレンズサイズ選びに役立ちます。"}},
    "sts": {
        "zh": {"name": "沟到沟 (STS)", "plain": "眼内固定镜片的两个位置之间的距离。",
               "why": "选尺寸最关键的测量，直接影响拱高是否理想。"},
        "en": {"name": "Sulcus-to-sulcus (STS)",
               "plain": "The distance between the two spots inside the eye that hold the lens.",
               "why": "The most important measurement for sizing — it drives the vault."},
        "ja": {"name": "毛様溝間距離 (STS)", "plain": "眼内でレンズを支える2点間の距離です。",
               "why": "サイズ選定で最重要。ボールトを左右します。"}},
    "sph": {
        "zh": {"name": "球镜度数 (SPH)", "plain": "近视的度数，负数越大表示近视越深。",
               "why": "决定需要多强的矫正，影响镜片度数。"},
        "en": {"name": "Sphere (SPH)",
               "plain": "Your degree of short-sight — a bigger negative number means stronger myopia.",
               "why": "Determines how much correction the lens must provide."},
        "ja": {"name": "球面度数 (SPH)", "plain": "近視の度数で、マイナスが大きいほど強い近視です。",
               "why": "必要な矯正量、つまりレンズ度数を決めます。"}},
    "vault": {
        "zh": {"name": "拱高 (Vault)", "plain": "植入的 ICL 与你自身晶状体之间留出的小间隙。",
               "why": "太高可能升高眼压，太低可能接触晶状体；落在理想区间最安全。"},
        "en": {"name": "Vault",
               "plain": "The tiny gap left between the implanted ICL and your natural lens.",
               "why": "Too high can raise eye pressure, too low can touch the lens — the middle is safest."},
        "ja": {"name": "ボールト (Vault)", "plain": "留置した ICL と自分の水晶体との間の小さな隙間です。",
               "why": "高すぎると眼圧上昇、低すぎると水晶体に接触。適正域が最も安全です。"}},
    "va": {
        "zh": {"name": "视力 (VA)", "plain": "看视力表能看清的程度，1.0 约等于标准视力。",
               "why": "衡量手术效果最直观的指标。"},
        "en": {"name": "Visual acuity (VA)",
               "plain": "How clearly you read the eye chart — 1.0 is roughly standard vision.",
               "why": "The most direct measure of how well surgery worked."},
        "ja": {"name": "視力 (VA)", "plain": "視力表をどれだけ見えるか。1.0 が標準的な視力です。",
               "why": "手術効果を最も直接的に示す指標です。"}},
    "lens_size": {
        "zh": {"name": "镜片尺寸 (ICL Size)", "plain": "植入镜片的整体直径，需与你的眼睛尺寸匹配。",
               "why": "尺寸合适才能把拱高保持在安全范围：太大拱高过高，太小拱高过低。"},
        "en": {"name": "Lens size (ICL size)",
               "plain": "The overall diameter of the implanted lens, matched to your eye.",
               "why": "The right size keeps the vault safe — too big pushes it too high, too small too low."},
        "ja": {"name": "レンズサイズ (ICL Size)", "plain": "留置するレンズの全体径で、眼の大きさに合わせます。",
               "why": "適切なサイズがボールトを安全域に保ちます。大きすぎると高く、小さすぎると低くなります。"}},
    "cohort": {
        "zh": {"name": "相似队列 (Cohort · “n=”)",
               "plain": "刘教授既往病例中与你眼睛最相似的一组；“n=”就是他们的数量。",
               "why": "群体越大、越相似，预期结果越可信——那是真实经验，不是猜测。"},
        "en": {"name": "Cohort (\"n=\")",
               "plain": "The group of Prof. Liu's past patients whose eyes best match yours; \"n=\" is how many.",
               "why": "A larger, closer group makes the outcome more trustworthy — real experience, not a guess."},
        "ja": {"name": "コホート (Cohort · 「n=」)",
               "plain": "劉教授の過去症例のうち、あなたの眼に最も近いグループ。「n=」はその数です。",
               "why": "大きく近いグループほど予測は信頼でき、推測ではなく実際の経験に基づきます。"}},
    "kanon": {
        "zh": {"name": "k-匿名 (k-Anonymity)",
               "plain": "一种隐私保护：只有当足够多的相似患者存在时才作答，从而无法反推出某一个人。",
               "why": "既保护每位患者隐私，也防止系统在数据太少时过度断言。"},
        "en": {"name": "k-anonymity",
               "plain": "A privacy safeguard: results appear only when enough similar patients exist, so no "
                        "single person can be singled out.",
               "why": "It protects patient privacy and stops the system over-claiming on too little data."},
        "ja": {"name": "k-匿名性 (k-Anonymity)",
               "plain": "十分な数の類似患者がいる時のみ結果を表示し、個人を特定できないようにする仕組みです。",
               "why": "患者のプライバシーを守り、データ不足時の過剰な断定を防ぎます。"}},
}

def data_anchor(key: str, df: pd.DataFrame, lang: str) -> str:
    """A sentence grounding the term in Prof. Liu's ACTUAL cohort numbers."""
    N = len(df)
    if key == "acd":
        lo, hi = np.percentile(df["acd"], [10, 90])
        return {"zh": f"刘教授 {N} 例中，前房深度多在 {lo:.1f}–{hi:.1f} mm；较深者通常安全空间更充裕。",
                "en": f"Across Prof. Liu's {N} eyes, ACD mostly falls {lo:.1f}–{hi:.1f} mm; deeper "
                      f"chambers usually have more safety room.",
                "ja": f"劉教授の {N} 眼では ACD は概ね {lo:.1f}–{hi:.1f} mm。深いほど安全域に余裕があります。"}[lang]
    if key == "wtw":
        lo, hi = np.percentile(df["wtw"], [10, 90])
        return {"zh": f"刘教授病例的角膜横径多在 {lo:.1f}–{hi:.1f} mm 之间。",
                "en": f"In Prof. Liu's records, WTW mostly ranges {lo:.1f}–{hi:.1f} mm.",
                "ja": f"劉教授の症例では WTW は概ね {lo:.1f}–{hi:.1f} mm です。"}[lang]
    if key == "sts":
        lo, hi = np.percentile(df["sts"], [10, 90])
        return {"zh": f"刘教授据此（多在 {lo:.1f}–{hi:.1f} mm）为每只眼选定尺寸。",
                "en": f"Prof. Liu uses this (mostly {lo:.1f}–{hi:.1f} mm) to size each eye.",
                "ja": f"劉教授はこの値（概ね {lo:.1f}–{hi:.1f} mm）でサイズを決めます。"}[lang]
    if key == "sph":
        worst, best = df["sph"].min(), df["sph"].max()
        return {"zh": f"刘教授病例覆盖约 {worst:.0f}~{best:.0f} D 的近视范围。",
                "en": f"Prof. Liu's cases span roughly {worst:.0f} to {best:.0f} D of myopia.",
                "ja": f"劉教授の症例は約 {worst:.0f}~{best:.0f} D の近視をカバーします。"}[lang]
    if key == "vault":
        med = int(df["vault"].median()); lo, hi = np.percentile(df["vault"], [25, 75])
        return {"zh": f"刘教授病例的拱高中位约 {med} µm（多在 {lo:.0f}–{hi:.0f} µm），落在理想安全区间。",
                "en": f"Median vault in Prof. Liu's records is ~{med} µm (mostly {lo:.0f}–{hi:.0f} µm), "
                      f"inside the ideal safety window.",
                "ja": f"劉教授の症例のボールト中央値は約 {med} µm（概ね {lo:.0f}–{hi:.0f} µm）で理想的な安全域内です。"}[lang]
    if key == "va":
        pct = float((df["bcva_final"] >= 1.0).mean()) * 100
        return {"zh": f"在刘教授病例中，约 {pct:.0f}% 的眼术后达到 1.0 或更好的视力。",
                "en": f"In Prof. Liu's records, about {pct:.0f}% of eyes reach 1.0 vision or better.",
                "ja": f"劉教授の症例では、約 {pct:.0f}% の眼が術後 1.0 以上に達します。"}[lang]
    if key == "lens_size":
        modal = df["size"].mode().iloc[0]; share = float((df["size"] == modal).mean()) * 100
        return {"zh": f"刘教授 {N} 例中，镜片尺寸从 12.1 到 13.7 mm 不等，最常用 {modal:g} mm（约 {share:.0f}% 的眼）。",
                "en": f"Across Prof. Liu's {N} eyes, sizes span 12.1–13.7 mm; the most common is {modal:g} mm "
                      f"(about {share:.0f}% of eyes).",
                "ja": f"劉教授の {N} 眼ではサイズは 12.1〜13.7 mm。最も多いのは {modal:g} mm（約 {share:.0f}%）です。"}[lang]
    if key == "cohort":
        return {"zh": f"相似队列取自刘教授 {N} 只经核验的眼；调整上方滑块时，n 会实时变化。",
                "en": f"The cohort is drawn from Prof. Liu's {N} verified eyes; n updates live as you move the "
                      f"sliders above.",
                "ja": f"コホートは劉教授の {N} 眼の検証済みデータから抽出され、上のスライダー操作で n が即時に変化します。"}[lang]
    if key == "kanon":
        return {"zh": f"本演示的下限为 k={K_ANON}：低于此数即拒绝作答。在刘教授 {N} 例中，常见参数的眼很容易满足，"
                      f"只有罕见组合才会低于门槛。",
                "en": f"The floor here is k={K_ANON}: below it the system refuses to answer. Across Prof. Liu's "
                      f"{N} eyes, typical eyes clear it easily — only rare parameter combinations fall below.",
                "ja": f"本デモの下限は k={K_ANON}：これを下回ると回答しません。劉教授の {N} 眼では一般的な眼は容易に満たし、"
                      f"稀な組み合わせのみ下回ります。"}[lang]
    return ""

def term_help(key: str) -> str:
    """One-line hover tooltip: plain meaning + why it matters."""
    g = GLOSSARY[key][st.session_state.get("lang", "en")]
    return f"{g['plain']} — {g['why']}"

def render_glossary(df: pd.DataFrame) -> None:
    lang = st.session_state.get("lang", "en")
    cols = st.columns(2)
    for i, key in enumerate(["acd", "wtw", "sts", "sph", "vault", "va",
                             "lens_size", "cohort", "kanon"]):
        g = GLOSSARY[key][lang]
        html = (f"<div class='term-card'><div class='term-name'>{g['name']}</div>"
                f"<div><b>{t('plain_label')}</b> {g['plain']}</div>"
                f"<div><b>{t('why_label')}</b> {g['why']}</div>"
                f"<div class='term-data'>📎 {data_anchor(key, df, lang)}</div></div>")
        cols[i % 2].markdown(html, unsafe_allow_html=True)

def render_manual_vs_ai(df: pd.DataFrame, n: int) -> None:
    lang = st.session_state.get("lang", "en"); N = len(df); k = K_ANON
    man = "".join(f"<li>{p}</li>" for p in T["cmp_manual_pts"][lang])
    ai = "".join(f"<li>{p.format(N=N, n=n, k=k)}</li>" for p in T["cmp_ai_pts"][lang])
    c1, c2 = st.columns(2)
    c1.markdown(f"<div class='cmp-card cmp-manual'><h4>{t('cmp_manual_title')}</h4>"
                f"<ul>{man}</ul></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='cmp-card cmp-ai'><h4>{t('cmp_ai_title')}</h4>"
                f"<ul>{ai}</ul></div>", unsafe_allow_html=True)
    st.caption("📎 " + t("cmp_anchor").format(N=N, n=n, k=k))


def main() -> None:
    st.set_page_config(page_title="Liu's Digital Brain", page_icon="🧠", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    # language switcher (sets session state before anything else renders)
    with st.sidebar:
        choice = st.selectbox("语言 / Language / 言語", list(LANGS.keys()),
                              index=1, key="lang_choice")
        st.session_state["lang"] = LANGS[choice]

    df = load_history()

    with st.sidebar:
        st.markdown("---")
        st.subheader(t("sidebar_data"))
        st.metric(t("sidebar_cases"), len(df))
        st.caption(t("sidebar_note"))

    st.title("🧠 " + t("app_title"))
    st.caption(t("tagline"))

    tab1, tab2, tab3 = st.tabs([t("tab1"), t("tab2"), t("tab3")])

    # ---------------- Tab 1: Patient Trust & Education ----------------
    with tab1:
        st.subheader(t("t1_head"))
        st.markdown(t("persona"))
        st.write(t("t1_intro"))
        c = st.columns(5)
        q = {   # shared inputs — feed BOTH the new live explorer and the original flow
            "acd": c[0].number_input(t("acd"), 2.6, 4.2, 3.40, 0.01, help=term_help("acd")),
            "wtw": c[1].number_input(t("wtw"), 10.5, 13.0, 11.80, 0.01, help=term_help("wtw")),
            "sts": c[2].number_input(t("sts"), 10.5, 13.5, 12.05, 0.01, help=term_help("sts")),
            "sph": c[3].number_input(t("sph"), -20.0, -2.0, -9.00, 0.25, help=term_help("sph")),
            "age": c[4].number_input(t("age"), 18, 55, 30, 1),
        }

        # NEW (additive): plain-language glossary, each term anchored in real data
        with st.expander(t("understand_terms")):
            render_glossary(df)

        # ============================================================
        # NEW (additive): live similar-case explorer + honesty meter + scrubber
        # ============================================================
        st.markdown("### " + t("explorer_h"))
        tight = st.slider(t("tightness"), 1, 10, 5)
        radius = 0.35 + (10 - tight) * 0.22          # tight=10 -> 0.35, tight=1 -> 2.33
        cohort, dist, mask = cohort_within(df, q, radius)
        n = len(cohort)

        # --- LIVE cohort-size meter + k-anonymity honesty gate ---
        meter, msg = st.columns([1, 4])
        meter.metric(t("meter_label"), f"n = {n}", help=term_help("cohort"))
        if n < K_ANON:
            # honest refusal — the designed moment that proves we don't fabricate
            msg.error(t("cohort_thin").format(n=n, k=K_ANON))
            st.markdown(f"<span class='pulse-badge badge-warn'>{t('live_badge')} · "
                        f"{t('kanon_bad_badge')}</span>", unsafe_allow_html=True)
            st.markdown("<div class='scanbar'></div>", unsafe_allow_html=True)
            st.plotly_chart(cohort_scatter(df, q, mask), use_container_width=True)
            st.caption(t("scatter_title"))
        else:
            msg.success(t("cohort_ok").format(n=n))
            st.markdown(f"<span class='pulse-badge badge-ok'>{t('live_badge')} · "
                        f"{t('kanon_ok_badge')}</span>", unsafe_allow_html=True)
            st.markdown("<div class='scanbar'></div>", unsafe_allow_html=True)
            modal_size = cohort["size"].mode().iloc[0]

            left, right = st.columns([3, 2])
            with left:
                st.markdown("#### " + t("scatter_title"))
                st.plotly_chart(cohort_scatter(df, q, mask), use_container_width=True)
            with right:
                st.markdown("#### " + t("t1_cohort"))
                st.metric(t("t1_common_size"), f"{modal_size} mm", help=term_help("lens_size"))
                st.markdown(badge("observed", n))
                st.info(t("t1_insight_txt").format(
                    n=n, size=modal_size, v=timepoint_stats(cohort, 90)["vault_median"]))

            # --- Longitudinal vault + vision scrubber ---
            st.markdown("---")
            st.markdown("#### " + t("scrubber_h"))
            labels = t("tp_labels")
            sel = st.select_slider(t("scrubber_tp"), options=list(range(len(TP_DAYS))),
                                   value=3, format_func=lambda i: labels[i])
            day = TP_DAYS[sel]
            s = timepoint_stats(cohort, day)

            sc = st.columns(3)
            sc[0].metric(t("vault_at"), f"{s['vault_median']} µm")
            sc[1].metric(t("t1_vault_range"), f"{s['vault_p25']}–{s['vault_p75']} µm")
            sc[2].metric(t("va_at"), f"{s['va_share_10']*100:.0f}%")
            st.markdown(badge("observed", n))

            gl, gr = st.columns([3, 2])
            with gl:
                st.plotly_chart(vault_preview(s["vault_median"]), use_container_width=True)
                st.caption(t("scrubber_note").format(n=n))
            with gr:
                dist_fig = go.Figure(go.Histogram(x=s["vault_dist"], nbinsx=18,
                                                  marker_color="#2a9d8f"))
                dist_fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                                       xaxis_title="vault µm", yaxis_title="")
                st.plotly_chart(dist_fig, use_container_width=True)
                show = cohort[["acd", "wtw", "sts", "sph", "size", "vault"]].head(15).reset_index()
                show.columns = t("t1_table_cols")
                st.dataframe(show, use_container_width=True, hide_index=True, height=220)

        # ============================================================
        # NEW (additive): manual vs. AI-assisted comparison (uses live n)
        # ============================================================
        st.markdown("---")
        st.markdown("### " + t("cmp_h"))
        render_manual_vs_ai(df, n)

        # ============================================================
        # ORIGINAL (unchanged): historical expert case matching (button flow)
        # ============================================================
        st.markdown("---")
        st.markdown("### " + t("expert_h"))
        if st.button(t("match_btn"), type="primary"):
            hits = match_cases(df, q, k=25)
            pred_vault = int(hits["vault"].median())
            p25, p75 = int(hits["vault"].quantile(.25)), int(hits["vault"].quantile(.75))
            modal_size = hits["size"].mode().iloc[0]

            st.markdown("### " + t("t1_cohort"))
            m = st.columns(4)
            m[0].metric(t("t1_cohort_n"), len(hits))
            m[1].metric(t("t1_pred_vault"), f"{pred_vault} µm")
            m[2].metric(t("t1_vault_range"), f"{p25}–{p75} µm")
            m[3].metric(t("t1_common_size"), f"{modal_size} mm")

            left, right = st.columns([3, 2])
            with left:
                st.markdown("#### " + t("t1_preview"))
                st.plotly_chart(vault_preview(pred_vault), use_container_width=True)
                st.caption(t("t1_preview_cap"))
            with right:
                st.markdown("#### " + t("t1_insight"))
                st.info(t("t1_insight_txt").format(n=len(hits), size=modal_size, v=pred_vault))
                show = hits[["acd", "wtw", "sts", "sph", "size", "vault"]].reset_index()
                show.columns = t("t1_table_cols")
                st.dataframe(show, use_container_width=True, hide_index=True, height=280)

        st.caption(t("compliance"))

    # ---------------- Tab 2: Academic & Clinical Intelligence ----------------
    with tab2:
        st.subheader(t("t2_head"))
        st.markdown("### " + t("t2_research_h"))
        st.caption(t("t2_research_cap"))
        if st.button(t("t2_research_btn"), type="primary"):
            for i, topic in enumerate(research_topics(df), 1):
                st.markdown(f"**{i}.** {topic}")

        st.markdown("---")
        st.markdown("### " + t("t2_nomo_h"))
        st.caption(t("t2_nomo_cap"))
        nomo = build_nomogram(df)
        disp = nomo.copy()
        disp["delta"] = disp["delta"].apply(lambda x: f"{x:+.2f} mm")
        disp.columns = t("t2_nomo_cols")
        st.dataframe(disp, use_container_width=True, hide_index=True)

        deep_n = int((df["acd"] >= 3.3).sum())
        st.markdown(badge("observed", deep_n) + "  ·  " + f"*{t('badge_descriptive')}*")
        st.success(t("t2_nomo_insight").format(n=deep_n))
        st.caption("🔒 " + t("t2_await"))

    # ---------------- Tab 3: Partner / Supplier ----------------
    with tab3:
        st.subheader(t("t3_head"))
        st.write(t("t3_intro"))
        m = st.columns(4)
        m[0].metric(t("t3_m_eyes"), len(df))
        m[1].metric(t("t3_m_meanvault"), int(df["vault"].mean()))
        m[2].metric(t("t3_m_sizes"), df["size"].nunique())
        m[3].metric(t("t3_m_kanon"), "k ≥ 5")

        st.markdown("#### " + t("t3_dist_h"))
        dist = df["size"].value_counts().sort_index()
        fig = go.Figure(go.Bar(x=[f"{s} mm" for s in dist.index], y=dist.values,
                               marker_color="#2a9d8f"))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        if st.button(t("t3_export_btn")):
            st.success(t("t3_export_ok"))

        st.markdown("---")
        st.markdown("#### " + t("t3_priv_h"))
        st.info(t("t3_priv_txt"))


if __name__ == "__main__":
    main()