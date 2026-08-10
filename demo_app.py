"""
Surgeon's Digital Brain — Local-First Demo App
==========================================
A leadership-facing demonstration of the eye-clinic intelligence + digital-asset
system, built on the three-layer spine (Facts -> Judgments -> Principles).

Local-first: runs entirely on the doctor's machine. The historical case data
here is SYNTHETIC demo data standing in for Surgeon's de-identified archive —
no real patient records are used in this demo.

Run locally:
    pip install streamlit plotly numpy pandas
    streamlit run demo_app.py

Trilingual: 简体中文 / English / 日本語 (switch in the sidebar).
"""
from __future__ import annotations
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ===========================================================================
# 0. i18n — every UI string, medical term, and insight in zh / en / ja
# ===========================================================================
LANGS = {"简体中文": "zh", "English": "en", "日本語": "ja"}

T: dict[str, dict[str, str]] = {
    "app_title": {"zh": "手术医生的数字大脑", "en": "Surgeon's Digital Brain",
                  "ja": "手術医師のデジタルブレイン"},
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
    "t1_intro": {"zh": "输入患者生物测量参数，系统即时检索手术医生历史病例中最相似的眼睛，"
                       "用真实数据建立信任与合理预期。",
                 "en": "Enter the patient's biometry; the system instantly retrieves the most "
                       "similar eyes from Surgeon's history to build trust and set expectations.",
                 "ja": "患者の生体計測値を入力すると、手術医師の症例から最も類似した眼を即時に検索し、"
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
        "zh": "在与本眼高度相似的 {n} 只历史眼中，手术医生最常选择 {size} mm，术后拱高中位 {v} µm，"
              "落在理想安全区间。这为患者提供了以真实经验为依据的可靠预期。",
        "en": "Across {n} highly similar historical eyes, Surgeon most often chose {size} mm, "
              "with a median post-op vault of {v} µm — within the ideal safety window. This gives "
              "the patient expectations grounded in real experience.",
        "ja": "本眼に酷似する {n} 眼の履歴では、手術医師は {size} mm を最も多く選択し、術後ボールト中央値は "
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
    "t2_nomo_cap": {"zh": "按前房深度分层，展示手术医生的实际选择倾向。此为“描述其既往行为”，"
                          "经本人确认后方可升级为处方性原则。",
                    "en": "Stratified by ACD, showing Surgeon's actual choice tendency. This is "
                          "DESCRIPTIVE (what was done); it becomes prescriptive only once he ratifies it.",
                    "ja": "前房深度で層別化し、手術医師の実際の選択傾向を表示。これは記述的（過去の行動）であり、"
                          "本人の承認を経て初めて処方的原則になります。"},
    "t2_nomo_cols": {"zh": ["ACD 区间", "病例数", "最常尺寸", "相对参考的平均偏移", "平均拱高µm"],
                     "en": ["ACD band", "n", "Modal size", "Mean Δ vs reference", "Mean vault µm"],
                     "ja": ["ACD 区間", "症例数", "最頻サイズ", "参照との平均偏差", "平均ボールトµm"]},
    "t2_nomo_insight": {
        "zh": "已识别的签名倾向：当 ACD ≥ 3.3 mm 时，手术医生倾向比参考尺寸上调一档 —— "
              "此模式已在 {n} 例中观察到，等待本人确认。",
        "en": "Detected signature tendency: when ACD ≥ 3.3 mm, Surgeon tends to upsize one step "
              "vs. the reference — observed in {n} cases, awaiting his ratification.",
        "ja": "検出された署名的傾向：ACD ≥ 3.3 mm のとき、手術医師は参照より一段大きいサイズを選ぶ傾向 —— "
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
    "persona": {"zh": "🎓 手术医生的数字学者：只引用手术医生本人的病例数据，绝不臆测。",
                "en": "🎓 Surgeon's Digital Scholar: cites only Surgeon's own case records — never speculates.",
                "ja": "🎓 手術医師のデジタル学者：手術医師自身の症例のみを引用し、推測は一切しません。"},
    "badge_observed": {"zh": "📎 手术医生病例中观察到", "en": "📎 observed in Surgeon's records",
                       "ja": "📎 手術医師の症例で観察"},
    "badge_ratified": {"zh": "✅ 手术医生已确认的原则", "en": "✅ Surgeon's ratified principle",
                       "ja": "✅ 手術医師が承認した原則"},
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
    "cohort_ok": {"zh": "在手术医生病例中找到 **{n}** 只与本眼高度相似的眼。",
                  "en": "Found **{n}** eyes in Surgeon's records highly similar to this one.",
                  "ja": "手術医師の症例に、本眼に酷似する眼が **{n}** 眼見つかりました。"},
    "cohort_thin": {"zh": "⚠️ 手术医生病例中相似的眼过少（n={n} < {k}），不足以在此可靠陈述。请放宽相似度或调整参数。",
                    "en": "⚠️ Too few similar eyes in Surgeon's records (n={n} < {k}) to speak confidently "
                          "here. Loosen the similarity or adjust the parameters.",
                    "ja": "⚠️ 手術医師の症例に類似する眼が少なすぎます（n={n} < {k}）。確信を持って述べられません。"
                          "類似度を緩めるかパラメータを調整してください。"},
    "scatter_title": {"zh": "「你在这里」：本眼在手术医生病例中的位置", "en": "\"You are here\": this eye among Surgeon's cases",
                      "ja": "「あなたはここ」：手術医師の症例における本眼の位置"},
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
    "understand_terms": {"zh": "👉 点此了解这些专业术语（通俗解释 + 手术医生真实数据）",
                         "en": "👉 Tap to understand these clinical terms (plain language + Surgeon's real data)",
                         "ja": "👉 これらの専門用語をやさしく理解する（手術医師の実データつき）"},
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
        "zh": ["从手术医生 {N} 例中客观检索", "精确相似队列 n={n}，可复现", "术语即时转为通俗语言",
               "k≥{k} 隐私校验实时通过", "预期基于真实结果分布"],
        "en": ["Objective search across Surgeon's {N} eyes", "Exact cohort n={n}, reproducible",
               "Terms auto-translated to plain language", "k≥{k} privacy check verified live",
               "Expectations from real outcome distribution"],
        "ja": ["手術医師の {N} 眼から客観的に検索", "正確なコホート n={n}、再現可能",
               "用語を即座にやさしく変換", "k≥{k} プライバシー検証を即時通過",
               "期待は実際の結果分布に基づく"]},
    "cmp_anchor": {"zh": "同一只眼：人工凭印象 → 本系统从手术医生 {N} 例中客观检索出 n={n} 例相似眼，并通过 k≥{k} 隐私校验。",
                   "en": "Same eye: manual impression → this system objectively retrieves n={n} similar eyes "
                         "from Surgeon's {N}, k≥{k} privacy-verified.",
                   "ja": "同じ眼：手動の印象 → 本システムは手術医師の {N} 眼から n={n} 眼を客観的に抽出し、k≥{k} で検証。"},
    "live_badge": {"zh": "● 实时", "en": "● LIVE", "ja": "● ライブ"},
    "kanon_ok_badge": {"zh": "✓ k-匿名已校验", "en": "✓ k-anonymity verified", "ja": "✓ k-匿名性 検証済"},
    "kanon_bad_badge": {"zh": "✕ 低于 k 阈值", "en": "✕ below k-threshold", "ja": "✕ k閾値未満"},
    # --- vault preview labels / views / risk simulation ---
    "lbl_lens": {"zh": "晶状体", "en": "Crystalline lens", "ja": "水晶体"},
    "lbl_vault": {"zh": "拱高间隙", "en": "Vault gap", "ja": "ボールト間隙"},
    "lbl_cornea": {"zh": "角膜", "en": "Cornea", "ja": "角膜"},
    "lbl_pupil": {"zh": "瞳孔", "en": "Pupil", "ja": "瞳孔"},
    "view_mode": {"zh": "视角", "en": "View", "ja": "視点"},
    "view_3d": {"zh": "🧊 立体图", "en": "🧊 3D", "ja": "🧊 3D"},
    "view_2d": {"zh": "📐 侧视图 + 俯视图", "en": "📐 Side + Top", "ja": "📐 側面＋俯瞰"},
    "view_side": {"zh": "侧视图（剖面）", "en": "Side view (profile)", "ja": "側面図（断面）"},
    "view_top": {"zh": "俯视图（俯瞰）", "en": "Top view (overhead)", "ja": "俯瞰図"},
    "zone_low": {"zh": "偏低", "en": "Low", "ja": "低い"},
    "zone_ideal": {"zh": "理想", "en": "Ideal", "ja": "理想"},
    "zone_monitor": {"zh": "偏高", "en": "High-ish", "ja": "やや高い"},
    "zone_high": {"zh": "过高", "en": "Too high", "ja": "高すぎ"},
    "risk_low": {"zh": "拱高偏低（<250µm）：ICL 可能接触自身晶状体，存在白内障风险。",
                 "en": "Vault low (<250 µm): the ICL may contact your natural lens — a cataract risk.",
                 "ja": "ボールトが低い（<250µm）：ICLが水晶体に接触する可能性があり、白内障リスク。"},
    "risk_ideal": {"zh": "拱高理想（250–750µm）：间隙充足，房水循环稳定，最安全。",
                   "en": "Ideal vault (250–750 µm): ample clearance and stable fluid flow — the safest range.",
                   "ja": "理想的なボールト（250–750µm）：十分なクリアランスと安定した房水循環で最も安全。"},
    "risk_monitor": {"zh": "拱高偏高（750–1000µm）：通常可接受，但建议随访观察。",
                     "en": "Vault a bit high (750–1000 µm): usually acceptable, but worth monitoring.",
                     "ja": "ボールトがやや高い（750–1000µm）：通常許容範囲ですが経過観察を推奨。"},
    "risk_high": {"zh": "拱高过高（>1000µm）：可能引起房角变窄或眼压升高。",
                  "en": "Vault too high (>1000 µm): may narrow the drainage angle or raise eye pressure.",
                  "ja": "ボールトが高すぎる（>1000µm）：隅角の狭小化や眼圧上昇のリスク。"},
    # --- patient-facing anatomy education & gentle-landing (outcome-free) ---
    "edu_head": {"zh": "了解你的眼睛，以及 ICL 温柔工作的原理",
                 "en": "Understand your eye — and how the ICL gently works",
                 "ja": "あなたの眼と、ICL のやさしい仕組みを知る"},
    "edu_persona": {"zh": "🎓 互动解剖导览 · 仅供科普教育，不预测任何手术结果。",
                    "en": "🎓 Interactive anatomy guide · educational only — no outcomes are predicted.",
                    "ja": "🎓 インタラクティブ解剖ガイド · 教育目的のみで、結果は予測しません。"},
    "edu_intro": {"zh": "输入你的生物测量数据，探索属于你自己的眼睛结构，并亲眼看到 ICL 如何在不切开角膜的情况下轻轻就位。",
                  "en": "Enter your measurements to explore your own eye's structure and watch how the ICL "
                        "gently settles into place — without cutting the cornea.",
                  "ja": "計測値を入力して、あなた自身の眼の構造を探り、角膜を切らずに ICL がそっと収まる様子をご覧ください。"},
    "meet_h": {"zh": "① 认识你的眼睛（结构自探索）", "en": "① Meet your eye (structural self-discovery)",
               "ja": "① あなたの眼を知る（構造の自己発見）"},
    "anat_h": {"zh": "你的前房结构示意", "en": "Your anterior-segment sketch", "ja": "あなたの前房の模式図"},
    "anat_caption": {"zh": "根据你的测量值绘制的示意图（仅供说明，非诊断）。",
                     "en": "Drawn to scale from your measurements (illustrative, not a diagnosis).",
                     "ja": "計測値に基づく模式図です（説明用であり診断ではありません）。"},
    "pop_h": {"zh": "你的数据 vs. 常见范围", "en": "Your measurements vs. the common range",
              "ja": "あなたの数値 vs. 一般的な範囲"},
    "pop_you": {"zh": "你", "en": "You", "ja": "あなた"},
    "pop_normal": {"zh": "好消息：你的数据落在该手术常见的范围内。具体是否适合，请由医生评估。",
                   "en": "Good news: your measurements sit within the common range for this procedure. "
                         "Your surgeon will assess what's right for you.",
                   "ja": "朗報：あなたの数値はこの手術で一般的な範囲内です。適否は医師が評価します。"},
    "landing_h": {"zh": "② 温柔就位：ICL 如何轻轻安家", "en": "② The gentle landing: how the ICL softly comes to rest",
                  "ja": "② やさしい着地：ICL がそっと収まるまで"},
    "landing_intro": {"zh": "拖动下方滑块，亲手看着晶体一步步就位——全程角膜完好、从不被切开。",
                      "en": "Drag the slider and watch the lens settle, step by step — the cornea stays whole "
                            "and is never cut.",
                      "ja": "スライダーを動かし、レンズが少しずつ収まる様子をご覧ください。角膜は終始そのままで切開されません。"},
    "stage_label": {"zh": "步骤", "en": "Step", "ja": "ステップ"},
    "cornea_intact": {"zh": "角膜全程完好，从未被切开", "en": "The cornea stays whole — never cut",
                      "ja": "角膜は終始そのまま——切開されません"},
    "lbl_iris": {"zh": "虹膜", "en": "Iris", "ja": "虹彩"},
    "lbl_chamber": {"zh": "前房", "en": "Anterior chamber", "ja": "前房"},
    "landing_open_tag": {"zh": "微小自闭合切口 ~2–3mm", "en": "tiny self-sealing opening ~2–3mm",
                         "ja": "自己閉鎖の極小切開 ~2–3mm"},
    "landing_fold_tag": {"zh": "折叠的柔软晶体", "en": "soft folded lens", "ja": "折りたたんだ柔らかいレンズ"},
    "landing_unfold_tag": {"zh": "在眼内缓缓展开", "en": "unfolding gently inside", "ja": "眼内でゆっくり展開"},
    "landing_rest_tag": {"zh": "安放于虹膜后方", "en": "resting behind the iris", "ja": "虹彩の後ろに着地"},
    "landing_stage_names": {
        "zh": ["微创小切口", "折叠植入", "缓缓展开", "虹膜后就位"],
        "en": ["Tiny opening", "Folded lens enters", "Unfolds softly", "Rests behind the iris"],
        "ja": ["極小の切開", "折りたたんで挿入", "ゆっくり展開", "虹彩の後ろに着地"]},
    "landing_narr": {
        "zh": ["医生先做一个约 2–3 毫米、可自行闭合的微小切口——比一粒米还小，通常无需缝线。",
               "柔软的晶体像卷起的花瓣一样折叠着，从小切口轻轻送入眼内。它比隐形眼镜还柔软。",
               "在眼内，晶体缓缓、平稳地自行展开——没有牵拉，没有切割。",
               "晶体轻轻安放在虹膜后方的自然空间里，由眼睛自身结构稳稳托住。什么也不会被取出，角膜始终完好。"],
        "en": ["A tiny ~2–3 mm self-sealing opening is made — smaller than a grain of rice, usually "
               "needing no stitches.",
               "The soft lens, folded like a rolled petal, is gently guided in through the tiny opening. "
               "It's softer than a contact lens.",
               "Inside the eye, the lens slowly and smoothly unfolds on its own — no pulling, no cutting.",
               "The lens rests gently in the natural space behind the iris, held by the eye's own "
               "structures. Nothing is removed, and the cornea stays intact."],
        "ja": ["約 2〜3mm の自己閉鎖する極小の切開を作ります。米粒より小さく、通常縫合は不要です。",
               "柔らかいレンズを花びらのように折りたたみ、小さな切開からそっと挿入します。コンタクトより柔らかい素材です。",
               "眼内でレンズがゆっくり滑らかに自然に展開します。引っ張りも切開もありません。",
               "レンズは虹彩の後ろの自然な空間にそっと収まり、眼自身の構造に支えられます。何も取り除かれず、角膜はそのままです。"]},
    "lbl_sulcus": {"zh": "睫状沟", "en": "Ciliary sulcus", "ja": "毛様溝"},
    "spotlight_h": {"zh": "点亮结构，认识你的眼睛", "en": "Spotlight a structure to explore",
                    "ja": "構造をハイライトして探索"},
    "play_word": {"zh": "播放", "en": "Play", "ja": "再生"},
    "play_hint": {"zh": "▶ 点击播放，或拖动下方时间轴，观看晶体温柔就位。",
                  "en": "▶ Tap Play, or drag the timeline, to watch the lens gently land.",
                  "ja": "▶ 再生をタップ、またはタイムラインをドラッグして、レンズの着地をご覧ください。"},
    "fact_cornea": {"zh": "角膜是眼睛透明的“窗户”，也是全身最敏感的组织之一——但手术全程它始终保持完好。",
                    "en": "The cornea is your eye's clear 'window' and one of the body's most sensitive "
                          "tissues — yet it stays whole throughout.",
                    "ja": "角膜は眼の透明な「窓」で、全身で最も敏感な組織の一つですが、手術中も無傷のままです。"},
    "fact_iris": {"zh": "虹膜是眼睛的“光圈”，决定你眼睛的颜色并调节进入的光线。ICL 就安放在它的正后方。",
                  "en": "The iris is your eye's 'aperture' — it gives your eyes their color and controls "
                        "the light. The ICL rests just behind it.",
                  "ja": "虹彩は眼の「絞り」で、瞳の色を決め、光の量を調整します。ICL はそのすぐ後ろに収まります。"},
    "fact_chamber": {"zh": "前房是角膜与虹膜之间充满清澈房水的空间，为眼睛提供缓冲与营养。",
                     "en": "The anterior chamber is the fluid-filled space between cornea and iris, "
                           "cushioning and nourishing the eye.",
                     "ja": "前房は角膜と虹彩の間の房水で満たされた空間で、眼を保護し栄養を与えます。"},
    "fact_lens": {"zh": "晶状体是你自身的天然镜片，ICL 与它协同工作——不取出、也不替换你的晶状体。",
                  "en": "The crystalline lens is your own natural lens; the ICL works alongside it — your "
                        "lens is never removed or replaced.",
                  "ja": "水晶体はあなた自身の天然レンズです。ICL はそれと協働し、水晶体を取り除いたり交換したりしません。"},
    "fact_sulcus": {"zh": "睫状沟是虹膜后方的天然“搁架”，ICL 就轻轻搁在这里，由眼睛自身结构稳稳托住。",
                    "en": "The ciliary sulcus is a natural 'shelf' behind the iris where the ICL gently "
                          "rests, held by your eye's own structures.",
                    "ja": "毛様溝は虹彩の後ろの天然の「棚」で、ICL はここにそっと収まり、眼自身の構造に支えられます。"},
    # --- Publishing Copilot (clinician) ---
    "copilot_h": {"zh": "📝 论文选题与大纲生成（发表副驾）", "en": "📝 Publishing Copilot — topic & outline generator",
                  "ja": "📝 論文テーマ・アウトライン生成（パブリッシング副操縦士）"},
    "copilot_cap": {"zh": "从你的去标识病例库出发，把临床直觉快速转化为可发表的结构化框架（仅为框架，非结论）。",
                    "en": "Turn a clinical hunch into a publishable, structured framework from your "
                          "de-identified archive (a scaffold, not conclusions).",
                    "ja": "非識別化アーカイブから、臨床の直感を発表可能な構造化フレームに（枠組みであり結論ではありません）。"},
    "theme_select": {"zh": "研究主题", "en": "Research theme", "ja": "研究テーマ"},
    "custom_title": {"zh": "自定义研究标题", "en": "Custom research title", "ja": "カスタム研究タイトル"},
    "copilot_btn": {"zh": "✨ 生成发表框架", "en": "✨ Generate publication package", "ja": "✨ 発表パッケージを生成"},
    "titles_h": {"zh": "推荐标题（中英对照）", "en": "Suggested titles (bilingual)", "ja": "推奨タイトル（対訳）"},
    "outline_h": {"zh": "论文结构大纲", "en": "Paper outline", "ja": "論文アウトライン"},
    "checklist_h": {"zh": "数据清单（对照现有病例库）", "en": "Data checklist (against your archive)",
                    "ja": "データチェックリスト（アーカイブ照合）"},
    "copilot_disclaimer": {"zh": "以上为写作框架，非研究结论；所有统计需在满足 k≥5 的分层上，用你的真实数据填充。",
                           "en": "This is a writing scaffold, not findings; fill every statistic from your real "
                                 "data on strata satisfying k≥5.",
                           "ja": "これは執筆の枠組みであり結論ではありません。統計は k≥5 を満たす層で実データから記入してください。"},
    "theme_label_vault_tight_sulcus": {"zh": "紧沟眼的拱高动态", "en": "Vault dynamics in tight sulci",
                                       "ja": "狭い毛様溝でのボールト動態"},
    "theme_label_ecd_trajectory": {"zh": "ICL 术后 ECD 纵向轨迹", "en": "Longitudinal ECD trajectory post-ICL",
                                   "ja": "ICL 術後 ECD の経時推移"},
    "theme_label_sizing_age": {"zh": "个性化尺寸规则的年龄分层效能",
                               "en": "Personalized sizing efficacy across age cohorts",
                               "ja": "年齢別の個別化サイズ規則の有効性"},
    "theme_label_custom": {"zh": "自定义…", "en": "Custom…", "ja": "カスタム…"},
    "theme_focus_vault_tight_sulcus": {"zh": "紧窄睫状沟眼中的拱高表现",
                                       "en": "vault behaviour in eyes with tight ciliary sulci",
                                       "ja": "狭い毛様溝の眼におけるボールト挙動"},
    "theme_focus_ecd_trajectory": {"zh": "ICL 植入术后角膜内皮细胞密度的纵向轨迹",
                                   "en": "the longitudinal endothelial cell density trajectory after ICL",
                                   "ja": "ICL 植込み後の角膜内皮細胞密度の経時的推移"},
    "theme_focus_sizing_age": {"zh": "个性化尺寸规则在不同年龄队列中的效能",
                               "en": "the efficacy of personalized sizing rules across age cohorts",
                               "ja": "年齢コホート間での個別化サイズ規則の有効性"},
    "theme_titles_vault_tight_sulcus": {
        "zh": ["紧窄睫状沟眼的拱高动态：一项真实世界回顾", "窄沟眼拱高预测：单中心经验"],
        "en": ["Vault Dynamics in Eyes with Tight Ciliary Sulci: A Real-World Analysis",
               "Predicting Vault in Narrow-Sulcus Eyes: Single-Centre Experience"],
        "ja": ["狭い毛様溝の眼におけるボールト動態：リアルワールド回顧", "狭沟眼のボールト予測：単施設経験"]},
    "theme_titles_ecd_trajectory": {
        "zh": ["ICL 植入术后角膜内皮细胞密度的纵向轨迹", "ICL 术后内皮安全性：真实世界队列研究"],
        "en": ["Longitudinal Endothelial Cell Density Trajectory Following ICL Implantation",
               "Endothelial Safety After ICL: A Real-World Cohort Study"],
        "ja": ["ICL 植込み後の角膜内皮細胞密度の経時推移", "ICL 術後の内皮安全性：リアルワールドコホート研究"]},
    "theme_titles_sizing_age": {
        "zh": ["个性化 ICL 尺寸规则在不同年龄队列中的效能", "ICL 手术中按年龄分层的尺寸优化"],
        "en": ["Efficacy of Personalized ICL Sizing Rules Across Age Cohorts",
               "Age-Stratified Sizing Refinement in ICL Surgery"],
        "ja": ["年齢コホート間での個別化 ICL サイズ規則の有効性", "ICL 手術における年齢層別サイズ最適化"]},
    "sec_abstract": {"zh": "摘要", "en": "Abstract", "ja": "抄録"},
    "sec_background": {"zh": "引言 / 背景", "en": "Introduction / Background", "ja": "序論 / 背景"},
    "sec_methods": {"zh": "材料与方法", "en": "Materials & Methods", "ja": "材料と方法"},
    "sec_results": {"zh": "结果（可支持的数据点）", "en": "Results (supported data points)", "ja": "結果（対応可能なデータ）"},
    "sec_discussion": {"zh": "讨论要点", "en": "Discussion highlights", "ja": "考察のポイント"},
    "outline_abstract": {
        "zh": "本研究基于 {n} 只去标识真实世界眼的回顾性队列，探讨{focus}。〔在此填写研究目的、主要终点与核心发现。〕",
        "en": "This study examines {focus} using a retrospective, de-identified real-world cohort of {n} "
              "eyes. [State objective, primary endpoint, and key finding.]",
        "ja": "本研究は {n} 眼の非識別化リアルワールド回顧コホートを用い、{focus}を検討する。〔目的・主要評価項目・主要所見を記載〕"},
    "outline_background": {
        "zh": "ICL 尺寸选择及其结构相关性仍是活跃领域。本文聚焦{focus}。〔综述既往文献并指出本病例库可填补的空白。〕",
        "en": "ICL sizing and its structural correlates remain active areas. This work focuses on {focus}. "
              "[Summarise prior literature and the specific gap your archive addresses.]",
        "ja": "ICL のサイズ選択とその構造的相関は依然活発な領域である。本稿は{focus}に焦点を当てる。〔先行文献と本アーカイブが埋める空白を記載〕"},
    "outline_methods": {
        "zh": "对来自单一高流量诊所的 {n} 只去标识眼进行回顾性分析；按标准化 22 字段方案提取生物测量（ACD、WTW、STS）"
              "与器械参数；样本量低于 n={k} 的分层予以抑制（k-匿名）。〔说明纳入标准、终点与统计方法。〕",
        "en": "Retrospective analysis of {n} de-identified eyes from a single high-volume practice. Biometry "
              "(ACD, WTW, STS) and device parameters were extracted per a standardized 22-field schema; "
              "strata below n={k} are suppressed for privacy (k-anonymity). [Specify inclusion criteria, "
              "endpoints, and statistical approach.]",
        "ja": "単一の高症例数施設の非識別化 {n} 眼を回顧的に解析。標準化 22 項目スキーマで生体計測（ACD・WTW・STS）と"
              "デバイス情報を抽出し、n={k} 未満の層はプライバシー保護のため抑制（k-匿名性）。〔選択基準・評価項目・統計手法を記載〕"},
    "outline_results": {
        "zh": "本主题在现有病例库可支持的分析：目标队列 n={cn}。〔填入描述性分布、分层汇总与图表，并标注每层的确切 n。〕",
        "en": "Analyses the archive can support for this theme: cohort of interest n={cn}. [Insert descriptive "
              "distributions, stratified summaries, and figures; report exact n per stratum.]",
        "ja": "本テーマで現アーカイブが対応可能な解析：対象コホート n={cn}。〔記述的分布・層別要約・図表を記入し、各層の n を明記〕"},
    "outline_discussion": {
        "zh": "〔鉴于观察性设计，将结果解释为关联而非因果。讨论该模式如何优化尺寸决策、单中心数据的局限，以及下一步验证。〕",
        "en": "[Interpret findings as association, not causation, given the observational design. Discuss how "
              "the pattern could refine sizing decisions, limits of single-centre data, and next validation "
              "steps.]",
        "ja": "〔観察研究であることを踏まえ、結果は因果ではなく関連として解釈。サイズ決定の改善可能性、単施設データの限界、次の検証段階を考察〕"},
    "chk_missing": {"zh": "病例库中暂无此字段——建议开始采集", "en": "not yet in the archive — consider collecting it",
                    "ja": "アーカイブに未収録——収集の検討を"},
    "chk_sts": {"zh": "沟到沟 STS", "en": "Sulcus-to-sulcus (STS)", "ja": "毛様溝間 STS"},
    "chk_size": {"zh": "选用镜片尺寸", "en": "Chosen lens size", "ja": "選択サイズ"},
    "chk_vault": {"zh": "拱高", "en": "Vault", "ja": "ボールト"},
    "chk_ecd_pre": {"zh": "术前内皮细胞密度", "en": "Preop endothelial cell density", "ja": "術前内皮細胞密度"},
    "chk_ecd_post": {"zh": "术后内皮细胞密度", "en": "Postop endothelial cell density", "ja": "術後内皮細胞密度"},
    "chk_vault_series": {"zh": "多时点拱高随访", "en": "Multi-timepoint vault follow-up", "ja": "多時点ボールト経過"},
    "chk_age": {"zh": "年龄", "en": "Age", "ja": "年齢"},
    "chk_acd": {"zh": "前房深度 ACD", "en": "Anterior chamber depth", "ja": "前房深度 ACD"},
    "nomo_kanon": {"zh": "🔒 所有分层均显示 n；低于 k=5 的分层将被抑制。",
                   "en": "🔒 Every stratum shows its n; strata below k=5 are suppressed.",
                   "ja": "🔒 各層は n を表示し、k=5 未満の層は抑制されます。"},
    # --- Live Case Ingestion (clinician) ---
    "ingest_h": {"zh": "➕ 录入新病例 · 动态更新病例库", "en": "➕ Upload new case · live database updater",
                 "ja": "➕ 新規症例の登録 · データベース動的更新"},
    "ingest_cap": {"zh": "把每一例成功手术即时录入，本地描述性 nomogram 随即刷新。数据仅存于本次会话（演示）。",
                   "en": "Log each successful case on the fly; the local descriptive nomogram refreshes at once. "
                         "Data is session-only in this demo.",
                   "ja": "成功症例をその場で登録すると、ローカルの記述的ノモグラムが即時更新されます（本デモではセッション内のみ保持）。"},
    "f_sts": {"zh": "STS (mm)", "en": "STS (mm)", "ja": "STS (mm)"},
    "f_acd": {"zh": "ACD (mm)", "en": "ACD (mm)", "ja": "ACD (mm)"},
    "f_wtw": {"zh": "WTW (mm)", "en": "WTW (mm)", "ja": "WTW (mm)"},
    "f_size": {"zh": "镜片尺寸 (mm)", "en": "Lens size (mm)", "ja": "サイズ (mm)"},
    "f_vault": {"zh": "拱高 (µm)", "en": "Vault (µm)", "ja": "ボールト (µm)"},
    "f_sph": {"zh": "球镜 SPH (D)", "en": "Sphere (D)", "ja": "球面度数 (D)"},
    "ingest_add_btn": {"zh": "录入本病例", "en": "Add this case", "ja": "この症例を登録"},
    "ingest_added_msg": {"zh": "✅ 已录入 1 例，病例库已更新。",
                         "en": "✅ Case added — archive updated.", "ja": "✅ 1 例を登録し、アーカイブを更新しました。"},
    "ingest_count": {"zh": "本次会话新增", "en": "Added this session", "ja": "今セッションの追加"},
    "sync_btn": {"zh": "同步并重新校准", "en": "Sync & Recalibrate", "ja": "同期して再校正"},
    "sync_progress": {"zh": "正在整合证据、刷新描述性 nomogram…", "en": "Integrating evidence, refreshing the "
                      "descriptive nomogram…", "ja": "エビデンスを統合し、記述的ノモグラムを更新中…"},
    "sync_done_msg": {"zh": "✅ 已整合 {added} 条新证据 · 病例库共 {total} 眼 · 描述性 nomogram 已刷新 · "
                            "候选信号已浮现，等待你的确认（不自动改写已确认原则）。",
                      "en": "✅ Integrated {added} new evidence record(s) · archive now {total} eyes · "
                            "descriptive nomogram refreshed · a candidate signal has surfaced, awaiting your "
                            "ratification (ratified rules are never auto-rewritten).",
                      "ja": "✅ {added} 件の新エビデンスを統合 · アーカイブ計 {total} 眼 · 記述的ノモグラム更新 · "
                            "候補シグナルが出現し承認待ちです（確認済み原則は自動改変されません）。"},
    "ingest_reset": {"zh": "清除本次新增", "en": "Clear session additions", "ja": "追加分をクリア"},
    "copilot_download": {"zh": "⬇️ 下载 Markdown 大纲", "en": "⬇️ Download Markdown outline",
                         "ja": "⬇️ Markdown アウトラインをダウンロード"},
    "sync_affected_h": {"zh": "受影响的生理分层（ACD）", "en": "Affected physiological band(s) (ACD)",
                        "ja": "影響を受けた生理学的層（ACD）"},
    "l3_candidate_updated": {"zh": "L3 候选规则已更新（待医生确认）",
                             "en": "L3 candidate rule updated (awaiting ratification)",
                             "ja": "L3 候補規則を更新（承認待ち）"},
    # --- Tab 4: Daily Command Center ---
    "tab4": {"zh": "④ 指挥中心", "en": "④ Command Center", "ja": "④ コマンドセンター"},
    "cc_head": {"zh": "每日临床指挥中心", "en": "Daily clinical command center",
                "ja": "デイリー臨床コマンドセンター"},
    "cc_cap": {"zh": "术前将每台手术的生物测量与本地 nomogram 对照，给出邻域先例与置信度——均为描述性参考，"
                     "最终决策由术者做出。",
               "en": "Each scheduled case's biometry is checked against the local nomogram, surfacing "
                     "neighbourhood precedents and confidence — descriptive reference only; the surgeon "
                     "makes every decision.",
               "ja": "予定手術ごとに生体計測を本地ノモグラムと照合し、近傍の先例と信頼度を提示します——"
                     "記述的な参考であり、判断はすべて術者が行います。"},
    "cc_total": {"zh": "今日手术", "en": "Today's cases", "ja": "本日の症例"},
    "cc_ready": {"zh": "就绪", "en": "Ready", "ja": "準備完了"},
    "cc_attention": {"zh": "需关注", "en": "Needs attention", "ja": "要確認"},
    "cc_list_h": {"zh": "今日手术清单与预分析", "en": "Today's surgical list & pre-analysis",
                  "ja": "本日の手術リストと事前分析"},
    "cc_biometry": {"zh": "生物测量", "en": "Biometry", "ja": "生体計測"},
    "cc_preanalysis": {"zh": "预分析", "en": "Pre-analysis", "ja": "事前分析"},
    "cc_flags": {"zh": "审核标记", "en": "Audit flags", "ja": "監査フラグ"},
    "cc_ref": {"zh": "参考尺寸（描述性基线）：{size} mm", "en": "Reference size (descriptive baseline): {size} mm",
               "ja": "参照サイズ（記述的基準）：{size} mm"},
    "cc_neigh_n": {"zh": "相似历史病例：n={n}", "en": "Similar historical cases: n={n}",
                   "ja": "類似症例：n={n}"},
    "cc_neigh_modal": {"zh": "最常选用尺寸：{size} mm", "en": "Most-chosen size: {size} mm",
                       "ja": "最頻サイズ：{size} mm"},
    "cc_neigh_vault": {"zh": "相似病例拱高中位：{v} µm（描述性）",
                       "en": "Median vault in similar cases: {v} µm (descriptive)",
                       "ja": "類似症例のボールト中央値：{v} µm（記述的）"},
    "cc_neigh_suppressed": {"zh": "先例过少（n<{k}），已抑制描述性统计。",
                            "en": "Too few precedents (n<{k}); descriptive stats suppressed.",
                            "ja": "先例が少なく（n<{k}）、記述統計を抑制。"},
    "cc_need_data": {"zh": "补齐数据后可进行邻域分析。", "en": "Complete the data to run neighbourhood analysis.",
                     "ja": "データを補完すると近傍分析が可能です。"},
    "cc_ready_msg": {"zh": "✓ 数据完整，无异常标记", "en": "✓ Complete data, no flags",
                     "ja": "✓ データ完備・フラグなし"},
    "cc_flag_missing": {"zh": "缺少 {field} —— 请在进入手术室前采集。", "en": "Missing {field} — capture before the OR.",
                        "ja": "{field} が欠落——手術室に入る前に取得してください。"},
    "cc_edge_tight_sts": {"zh": "睫状沟偏紧（STS<11.0mm）——历史先例较少，尺寸选择请谨慎。",
                          "en": "Tight ciliary sulcus (STS<11.0 mm) — limited precedent; size with care.",
                          "ja": "毛様溝が狭い（STS<11.0mm）——先例が少なく、サイズ選択に注意。"},
    "cc_edge_deep_acd": {"zh": "前房较深（ACD>3.7mm）——请核对尺寸邻域。",
                         "en": "Deep anterior chamber (ACD>3.7 mm) — review the sizing neighbourhood.",
                         "ja": "前房が深い（ACD>3.7mm）——サイズ近傍を確認。"},
    "cc_edge_shallow_acd": {"zh": "前房较浅（ACD<2.9mm）——请谨慎评估。",
                            "en": "Shallow anterior chamber (ACD<2.9 mm) — assess with care.",
                            "ja": "前房が浅い（ACD<2.9mm）——慎重に評価。"},
    "cc_edge_high_myopia": {"zh": "高度近视（SPH≤−12D）——请核对度数与尺寸。",
                            "en": "High myopia (SPH≤−12 D) — verify power and sizing.",
                            "ja": "強度近視（SPH≤−12D）——度数とサイズを確認。"},
    "cc_edge_sparse": {"zh": "相似历史病例过少——本例先例稀疏，需额外谨慎。",
                       "en": "Very few similar historical cases — sparse precedent; extra caution warranted.",
                       "ja": "類似症例が非常に少なく先例が乏しい——追加の注意が必要。"},
    "conf_high": {"zh": "高置信", "en": "High confidence", "ja": "高信頼"},
    "conf_med": {"zh": "中等置信", "en": "Medium confidence", "ja": "中程度"},
    "conf_sparse": {"zh": "先例稀疏", "en": "Sparse precedent", "ja": "先例少"},
    "cc_kanon_note": {"zh": "所有邻域统计均满足 k≥5；先例过少的分层予以抑制。均为描述性参考。",
                      "en": "All neighbourhood stats satisfy k≥5; sparse strata are suppressed. Descriptive reference only.",
                      "ja": "近傍統計はすべて k≥5 を満たし、先例の少ない層は抑制。記述的参考のみです。"},
    "cc_postop_h": {"zh": "术后回填 · 保持 L3 反馈闭环", "en": "Post-op capture · keep the L3 loop alive",
                    "ja": "術後入力 · L3ループを維持"},
    "cc_postop_cap": {"zh": "手术结束即录入实测拱高与所选尺寸，直接汇入实时病例库。已去标识化。",
                      "en": "Log the achieved vault and chosen size right after surgery — it flows straight "
                            "into the live archive. De-identified.",
                      "ja": "術直後に実測ボールトと選択サイズを入力すると、ライブアーカイブに直接反映されます。非識別化済み。"},
    "cc_postop_case": {"zh": "选择今日病例", "en": "Select today's case", "ja": "本日の症例を選択"},
    "cc_postop_btn": {"zh": "回填并同步", "en": "Log & sync", "ja": "入力して同期"},
    "cc_postop_done": {"zh": "✅ 已回填 {code} 的术后数据，病例库已更新。",
                       "en": "✅ Logged post-op data for {code} — archive updated.",
                       "ja": "✅ {code} の術後データを入力し、アーカイブを更新しました。"},
    "cc_postop_note": {"zh": "🧠 本例作为 L3 候选证据加入，等待术者确认后方可升级为处方性原则。",
                       "en": "🧠 Added as L3 candidate evidence — becomes a prescriptive principle only after "
                             "the surgeon ratifies it.",
                       "ja": "🧠 L3 候補エビデンスとして追加。術者の承認後に処方的原則となります。"},
}

def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return T.get(key, {}).get(lang, T.get(key, {}).get("en", key))


# ===========================================================================
# 1. Synthetic "Surgeon" historical archive (stands in for de-identified data)
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
    # Surgeon's signature: deep ACD -> upsize one step in most such eyes
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
# 4. Patient-facing anatomy education & "Gentle Landing" mechanics (OUTCOME-FREE)
#    High-fidelity layered rendering + smooth Plotly-frame animation.
#    NO post-op vault/vision predictions and NO risk-zone outcome claims here.
# ===========================================================================
def hex_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

def _arc(x0: float, x1: float, apex_y: float, base_y: float, n: int = 60):
    xs = np.linspace(x0, x1, n)
    mid = (x0 + x1) / 2; half = (x1 - x0) / 2
    ys = base_y + (apex_y - base_y) * (1 - ((xs - mid) / half) ** 2)
    return xs, ys

PAL = dict(cornea="#8fd3e8", chamber="#bfe6ef", iris="#b07a3c", iris2="#8d5f2a",
           lens="#e6c34d", glow="#ffe08a", node="#2a9d8f", icl="#2a9d8f")

def _glow(fig, xs, ys, color, width=24):
    fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=hex_rgba(color, 0.28), width=width),
                    hoverinfo="skip")

def _eye_scene(fig: "go.Figure", acd: float, wtw: float, highlight: str = None) -> float:
    """Layered, organic anterior-segment rendering. `highlight` makes one
    structure glow. Returns half-width. Pure anatomy — no device, no outcome."""
    half = wtw / 2
    hi = lambda name: highlight == name

    # crystalline lens (concentric gradient arcs) — drawn first (deepest)
    for k, sc in enumerate([1.0, 0.72, 0.44]):
        lx, ly = _arc(-2.7 * sc, 2.7 * sc, -acd - 1.6 * sc, -acd)
        tx, _ty = _arc(-2.7 * sc, 2.7 * sc, -acd, -acd)
        if hi("lens") and k == 0:
            _glow(fig, lx, ly, PAL["glow"], 26)
        fig.add_scatter(x=np.concatenate([lx, tx[::-1]]), y=np.concatenate([ly, [-acd] * len(tx)]),
                        mode="lines", fill="toself", fillcolor=hex_rgba(PAL["lens"], 0.25 + 0.18 * k),
                        line=dict(color=hex_rgba("#a9832a", 0.5), width=1), hoverinfo="skip")

    # anterior chamber (soft fill between cornea inner and iris plane)
    cx, cy = _arc(-half, half, 0.55, -0.05)
    if hi("chamber"):
        _glow(fig, cx, cy, PAL["glow"], 22)
    fig.add_scatter(x=np.concatenate([cx, [half, -half]]), y=np.concatenate([cy, [-acd, -acd]]),
                    mode="lines", fill="toself", fillcolor=hex_rgba(PAL["chamber"], 0.30),
                    line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip")

    # pupil (dark ellipse seen through the gap)
    th = np.linspace(0, 2 * np.pi, 48)
    fig.add_scatter(x=1.4 * np.cos(th), y=-acd + 0.5 * np.sin(th), mode="lines", fill="toself",
                    fillcolor="rgba(20,20,28,.92)", line=dict(color="#111"), hoverinfo="skip")

    # iris (both sides): glow + body + shading + radial fibres
    for sgn in (-1, 1):
        x0, x1 = sgn * 1.5, sgn * half
        if hi("iris"):
            fig.add_scatter(x=[x0, x1], y=[-acd, -acd], mode="lines",
                            line=dict(color=hex_rgba(PAL["glow"], 0.55), width=22), hoverinfo="skip")
        fig.add_scatter(x=[x0, x1], y=[-acd, -acd], mode="lines",
                        line=dict(color=PAL["iris"], width=13), hoverinfo="skip")
        fig.add_scatter(x=[x0, x1], y=[-acd, -acd], mode="lines",
                        line=dict(color=hex_rgba(PAL["iris2"], 0.6), width=6), hoverinfo="skip")
        fx, fy = [], []
        for xx in np.linspace(x0, x1, 11):
            fx += [xx, xx, None]; fy += [-acd - 0.26, -acd + 0.26, None]
        fig.add_scatter(x=fx, y=fy, mode="lines", line=dict(color=hex_rgba("#5c3a17", 0.5), width=1),
                        hoverinfo="skip")

    # cornea (glassy layered crescent + glint)
    ox, oy = _arc(-half, half, 1.25, 0.15)
    ix, iy = _arc(-half, half, 0.72, -0.05)
    if hi("cornea"):
        _glow(fig, ox, oy, PAL["glow"], 26)
    fig.add_scatter(x=np.concatenate([ox, ix[::-1]]), y=np.concatenate([oy, iy[::-1]]),
                    mode="lines", fill="toself", fillcolor=hex_rgba(PAL["cornea"], 0.33),
                    line=dict(color=hex_rgba(PAL["cornea"], 0.9), width=2), hoverinfo="skip")
    gx, gy = _arc(-half * 0.5, half * 0.15, 1.13, 0.72)
    fig.add_scatter(x=gx, y=gy, mode="lines", line=dict(color="rgba(255,255,255,.7)", width=3),
                    hoverinfo="skip")

    # ciliary sulcus nodes (the ICL's natural resting shelf)
    for sgn in (-1, 1):
        glowing = hi("sulcus")
        fig.add_scatter(x=[sgn * 2.9], y=[-acd], mode="markers",
                        marker=dict(size=16 if glowing else 9,
                                    color=PAL["glow"] if glowing else PAL["node"],
                                    line=dict(color="white", width=1)), hoverinfo="skip")
    return half

def _eye_layout(fig: "go.Figure", half: float, height: int = 380) -> None:
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=8, b=0), showlegend=False,
                      plot_bgcolor="#fbfdff", paper_bgcolor="#fbfdff",
                      xaxis=dict(visible=False, range=[-half - 1.2, half + 1.2]),
                      yaxis=dict(visible=False, scaleanchor="x", scaleratio=1))

def anatomy_explorer(acd: float, wtw: float, highlight: str = None) -> go.Figure:
    """Meet-Your-Eye: the patient's own anterior segment, richly rendered, with
    the selected structure glowing. ACD/WTW shown as anatomy dimensions."""
    fig = go.Figure()
    half = _eye_scene(fig, acd, wtw, highlight)
    fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=-acd,
                  line=dict(color="#2a9d8f", width=2, dash="dot"))
    fig.add_annotation(x=0.15, y=-acd / 2, text=f"ACD {acd:.2f} mm", showarrow=False,
                       font=dict(size=12, color="#2a7f73"), xanchor="left",
                       bgcolor="rgba(255,255,255,.75)")
    fig.add_shape(type="line", x0=-half, x1=half, y0=1.45, y1=1.45,
                  line=dict(color="#457b9d", width=2))
    fig.add_annotation(x=0, y=1.7, text=f"WTW {wtw:.2f} mm", showarrow=False,
                       font=dict(size=12, color="#457b9d"))
    for name, x, y, col in [("cornea", 0, 1.05, "#4f8296"),
                            ("iris", -half + 0.3, -acd + 0.4, "#8d6e63"),
                            ("lens", 0, -acd - 1.85, "#a07d1e"),
                            ("sulcus", 2.9, -acd - 0.5, "#2a7f73")]:
        fig.add_annotation(x=x, y=y, text=t("lbl_" + ("chamber" if name == "x" else name)),
                           showarrow=False, font=dict(size=11, color=col),
                           xanchor="left" if x < 0 else "center")
    _eye_layout(fig, half, 380)
    return fig

def pop_hist(df: pd.DataFrame, col: str, value: float, title: str, color: str) -> go.Figure:
    """Where the patient's measurement sits in the population range — positioning,
    not a suitability score or outcome."""
    fig = go.Figure(go.Histogram(x=df[col], nbinsx=24, marker_color=hex_rgba(color, 0.5)))
    fig.add_shape(type="line", x0=value, x1=value, y0=0, y1=1, yref="paper",
                  line=dict(color="#111", width=3))
    fig.add_annotation(x=value, y=1.06, yref="paper", text=t("pop_you"), showarrow=False,
                       font=dict(size=12, color="#111"))
    fig.update_layout(height=200, margin=dict(l=0, r=0, t=22, b=0), showlegend=False,
                      title=title, xaxis_title="mm", yaxis=dict(visible=False), bargap=0.05)
    return fig

def _icl_polyline(t: float):
    """Smooth path of the ICL as a function of t in [0,1]:
    glide (folded) -> unfold -> descend & rest behind the iris."""
    if t < 0.30:
        f = t / 0.30; cx, cy, w = 3.9 - 2.4 * f, -1.0 - 0.6 * f, 0.55 + 0.15 * f
    elif t < 0.62:
        f = (t - 0.30) / 0.32; cx, cy, w = 1.5 - 1.5 * f, -1.6 - 0.1 * f, 0.70 + 1.75 * f
    else:
        f = (t - 0.62) / 0.38; cx, cy, w = 0.0, -1.7 - 1.45 * f, 2.45 + 0.1 * f
    xs = np.linspace(cx - w, cx + w, 44)
    ys = cy - 0.16 * (1 - ((xs - cx) / w) ** 2)
    return xs, ys

def _icl_stage(t: float) -> int:
    return 0 if t < 0.28 else 1 if t < 0.60 else 2 if t < 0.88 else 3

def gentle_landing_animation(n_frames: int = 42) -> go.Figure:
    """A single smooth, auto-playing / scrubbable Plotly animation: the folded
    ICL glides through the tiny opening, unfolds, and settles behind the iris.
    The cornea is intact throughout and labelled 'never cut'. No numbers."""
    acd, wtw = 3.0, 11.2
    fig = go.Figure()
    half = _eye_scene(fig, acd, wtw)
    base = len(fig.data)

    # incision hint at the corneal edge (static)
    fig.add_scatter(x=[half - 0.9, half - 0.2], y=[0.55, 0.15], mode="lines",
                    line=dict(color=hex_rgba("#e76f51", 0.8), width=4), hoverinfo="skip")

    # initial animated traces: glow, crisp lens, narration text
    x0, y0 = _icl_polyline(0.0)
    fig.add_scatter(x=x0, y=y0, mode="lines", line=dict(color=hex_rgba(PAL["icl"], 0.30), width=20),
                    hoverinfo="skip")
    fig.add_scatter(x=x0, y=y0, mode="lines", line=dict(color=PAL["icl"], width=6), hoverinfo="skip")
    fig.add_scatter(x=[0], y=[1.78], mode="text", text=[t("landing_narr")[0]],
                    textfont=dict(size=12, color="#2a7f73"), hoverinfo="skip")

    frames = []
    for i in range(n_frames):
        tt = i / (n_frames - 1)
        xs, ys = _icl_polyline(tt)
        frames.append(go.Frame(name=str(i), traces=[base + 1, base + 2, base + 3], data=[
            go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=hex_rgba(PAL["icl"], 0.30), width=20),
                       hoverinfo="skip"),
            go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=PAL["icl"], width=6), hoverinfo="skip"),
            go.Scatter(x=[0], y=[1.78], mode="text", text=[t("landing_narr")[_icl_stage(tt)]],
                       textfont=dict(size=12, color="#2a7f73"), hoverinfo="skip"),
        ]))
    fig.frames = frames

    # persistent labels (not touched by frames)
    fig.add_annotation(x=0, y=2.25, text="✔ " + t("cornea_intact"), showarrow=False,
                       font=dict(size=13, color="#2a9d8f"))
    fig.add_annotation(x=-half + 0.3, y=-acd + 0.4, text=t("lbl_iris"), showarrow=False,
                       font=dict(size=11, color="#8d6e63"), xanchor="left")
    fig.add_annotation(x=0, y=-acd - 2.0, text=t("lbl_lens"), showarrow=False,
                       font=dict(size=11, color="#a07d1e"))

    _eye_layout(fig, half, 430)
    fig.update_layout(
        updatemenus=[dict(type="buttons", showactive=False, direction="left",
                          x=0.02, y=-0.02, xanchor="left", yanchor="top",
                          buttons=[
                              dict(label="▶ " + t("play_word"), method="animate",
                                   args=[None, dict(frame=dict(duration=90, redraw=True),
                                                    fromcurrent=True,
                                                    transition=dict(duration=60, easing="cubic-in-out"))]),
                              dict(label="⏸", method="animate",
                                   args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                      mode="immediate")])])],
        sliders=[dict(active=0, x=0.14, len=0.82, y=-0.02, pad=dict(t=0),
                      currentvalue=dict(visible=False),
                      steps=[dict(method="animate", label="",
                                  args=[[str(i)], dict(mode="immediate",
                                                       frame=dict(duration=0, redraw=True),
                                                       transition=dict(duration=0))])
                             for i in range(n_frames)])])
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
            "手术医生 ICL 手术拱高长期稳定性的真实世界证据（RWE）研究",
        ],
        "en": [
            f"Vault distribution and predictors after ICL in high myopia (SPH ≤ −10 D, n={len(hi)})",
            f"Effect of anterior chamber depth on ICL sizing and post-op vault: a {n}-eye RWD review",
            f"Individualized sizing strategy and safety in deep-ACD eyes (ACD ≥ 3.3 mm, n={len(deep)})",
            "Building and validating a vault-prediction model across STS/WTW differences",
            "Real-world evidence (RWE) on long-term vault stability in Surgeon's ICL series",
        ],
        "ja": [
            f"強度近視（SPH ≤ −10 D、n={len(hi)}）におけるICL術後ボールトの分布と規定因子",
            f"前房深度がICLサイズ選択と術後ボールトに与える影響：{n}眼のRWDレビュー",
            f"深前房眼（ACD ≥ 3.3 mm、n={len(deep)}）における個別化サイズ戦略と安全性",
            "STS/WTW差に基づくボールト予測モデルの構築と検証",
            "手術医師のICL症例におけるボールト長期安定性のリアルワールドエビデンス（RWE）",
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
@keyframes glowpulse{0%{box-shadow:0 0 6px rgba(42,157,143,.35)}50%{box-shadow:0 0 20px rgba(42,157,143,.85)}100%{box-shadow:0 0 6px rgba(42,157,143,.35)}}
.glow-panel{border-radius:12px;padding:12px 16px;background:linear-gradient(135deg,#e7f5f1,#eef6ff);
  border:1px solid #b7e4d3;animation:glowpulse 2.6s ease-in-out infinite;font-size:.95rem}
.play-hint{display:inline-block;padding:6px 14px;border-radius:20px;background:#2a9d8f;color:#fff;
  font-weight:700;animation:pulse 1.6s ease-in-out infinite}
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
               "plain": "手术医生既往病例中与你眼睛最相似的一组；“n=”就是他们的数量。",
               "why": "群体越大、越相似，预期结果越可信——那是真实经验，不是猜测。"},
        "en": {"name": "Cohort (\"n=\")",
               "plain": "The group of Surgeon's past patients whose eyes best match yours; \"n=\" is how many.",
               "why": "A larger, closer group makes the outcome more trustworthy — real experience, not a guess."},
        "ja": {"name": "コホート (Cohort · 「n=」)",
               "plain": "手術医師の過去症例のうち、あなたの眼に最も近いグループ。「n=」はその数です。",
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
    """A sentence grounding the term in Surgeon's ACTUAL cohort numbers."""
    N = len(df)
    if key == "acd":
        lo, hi = np.percentile(df["acd"], [10, 90])
        return {"zh": f"手术医生 {N} 例中，前房深度多在 {lo:.1f}–{hi:.1f} mm；较深者通常安全空间更充裕。",
                "en": f"Across Surgeon's {N} eyes, ACD mostly falls {lo:.1f}–{hi:.1f} mm; deeper "
                      f"chambers usually have more safety room.",
                "ja": f"手術医師の {N} 眼では ACD は概ね {lo:.1f}–{hi:.1f} mm。深いほど安全域に余裕があります。"}[lang]
    if key == "wtw":
        lo, hi = np.percentile(df["wtw"], [10, 90])
        return {"zh": f"手术医生病例的角膜横径多在 {lo:.1f}–{hi:.1f} mm 之间。",
                "en": f"In Surgeon's records, WTW mostly ranges {lo:.1f}–{hi:.1f} mm.",
                "ja": f"手術医師の症例では WTW は概ね {lo:.1f}–{hi:.1f} mm です。"}[lang]
    if key == "sts":
        lo, hi = np.percentile(df["sts"], [10, 90])
        return {"zh": f"手术医生据此（多在 {lo:.1f}–{hi:.1f} mm）为每只眼选定尺寸。",
                "en": f"Surgeon uses this (mostly {lo:.1f}–{hi:.1f} mm) to size each eye.",
                "ja": f"手術医師はこの値（概ね {lo:.1f}–{hi:.1f} mm）でサイズを決めます。"}[lang]
    if key == "sph":
        worst, best = df["sph"].min(), df["sph"].max()
        return {"zh": f"手术医生病例覆盖约 {worst:.0f}~{best:.0f} D 的近视范围。",
                "en": f"Surgeon's cases span roughly {worst:.0f} to {best:.0f} D of myopia.",
                "ja": f"手術医師の症例は約 {worst:.0f}~{best:.0f} D の近視をカバーします。"}[lang]
    if key == "vault":
        med = int(df["vault"].median()); lo, hi = np.percentile(df["vault"], [25, 75])
        return {"zh": f"手术医生病例的拱高中位约 {med} µm（多在 {lo:.0f}–{hi:.0f} µm），落在理想安全区间。",
                "en": f"Median vault in Surgeon's records is ~{med} µm (mostly {lo:.0f}–{hi:.0f} µm), "
                      f"inside the ideal safety window.",
                "ja": f"手術医師の症例のボールト中央値は約 {med} µm（概ね {lo:.0f}–{hi:.0f} µm）で理想的な安全域内です。"}[lang]
    if key == "va":
        pct = float((df["bcva_final"] >= 1.0).mean()) * 100
        return {"zh": f"在手术医生病例中，约 {pct:.0f}% 的眼术后达到 1.0 或更好的视力。",
                "en": f"In Surgeon's records, about {pct:.0f}% of eyes reach 1.0 vision or better.",
                "ja": f"手術医師の症例では、約 {pct:.0f}% の眼が術後 1.0 以上に達します。"}[lang]
    if key == "lens_size":
        modal = df["size"].mode().iloc[0]; share = float((df["size"] == modal).mean()) * 100
        return {"zh": f"手术医生 {N} 例中，镜片尺寸从 12.1 到 13.7 mm 不等，最常用 {modal:g} mm（约 {share:.0f}% 的眼）。",
                "en": f"Across Surgeon's {N} eyes, sizes span 12.1–13.7 mm; the most common is {modal:g} mm "
                      f"(about {share:.0f}% of eyes).",
                "ja": f"手術医師の {N} 眼ではサイズは 12.1〜13.7 mm。最も多いのは {modal:g} mm（約 {share:.0f}%）です。"}[lang]
    if key == "cohort":
        return {"zh": f"相似队列取自手术医生 {N} 只经核验的眼；调整上方滑块时，n 会实时变化。",
                "en": f"The cohort is drawn from Surgeon's {N} verified eyes; n updates live as you move the "
                      f"sliders above.",
                "ja": f"コホートは手術医師の {N} 眼の検証済みデータから抽出され、上のスライダー操作で n が即時に変化します。"}[lang]
    if key == "kanon":
        return {"zh": f"本演示的下限为 k={K_ANON}：低于此数即拒绝作答。在手术医生 {N} 例中，常见参数的眼很容易满足，"
                      f"只有罕见组合才会低于门槛。",
                "en": f"The floor here is k={K_ANON}: below it the system refuses to answer. Across Surgeon's "
                      f"{N} eyes, typical eyes clear it easily — only rare parameter combinations fall below.",
                "ja": f"本デモの下限は k={K_ANON}：これを下回ると回答しません。手術医師の {N} 眼では一般的な眼は容易に満たし、"
                      f"稀な組み合わせのみ下回ります。"}[lang]
    return ""

def term_help(key: str) -> str:
    """One-line hover tooltip: plain meaning + why it matters."""
    g = GLOSSARY[key][st.session_state.get("lang", "en")]
    return f"{g['plain']} — {g['why']}"

def render_glossary(df: pd.DataFrame) -> None:
    lang = st.session_state.get("lang", "en")
    cols = st.columns(2)
    for i, key in enumerate(["acd", "wtw", "sts", "sph"]):
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


# ===========================================================================
# 5d. Publishing Copilot + Live Case Ingestion (clinician-facing, compliant)
# ===========================================================================
COPILOT_THEMES = ["vault_tight_sulcus", "ecd_trajectory", "sizing_age", "custom"]

def get_archive(base: pd.DataFrame) -> pd.DataFrame:
    """Base archive + any cases added live this session."""
    added = st.session_state.get("added_cases", [])
    if not added:
        return base
    return pd.concat([base, pd.DataFrame(added)], ignore_index=True)

_THEME_REQUIRED = {
    "vault_tight_sulcus": ["sts", "size", "vault"],
    "ecd_trajectory": ["ecd_pre", "ecd_post", "vault_series"],
    "sizing_age": ["age", "size", "vault", "sts"],
    "custom": ["acd", "sts", "size", "vault"],
}

def _theme_cohort_n(theme: str, arch: pd.DataFrame) -> int:
    if theme == "vault_tight_sulcus":
        return int((arch["sts"] < 11.0).sum())
    return len(arch)

def data_checklist(theme: str, arch: pd.DataFrame):
    """Map required data points to what the archive actually holds. k-anon gated:
    counts below K_ANON are flagged, missing columns are shown honestly."""
    out = []
    for key in _THEME_REQUIRED[theme]:
        if key in arch.columns:
            n = int(arch[key].notna().sum())
            status = "ok" if n >= K_ANON else "thin"
            detail = f"n={n}" if n >= K_ANON else f"n={n} < {K_ANON}"
        else:
            status, detail = "missing", t("chk_missing")
        label = t("chk_" + key) if ("chk_" + key) in T else key
        out.append((label, status, detail))
    return out

def build_outline(theme: str, title: str, arch: pd.DataFrame) -> dict:
    lang = st.session_state.get("lang", "en")
    n = len(arch)
    focus = title if theme == "custom" else t("theme_focus_" + theme)
    cn = _theme_cohort_n(theme, arch)
    sec = {}
    for key in ["abstract", "background", "methods", "results", "discussion"]:
        sec[key] = T["outline_" + key][lang].format(title=title, n=n, focus=focus,
                                                     cn=cn, k=K_ANON)
    return sec

def build_outline_markdown(theme, local_titles, en_titles, sec, checklist) -> str:
    """Assemble the generated package into a Markdown manuscript starter."""
    lang = st.session_state.get("lang", "en")
    primary = local_titles[0] if local_titles else t("custom_title")
    lines = [f"# {primary}", "", f"## {t('titles_h')}"]
    if en_titles and lang != "en":
        for loc, en in zip(local_titles, en_titles):
            lines += [f"- {loc}", f"  - *{en}*"]
    else:
        lines += [f"- {v}" for v in local_titles]
    lines.append("")
    for key in ["abstract", "background", "methods", "results", "discussion"]:
        lines += [f"## {t('sec_' + key)}", sec[key], ""]
    lines.append(f"## {t('checklist_h')}")
    marks = {"ok": "[x]", "thin": "[!]", "missing": "[ ]"}
    for label, status, detail in checklist:
        lines.append(f"- {marks[status]} {label} — {detail}")
    lines += ["", f"> {t('copilot_disclaimer')}"]
    return "\n".join(lines)

def render_publishing_copilot(arch: pd.DataFrame) -> None:
    st.markdown("### " + t("copilot_h"))
    st.caption(t("copilot_cap"))
    theme = st.selectbox(t("theme_select"), COPILOT_THEMES,
                         format_func=lambda k: t("theme_label_" + k))
    custom = st.text_input(t("custom_title")) if theme == "custom" else ""
    if st.button(t("copilot_btn"), type="primary"):
        if theme == "custom":
            local_titles = [custom.strip()] if custom.strip() else []
            en_titles = []
        else:
            local_titles = t("theme_titles_" + theme)
            en_titles = T["theme_titles_" + theme]["en"]   # SCI-grade English, always
        title = (local_titles[0] if local_titles else custom) or t("custom_title")
        sec = build_outline(theme, title, arch)
        checklist = data_checklist(theme, arch)
        st.session_state["copilot_pkg"] = dict(
            theme=theme, local=local_titles, en=en_titles, sec=sec, checklist=checklist,
            md=build_outline_markdown(theme, local_titles, en_titles, sec, checklist))

    pkg = st.session_state.get("copilot_pkg")
    if pkg:
        lang = st.session_state.get("lang", "en")
        st.markdown("#### " + t("titles_h"))
        if pkg["en"] and lang != "en":                    # dual-language: local + English SCI
            for loc, en in zip(pkg["local"], pkg["en"]):
                st.markdown(f"- **{loc}**  \n  <span style='color:#5a7d8c'>*{en}*</span>",
                            unsafe_allow_html=True)
        else:
            for v in pkg["local"]:
                st.markdown(f"- **{v}**")

        st.markdown("#### " + t("outline_h"))
        for key in ["abstract", "background", "methods", "results", "discussion"]:
            with st.expander(t("sec_" + key)):
                st.write(pkg["sec"][key])

        st.markdown("#### " + t("checklist_h"))
        icons = {"ok": "✅", "thin": "⚠️", "missing": "❌"}
        for label, status, detail in pkg["checklist"]:
            st.markdown(f"{icons[status]} **{label}** — {detail}")

        st.download_button("⬇️ " + t("copilot_download"), data=pkg["md"],
                           file_name=f"outline_{pkg['theme']}.md", mime="text/markdown")
        st.caption(t("copilot_disclaimer"))

def render_case_ingestion(base: pd.DataFrame) -> None:
    st.markdown("### " + t("ingest_h"))
    st.caption(t("ingest_cap"))
    with st.form("new_case", clear_on_submit=True):
        a = st.columns(3)
        sts = a[0].number_input(t("f_sts"), 10.0, 13.5, 11.90, 0.01)
        acd = a[1].number_input(t("f_acd"), 2.5, 4.5, 3.30, 0.01)
        wtw = a[2].number_input(t("f_wtw"), 10.0, 13.5, 11.80, 0.01)
        b = st.columns(3)
        size = b[0].selectbox(t("f_size"), [12.1, 12.6, 13.2, 13.7], index=2)
        vault = b[1].number_input(t("f_vault"), 100, 1200, 480, 10)
        sph = b[2].number_input(t("f_sph"), -25.0, -1.0, -8.00, 0.25)
        submitted = st.form_submit_button("➕ " + t("ingest_add_btn"))
    if submitted:
        st.session_state.setdefault("added_cases", []).append({
            "acd": acd, "wtw": wtw, "sts": sts, "sph": sph, "age": 30,
            "ref_size": _reference_size(sts), "size": float(size),
            "vault": int(vault), "bcva_final": 1.05})
        st.success(t("ingest_added_msg"))

    added = st.session_state.get("added_cases", [])
    m = st.columns([2, 1, 1])
    m[0].metric(t("ingest_count"), len(added))
    if m[1].button("🔄 " + t("sync_btn")):
        with st.spinner(t("sync_progress")):
            pb = st.progress(0)
            for i in range(1, 6):
                time.sleep(0.18)
                pb.progress(i / 5)
        arch = get_archive(base)
        st.success(t("sync_done_msg").format(total=len(arch), added=len(added)))
        if added:
            base_bands = base["acd"].apply(_band_label)
            arch_bands = arch["acd"].apply(_band_label)
            band_key = lambda b: -9e9 if "∞" in b.split(",")[0] else float(b.split(",")[0].strip("[ "))
            st.markdown("**" + t("sync_affected_h") + "**")
            for band in sorted({_band_label(c["acd"]) for c in added}, key=band_key):
                before = int((base_bands == band).sum())
                after = int((arch_bands == band).sum())
                st.markdown(f"- `{band} mm` · n {before} → {after} · {t('l3_candidate_updated')}")
    if added and m[2].button("↺ " + t("ingest_reset")):
        st.session_state["added_cases"] = []


# ===========================================================================
# Daily Workflow Command Center (clinician / Departmental OS)
# Strictly descriptive, k-anonymity gated. Embeds the agent in the daily rhythm.
# ===========================================================================
CC_REQUIRED = ["sts", "acd", "wtw", "sph"]
_CONF_STYLE = {"high": ("#d8f3dc", "#1b4332"), "med": ("#fff3cd", "#7a5b00"),
               "sparse": ("#fde0e3", "#7f1d1d")}

@st.cache_data
def todays_roster() -> list:
    """Simulated daily OR roster. Anonymous codes only — no patient identity.
    A few cases carry missing fields / edge biometry to exercise the audits."""
    return [
        {"code": "OR-01", "sts": 11.9, "acd": 3.30, "wtw": 11.8, "sph": -8.0, "age": 31},
        {"code": "OR-02", "sts": None, "acd": 3.20, "wtw": 11.6, "sph": -6.5, "age": 28},
        {"code": "OR-03", "sts": 10.7, "acd": 3.55, "wtw": 11.0, "sph": -14.0, "age": 42},
        {"code": "OR-04", "sts": 12.6, "acd": 3.92, "wtw": 12.5, "sph": -9.0, "age": 37},
        {"code": "OR-05", "sts": 12.0, "acd": 3.25, "wtw": 11.9, "sph": -7.5, "age": 33},
        {"code": "OR-06", "sts": 11.4, "acd": 2.82, "wtw": 11.3, "sph": -18.0, "age": 46},
    ]

def analyze_case(case: dict, arch: pd.DataFrame) -> dict:
    """Descriptive pre-analysis: reference size, neighbourhood precedent (k-anon
    gated), a confidence indicator, and completeness / edge-case flags."""
    missing = [f for f in CC_REQUIRED if case.get(f) is None]
    flags = [("missing", f) for f in missing]
    for key, cond in [("tight_sts", case.get("sts") is not None and case["sts"] < 11.0),
                      ("deep_acd", case.get("acd") is not None and case["acd"] > 3.7),
                      ("shallow_acd", case.get("acd") is not None and case["acd"] < 2.9),
                      ("high_myopia", case.get("sph") is not None and case["sph"] <= -12.0)]:
        if cond:
            flags.append(("edge", key))

    ref = _reference_size(case["sts"]) if case.get("sts") is not None else None
    neigh = None
    if not missing:
        query = {f: case[f] for f in KNN_FEATURES}
        cohort, _, _ = cohort_within(arch, query, radius=1.2)
        n = len(cohort)
        if n >= K_ANON:
            neigh = {"n": n, "modal": cohort["size"].mode().iloc[0],
                     "vault_med": int(cohort["vault"].median()),
                     "conf": "high" if n >= 25 else "med"}
        else:
            neigh = {"n": n, "modal": None, "vault_med": None, "conf": "sparse"}
            flags.append(("edge", "sparse"))
    return {"missing": missing, "flags": flags, "ref": ref, "neigh": neigh}

def _conf_badge(conf: str) -> str:
    bg, fg = _CONF_STYLE[conf]
    return (f"<span style='background:{bg};color:{fg};padding:2px 10px;border-radius:12px;"
            f"font-size:.8rem;font-weight:700'>{t('conf_' + conf)}</span>")

def _cc_field(f: str) -> str:
    return {"sts": "STS", "acd": "ACD", "wtw": "WTW", "sph": "SPH"}[f]

def render_command_center(df: pd.DataFrame) -> None:
    st.subheader(t("cc_head"))
    st.caption(t("cc_cap"))
    arch = get_archive(df)
    roster = todays_roster()
    analyses = [analyze_case(c, arch) for c in roster]
    ready = sum(1 for a in analyses if not a["flags"])

    m = st.columns(3)
    m[0].metric(t("cc_total"), len(roster))
    m[1].metric(t("cc_ready"), ready)
    m[2].metric(t("cc_attention"), len(roster) - ready)

    st.markdown("### " + t("cc_list_h"))
    for case, a in zip(roster, analyses):
        icon = "🟢" if not a["flags"] else ("🔴" if a["missing"] else "🟡")
        with st.expander(f"{icon}  {case['code']}", expanded=bool(a["flags"])):
            cc = st.columns([2, 2, 3])
            with cc[0]:
                st.markdown("**" + t("cc_biometry") + "**")
                for f in CC_REQUIRED:
                    v = case.get(f)
                    st.markdown(f"- {_cc_field(f)}: {'—' if v is None else f'{v:g}'}")
            with cc[1]:
                st.markdown("**" + t("cc_preanalysis") + "**")
                if a["ref"] is not None:
                    st.markdown(t("cc_ref").format(size=f"{a['ref']:g}"))
                if a["neigh"] is not None:
                    nb = a["neigh"]
                    st.markdown(t("cc_neigh_n").format(n=nb["n"]) + "  " + _conf_badge(nb["conf"]),
                                unsafe_allow_html=True)
                    if nb["modal"] is not None:
                        st.markdown(t("cc_neigh_modal").format(size=f"{nb['modal']:g}"))
                        st.markdown(t("cc_neigh_vault").format(v=nb["vault_med"]))
                    else:
                        st.caption(t("cc_neigh_suppressed").format(k=K_ANON))
                else:
                    st.caption(t("cc_need_data"))
            with cc[2]:
                st.markdown("**" + t("cc_flags") + "**")
                if not a["flags"]:
                    st.success(t("cc_ready_msg"))
                for typ, val in a["flags"]:
                    if typ == "missing":
                        st.error(t("cc_flag_missing").format(field=_cc_field(val)))
                    else:
                        st.warning(t("cc_edge_" + val))
    st.caption("🔒 " + t("cc_kanon_note"))

    # ---- Frictionless post-op loop -> live ingestion engine ----
    st.markdown("---")
    st.markdown("### " + t("cc_postop_h"))
    st.caption(t("cc_postop_cap"))
    codes = [c["code"] for c in roster]
    with st.form("postop_capture"):
        pc = st.columns(3)
        sel = pc[0].selectbox(t("cc_postop_case"), codes)
        size = pc[1].selectbox(t("f_size"), [12.1, 12.6, 13.2, 13.7], index=2)
        vault = pc[2].number_input(t("f_vault"), 100, 1200, 480, 10)
        logged = st.form_submit_button("✅ " + t("cc_postop_btn"))
    if logged:
        case = next(c for c in roster if c["code"] == sel)
        rec = {"acd": case.get("acd") or 3.2, "wtw": case.get("wtw") or 11.7,
               "sts": case.get("sts") or 11.9, "sph": case.get("sph") or -8.0,
               "age": case.get("age") or 32, "ref_size": _reference_size(case.get("sts") or 11.9),
               "size": float(size), "vault": int(vault), "bcva_final": 1.05}
        band = _band_label(rec["acd"])
        before = int((get_archive(df)["acd"].apply(_band_label) == band).sum())
        st.session_state.setdefault("added_cases", []).append(rec)
        after = int((get_archive(df)["acd"].apply(_band_label) == band).sum())
        st.success(t("cc_postop_done").format(code=sel))
        st.markdown(f"- `{band} mm` · n {before} → {after} · {t('l3_candidate_updated')}")
        st.caption(t("cc_postop_note"))


def main() -> None:
    st.set_page_config(page_title="Surgeon's Digital Brain", page_icon="🧠", layout="wide")
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

    tab1, tab2, tab3, tab4 = st.tabs([t("tab1"), t("tab2"), t("tab3"), t("tab4")])

    # ---------------- Tab 1: Patient Trust & Education ----------------
    with tab1:
        st.subheader(t("edu_head"))
        st.markdown(t("edu_persona"))
        st.write(t("edu_intro"))

        # ---- A. Meet Your Eye — spotlight anatomy explorer (outcome-free) ----
        st.markdown("### " + t("meet_h"))
        c = st.columns(4)
        acd = c[0].number_input(t("acd"), 2.6, 4.2, 3.40, 0.01, help=term_help("acd"))
        wtw = c[1].number_input(t("wtw"), 10.5, 13.0, 11.80, 0.01, help=term_help("wtw"))
        sts = c[2].number_input(t("sts"), 10.5, 13.5, 12.05, 0.01, help=term_help("sts"))
        sph = c[3].number_input(t("sph"), -20.0, -2.0, -9.00, 0.25, help=term_help("sph"))
        with st.expander(t("understand_terms")):
            render_glossary(df)

        structs = ["cornea", "iris", "chamber", "lens", "sulcus"]
        spot = st.radio(t("spotlight_h"), structs, horizontal=True,
                        format_func=lambda k: t("lbl_" + k))
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(anatomy_explorer(acd, wtw, spot), use_container_width=True)
            st.caption(t("anat_caption"))
            st.markdown(f"<div class='glow-panel'>💡 {t('fact_' + spot)}</div>",
                        unsafe_allow_html=True)
        with right:
            st.markdown("#### " + t("pop_h"))
            st.plotly_chart(pop_hist(df, "acd", acd, t("acd"), "#2a9d8f"),
                            use_container_width=True)
            st.plotly_chart(pop_hist(df, "wtw", wtw, t("wtw"), "#457b9d"),
                            use_container_width=True)
            st.info(t("pop_normal"))

        # ---- B. Gentle Landing — smooth frame-animated micro-assembly ----
        st.markdown("---")
        st.markdown("### " + t("landing_h"))
        st.write(t("landing_intro"))
        st.markdown(f"<span class='play-hint'>{t('play_hint')}</span>", unsafe_allow_html=True)
        st.plotly_chart(gentle_landing_animation(), use_container_width=True)
        st.success("🛡️ " + t("cornea_intact"))

        st.caption(t("compliance"))

    # ---------------- Tab 2: Academic & Clinical Intelligence ----------------
    with tab2:
        st.subheader(t("t2_head"))
        arch = get_archive(df)          # base archive + cases added live this session
        st.markdown("### " + t("t2_research_h"))
        st.caption(t("t2_research_cap"))
        if st.button(t("t2_research_btn"), type="primary"):
            for i, topic in enumerate(research_topics(arch), 1):
                st.markdown(f"**{i}.** {topic}")

        st.markdown("---")
        st.markdown("### " + t("t2_nomo_h"))
        st.caption(t("t2_nomo_cap"))
        nomo = build_nomogram(arch)
        disp = nomo.copy()
        disp["delta"] = disp["delta"].apply(lambda x: f"{x:+.2f} mm")
        disp.columns = t("t2_nomo_cols")
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.caption(t("nomo_kanon"))

        deep_n = int((arch["acd"] >= 3.3).sum())
        st.markdown(badge("observed", deep_n) + "  ·  " + f"*{t('badge_descriptive')}*")
        st.success(t("t2_nomo_insight").format(n=deep_n))
        st.caption("🔒 " + t("t2_await"))

        st.markdown("---")
        render_case_ingestion(df)       # + Upload New Case · live recalibration
        st.markdown("---")
        render_publishing_copilot(arch)  # Publishing Copilot

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

    # ---------------- Tab 4: Daily Command Center ----------------
    with tab4:
        render_command_center(df)


if __name__ == "__main__":
    main()