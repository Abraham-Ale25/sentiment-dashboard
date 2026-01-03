import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO
import re
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ----------------------------
# Pipeline-consistent metrics helper (mirrors your evaluate_model)
# ----------------------------
LABELS_ORDER = ["negative", "neutral", "positive"]

def compute_pipeline_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    neg_f1 = f1_score(
        y_true, y_pred,
        labels=["negative"],
        average="macro",
        zero_division=0
    )

    # optional but useful (your pipeline used it)
    neutral_mask = (y_pred == "neutral")
    neutral_pct = neutral_mask.mean() * 100

    mask_non_neutral = ~neutral_mask
    if mask_non_neutral.sum() > 0:
        non_neutral_f1 = f1_score(
            y_true[mask_non_neutral],
            y_pred[mask_non_neutral],
            labels=["negative", "positive"],
            average="macro",
            zero_division=0
        )
    else:
        non_neutral_f1 = 0.0

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "neg_f1": float(neg_f1),
        "neutral_pct": float(neutral_pct),
        "non_neutral_f1": float(non_neutral_f1),
    }

# ----------------------------
# Load "frozen" global benchmark metrics (preferred)
# If not found, use the GLOBAL_METRICS you embedded in the pipeline
# ----------------------------
DEFAULT_GLOBAL_METRICS = {
    "TextBlob": {"accuracy": 0.501602, "macro_f1": 0.470571, "neg_f1": 0.349116},
    "VADER (Base)": {"accuracy": 0.539853, "macro_f1": 0.529824, "neg_f1": 0.484943},
    "VADER (Enhanced)": {"accuracy": 0.556058, "macro_f1": 0.541701, "neg_f1": 0.488244},
}

def load_global_metrics():
    # If you ship final_results/frozen_experiment_results.json with the app, it will auto-load.
    candidate_paths = [
        os.path.join("final_results", "frozen_experiment_results.json"),
        "frozen_experiment_results.json",
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                test_metrics = data.get("test_set_metrics", {})
                # Normalize keys to match our display names
                out = {}
                if "TextBlob" in test_metrics:
                    out["TextBlob"] = {
                        "accuracy": test_metrics["TextBlob"]["accuracy"],
                        "macro_f1": test_metrics["TextBlob"]["macro_f1"],
                        "neg_f1": test_metrics["TextBlob"]["neg_f1"],
                    }
                if "VADER (Base)" in test_metrics:
                    out["VADER (Base)"] = {
                        "accuracy": test_metrics["VADER (Base)"]["accuracy"],
                        "macro_f1": test_metrics["VADER (Base)"]["macro_f1"],
                        "neg_f1": test_metrics["VADER (Base)"]["neg_f1"],
                    }
                if "VADER (Enhanced)" in test_metrics:
                    out["VADER (Enhanced)"] = {
                        "accuracy": test_metrics["VADER (Enhanced)"]["accuracy"],
                        "macro_f1": test_metrics["VADER (Enhanced)"]["macro_f1"],
                        "neg_f1": test_metrics["VADER (Enhanced)"]["neg_f1"],
                    }
                if out:
                    return out
            except Exception:
                pass
    return DEFAULT_GLOBAL_METRICS

GLOBAL_METRICS = load_global_metrics()

# ----------------------------
# Enhanced VADER class (aligned to your pipeline: phrase rules + dominance + weighting)
# ----------------------------
class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.sia_base = SentimentIntensityAnalyzer()
        self.sia_enh = SentimentIntensityAnalyzer()

        # Phrase replacements (pipeline-consistent)
        self.phrase_replacements = [
            (r"\byeah right\b", " yeah_right "),
            (r"\bas if\b", " as_if "),
            (r"\bnot bad\b", " not_bad "),
            (r"\bnot too good\b", " not_too_good "),
        ]

        # IMPORTANT: your pipeline uses stronger weights for these phrase tokens
        self.sia_enh.lexicon.update({
            "yeah_right": -2.0,
            "as_if": -1.8,
            "not_bad": 1.5,
            "not_too_good": -1.5,
        })

        # Optional: add your domain lexicon (extend as needed)
        # (Keep this small here; you can paste your full +38 lexicon from pipeline)
        self.sia_enh.lexicon.update({
            "fuel-efficient": 2.5,
            "overpriced": -3.0,
            "market crashed": -3.5,
            "bull market": 2.5,
        })

        # Thresholds: ideally load from frozen config if you saved one
        # Falls back to your current tuned-like defaults.
        self.thresholds = self._load_thresholds_or_default()

        self.palette = {
            "TextBlob": "#EF476F",
            "VADER (Base)": "#118AB2",
            "VADER (Enhanced)": "#06D6A0",
            "negative": "#EF476F",
            "neutral": "#FFD166",
            "positive": "#06D6A0",
        }

    def _load_thresholds_or_default(self):
        # If you ship final_results/enhanced_vader_frozen_config.json, it will auto-load thresholds.
        candidate_paths = [
            os.path.join("final_results", "enhanced_vader_frozen_config.json"),
            "enhanced_vader_frozen_config.json",
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    t = data.get("best_thresholds", {})
                    if all(k in t for k in ["pos_thr", "neg_thr", "strong_neg_thr", "strong_pos_thr"]):
                        return {
                            "pos_thr": float(t["pos_thr"]),
                            "neg_thr": float(t["neg_thr"]),
                            "strong_neg_thr": float(t["strong_neg_thr"]),
                            "strong_pos_thr": float(t["strong_pos_thr"]),
                        }
                except Exception:
                    pass

        # fallback (your current)
        return {
            "pos_thr": 0.30,
            "neg_thr": -0.05,
            "strong_neg_thr": -0.25,
            "strong_pos_thr": 0.45
        }

    def textblob_predict(self, text):
        if not isinstance(text, str):
            text = str(text)
        polarity = TextBlob(text).sentiment.polarity
        if polarity >= 0.05:
            return "positive", float(polarity)
        elif polarity <= -0.05:
            return "negative", float(polarity)
        return "neutral", float(polarity)

    def vader_base_predict(self, text):
        if not isinstance(text, str):
            text = str(text)
        scores = self.sia_base.polarity_scores(text)
        c = float(scores["compound"])
        if c >= 0.05:
            return "positive", c, scores
        elif c <= -0.05:
            return "negative", c, scores
        return "neutral", c, scores

    def _preprocess_enh(self, text):
        if not isinstance(text, str):
            text = str(text)
        modified = text
        for pattern, repl in self.phrase_replacements:
            modified = re.sub(pattern, repl, modified, flags=re.IGNORECASE)
        return modified

    def _simple_sent_tokenize(self, text):
        # Robust sentence splitting without requiring NLTK downloads
        # (Works well for long clause text)
        sents = re.split(r'(?<=[.!?])\s+|\n+', text)
        return [s.strip() for s in sents if s and s.strip()]

    def _compute_sentence_weight(self, sentence):
        tokens = re.findall(r"\b\w+\b", sentence)
        n_tokens = len(tokens)
        if n_tokens == 0:
            return 1.0
        len_weight = min(n_tokens / 8.0, 3.0)
        exclam_weight = 1.0 + min(sentence.count("!"), 3) * 0.15
        caps_words = [w for w in tokens if w.isupper() and len(w) > 2]
        caps_weight = 1.0 + min(len(caps_words), 3) * 0.12
        return float(len_weight * exclam_weight * caps_weight)

    def enhanced_vader_predict(self, text):
        t = self.thresholds
        text_proc = self._preprocess_enh(text)
        sentences = self._simple_sent_tokenize(text_proc)
        if not sentences:
            return "neutral", 0.0, []

        comps = []
        weights = []
        for s in sentences:
            vs = self.sia_enh.polarity_scores(s)
            comp = float(vs["compound"])
            w = self._compute_sentence_weight(s)
            comps.append(comp)
            weights.append(w)

        comps = np.array(comps, dtype=float)
        weights = np.array(weights, dtype=float)

        # Dominance rules (pipeline feature)
        if (comps <= t["strong_neg_thr"]).any():
            label = "negative"
        elif (comps >= t["strong_pos_thr"]).any():
            label = "positive"
        else:
            avg_score = float(np.average(comps, weights=weights))
            if avg_score >= t["pos_thr"]:
                label = "positive"
            elif avg_score <= t["neg_thr"]:
                label = "negative"
            else:
                label = "neutral"

        # Return label + avg + per-sentence compounds for explainability
        avg_score = float(np.average(comps, weights=weights)) if len(comps) else 0.0
        return label, avg_score, comps.tolist()

    def analyze_dataframe(self, df, text_col="text"):
        df = df.dropna(subset=[text_col]).copy()
        df["text"] = df[text_col].astype(str)

        tb = df["text"].apply(lambda x: self.textblob_predict(x))
        vb = df["text"].apply(lambda x: self.vader_base_predict(x))
        ve = df["text"].apply(lambda x: self.enhanced_vader_predict(x))

        out = pd.DataFrame({
            "text": df["text"],
            "TextBlob": tb.apply(lambda x: x[0]),
            "TextBlob_score": tb.apply(lambda x: x[1]),
            "VADER_Base": vb.apply(lambda x: x[0]),
            "VADER_Base_compound": vb.apply(lambda x: x[1]),
            "VADER_Enhanced": ve.apply(lambda x: x[0]),
            "VADER_Enhanced_avg": ve.apply(lambda x: x[1]),
            "VADER_Enhanced_sent_compounds": ve.apply(lambda x: x[2]),
        })

        # Agreement signals (useful in viva)
        out["Consensus"] = out[["TextBlob", "VADER_Base", "VADER_Enhanced"]].mode(axis=1)[0]
        out["Disagreement_Flag"] = (
            (out["TextBlob"] != out["VADER_Enhanced"]) |
            (out["VADER_Base"] != out["VADER_Enhanced"])
        )
        return out

analyzer = EnhancedSentimentAnalyzer()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Enhanced VADER Sentiment Analysis", layout="wide")

# Session state for reactive viz
if "last_run_type" not in st.session_state:
    st.session_state.last_run_type = None
if "last_single" not in st.session_state:
    st.session_state.last_single = None
if "last_batch" not in st.session_state:
    st.session_state.last_batch = None
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None

st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 26px; border-radius: 10px; text-align: center; margin-bottom: 18px;'>
  <h1 style='margin:0;'>🚀 Enhanced VADER Dashboard (Pipeline-Aligned)</h1>
  <p style='margin:6px 0 0 0; opacity:0.95;'>Sentence dominance + weighted averaging + phrase/sarcasm rules</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Single Analysis", "📊 Batch Analysis", "📈 Visualisations"])

def render_benchmark_table(pred_row):
    # Shows predicted class + frozen global metrics (pipeline benchmark)
    rows = []
    for model_key, pred_label in pred_row.items():
        m = GLOBAL_METRICS.get(model_key, {})
        rows.append({
            "Model": model_key,
            "Predicted class": pred_label,
            "Benchmark Accuracy (TEST)": m.get("accuracy", np.nan),
            "Benchmark Macro F1 (TEST)": m.get("macro_f1", np.nan),
            "Benchmark Neg F1 (TEST)": m.get("neg_f1", np.nan),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

def render_metrics_if_labels_exist(df_with_preds, label_col, pred_cols_map):
    # Returns metrics dict; also renders a table
    y_true = df_with_preds[label_col].astype(str)

    table_rows = []
    metrics_dict = {}

    for display_name, col in pred_cols_map.items():
        y_pred = df_with_preds[col].astype(str)
        m = compute_pipeline_metrics(y_true, y_pred)
        metrics_dict[display_name] = m
        table_rows.append({
            "Model": display_name,
            "Predicted class": "— (batch)",
            "Accuracy": m["accuracy"],
            "Macro F1": m["macro_f1"],
            "Neg F1": m["neg_f1"],
            "Non-neutral F1": m["non_neutral_f1"],
            "Neutral %": m["neutral_pct"],
        })

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    return metrics_dict

with tab1:
    st.subheader("🔍 Single Text (Prediction + Explainability + Global Benchmark)")

    examples = [
        ("Select example...", ""),
        ("Car: Great economy but uncomfortable seats",
         "The car has excellent fuel economy which saves me money. However, the seats are very uncomfortable on long drives."),
        ("Finance: Market crashed but recovery expected",
         "The stock market crashed by 15% today due to economic concerns. However, analysts expect a recovery next quarter."),
        ("Twitter: Absolutely amazing performance!",
         "Just watched the concert and it was absolutely amazing! The performance was 🔥🔥🔥"),
        ("Mixed: terrible service but good food",
         "The service at this restaurant was terrible — we waited 45 minutes. However, the food was surprisingly good."),
    ]
    selected = st.selectbox("Load example:", [e[0] for e in examples])
    selected_text = next((e[1] for e in examples if e[0] == selected), "")

    text = st.text_area("Input text:", value=selected_text, height=160)

    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.error("Please enter text.")
        else:
            tb_label, tb_score = analyzer.textblob_predict(text)
            vb_label, vb_comp, vb_scores = analyzer.vader_base_predict(text)
            ve_label, ve_avg, ve_sent_scores = analyzer.enhanced_vader_predict(text)

            st.session_state.last_run_type = "single"
            st.session_state.last_single = {
                "text": text,
                "preds": {
                    "TextBlob": tb_label,
                    "VADER (Base)": vb_label,
                    "VADER (Enhanced)": ve_label,
                },
                "scores": {
                    "TextBlob_polarity": tb_score,
                    "VADER_Base_compound": vb_comp,
                    "VADER_Enhanced_avg": ve_avg,
                    "VADER_Enhanced_sentence_compounds": ve_sent_scores,
                }
            }

            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("#### ✅ Predicted classes")
                render_benchmark_table(st.session_state.last_single["preds"])

            with c2:
                st.markdown("#### 🔎 Explainability (Enhanced VADER)")
                st.write(f"**Avg weighted compound (Enhanced):** `{ve_avg:.3f}`")
                st.write(f"**Sentence compounds:** `{[round(x,3) for x in ve_sent_scores]}`")
                st.caption(
                    "Enhanced VADER applies (1) phrase/sarcasm substitutions, "
                    "(2) sentence-level dominance rules, and "
                    "(3) weighted averaging for long/clausal text."
                )

            st.info(
                "Note: Accuracy/Macro-F1/Neg-F1 are **dataset-level** metrics. "
                "For a single text, the correct display is the predicted class + scores, "
                "and the frozen TEST benchmark metrics shown above."
            )

with tab2:
    st.subheader("📊 Batch Analysis (Optional true-label evaluation)")

    uploaded = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"])
    text_col = st.text_input("Text column name:", value="text")
    label_col = st.text_input("Optional label column name (for real metrics):", value="label")

    if uploaded and st.button("Run batch analysis", type="primary"):
        content = StringIO(uploaded.getvalue().decode("utf-8", errors="ignore"))
        try:
            df = pd.read_csv(content)
        except Exception:
            content.seek(0)
            lines = [l.strip() for l in content.readlines() if l.strip()]
            df = pd.DataFrame({text_col: lines})

        if text_col not in df.columns:
            st.error(f"Column '{text_col}' not found.")
        else:
            preds_df = analyzer.analyze_dataframe(df, text_col=text_col)

            # attach ground truth if present
            if label_col in df.columns:
                preds_df[label_col] = df[label_col].astype(str).values

            st.session_state.last_run_type = "batch"
            st.session_state.last_batch = preds_df

            st.success(f"Analyzed {len(preds_df)} texts.")
            st.dataframe(preds_df.head(30), use_container_width=True)

            # Real metrics only if labels exist
            if label_col in preds_df.columns:
                st.markdown("#### 📌 Dataset-level metrics (matches your pipeline logic)")
                metrics = render_metrics_if_labels_exist(
                    preds_df,
                    label_col=label_col,
                    pred_cols_map={
                        "TextBlob": "TextBlob",
                        "VADER (Base)": "VADER_Base",
                        "VADER (Enhanced)": "VADER_Enhanced",
                    },
                )
                st.session_state.last_metrics = metrics
            else:
                st.warning(
                    "No ground-truth label column found, so Accuracy/Macro-F1/Neg-F1 cannot be computed. "
                    "You can still use the Visualisations tab for prediction distributions and disagreement analysis."
                )
                st.session_state.last_metrics = None

            csv = preds_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download results CSV",
                csv,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

with tab3:
    st.subheader("📈 Visualisations (auto-updates after Single or Batch)")

    if st.session_state.last_run_type is None:
        st.info("Run a Single or Batch analysis first — charts will appear here automatically.")
    else:
        if st.session_state.last_run_type == "single":
            s = st.session_state.last_single
            preds = s["preds"]

            st.markdown("#### Single-text model comparison (predicted class + benchmark quality)")
            render_benchmark_table(preds)

            # Show sentence-compound plot for explainability
            sent_scores = s["scores"]["VADER_Enhanced_sentence_compounds"]
            if sent_scores:
                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.plot(range(1, len(sent_scores) + 1), sent_scores, marker="o")
                ax.set_title("Enhanced VADER — Sentence compound scores")
                ax.set_xlabel("Sentence index")
                ax.set_ylabel("Compound")
                ax.axhline(analyzer.thresholds["strong_neg_thr"], linestyle="--")
                ax.axhline(analyzer.thresholds["strong_pos_thr"], linestyle="--")
                st.pyplot(fig, use_container_width=True)

        else:
            dfp = st.session_state.last_batch

            # Distribution chart (Enhanced VADER)
            st.markdown("#### Prediction distribution (Enhanced VADER)")
            counts = dfp["VADER_Enhanced"].value_counts().reindex(LABELS_ORDER).fillna(0)

            fig1, ax1 = plt.subplots(figsize=(7, 4))
            ax1.bar(counts.index, counts.values,
                    color=[analyzer.palette["negative"], analyzer.palette["neutral"], analyzer.palette["positive"]])
            ax1.set_title("Enhanced VADER — Predicted sentiment distribution")
            ax1.set_ylabel("Count")
            st.pyplot(fig1, use_container_width=True)

            # Disagreement rate
            st.markdown("#### Disagreement diagnostics (why Enhanced VADER helps)")
            disagree_rate = float(dfp["Disagreement_Flag"].mean() * 100)
            st.metric("Disagreement rate (vs Enhanced)", f"{disagree_rate:.1f}%")

            # Confusion matrix if labels exist
            if "label" in dfp.columns:
                st.markdown("#### Confusion matrix (Enhanced VADER)")
                cm = confusion_matrix(dfp["label"], dfp["VADER_Enhanced"], labels=LABELS_ORDER)
                fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
                sns.heatmap(cm, annot=True, fmt="d",
                            xticklabels=[x.title() for x in LABELS_ORDER],
                            yticklabels=[x.title() for x in LABELS_ORDER],
                            ax=ax2)
                ax2.set_xlabel("Predicted")
                ax2.set_ylabel("True")
                st.pyplot(fig2, use_container_width=True)

                # Metric bar chart using computed metrics (if stored)
                if st.session_state.last_metrics:
                    st.markdown("#### Model metrics comparison (from your batch run)")
                    m = st.session_state.last_metrics
                    models = list(m.keys())
                    accs = [m[k]["accuracy"] for k in models]
                    macrof1 = [m[k]["macro_f1"] for k in models]
                    negf1 = [m[k]["neg_f1"] for k in models]

                    fig3, ax3 = plt.subplots(figsize=(8, 4))
                    ax3.plot(models, accs, marker="o", label="Accuracy")
                    ax3.plot(models, macrof1, marker="o", label="Macro F1")
                    ax3.plot(models, negf1, marker="o", label="Neg F1")
                    ax3.set_ylim(0, 1)
                    ax3.set_title("Accuracy / Macro F1 / Neg F1 (Batch)")
                    ax3.legend()
                    st.pyplot(fig3, use_container_width=True)

st.markdown("---")
st.caption(
    "Tip for deployment: include your `final_results/frozen_experiment_results.json` and "
    "`final_results/enhanced_vader_frozen_config.json` alongside the app so metrics/thresholds load automatically."
)
