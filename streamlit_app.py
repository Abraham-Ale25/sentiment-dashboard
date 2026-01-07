# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from io import StringIO
from datetime import datetime

import plotly.graph_objects as go
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Enhanced VADER Sentiment Analysis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# CUSTOM CSS (Improved: optional max-width + presenter mode friendly)
# =========================================================
st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
        padding-top: 0 !important;
    }

    /* Optional max width container */
    .app-shell {
        background: rgba(255,255,255,0.92);
        border-radius: 20px;
        padding: 18px 18px 6px 18px;
        margin: 12px auto 18px auto;
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 10px 28px rgba(0,0,0,0.08);
    }

    .main-title {
        font-size: 3.4rem !important;
        font-weight: 900 !important;
        background: linear-gradient(90deg, #667eea, #764ba2, #06D6A0) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-align: center;
        margin-bottom: 0.2rem !important;
        margin-top: 0 !important;
        text-shadow: none !important;
        filter: none !important;
        opacity: 1 !important;
        letter-spacing: 0.5px;
    }

    .sub-title {
        font-size: 1.05rem !important;
        color: #444 !important;
        text-align: center;
        margin-bottom: 1.0rem !important;
        font-weight: 450 !important;
    }

    .divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #06D6A0);
        border-radius: 3px;
        margin: 18px 0;
        opacity: 0.85;
    }

    .metric-card {
        background: white !important;
        border-radius: 16px;
        padding: 18px;
        margin: 8px 0;
        box-shadow: 0 5px 18px rgba(0,0,0,0.08);
        border: 1.5px solid #eaeaea;
        transition: all 0.25s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        border-color: #06D6A0;
    }

    .best-model-card {
        background: linear-gradient(135deg, #06D6A0 0%, #04b586 100%) !important;
        color: white !important;
        border-radius: 16px;
        padding: 18px;
        margin: 8px 0;
        box-shadow: 0 12px 30px rgba(6, 214, 160, 0.25);
        border: 1px solid rgba(255,255,255,0.25);
        animation: pulse 2.2s infinite;
        height: 100%;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(6, 214, 160, 0.32); }
        70% { box-shadow: 0 0 0 10px rgba(6, 214, 160, 0); }
        100% { box-shadow: 0 0 0 0 rgba(6, 214, 160, 0); }
    }

    .badge {
        display: inline-block !important;
        padding: 6px 12px !important;
        border-radius: 999px !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        margin: 2px !important;
        border: 1px solid rgba(0,0,0,0.06);
    }
    .badge-green { background: rgba(6,214,160,0.16) !important; color: #047a5b !important; }
    .badge-blue  { background: rgba(17,138,178,0.16) !important; color: #0a5a73 !important; }
    .badge-red   { background: rgba(239,71,111,0.16) !important; color: #9e1638 !important; }
    .badge-yellow{ background: rgba(255,209,102,0.22) !important; color: #7a5a00 !important; }
    .badge-purple{ background: rgba(118,75,162,0.16) !important; color: #4d2f73 !important; }

    .badge-live { background: rgba(102,126,234,0.16) !important; color: #3b4ed1 !important; }
    .badge-fire { background: rgba(255,154,118,0.20) !important; color: #b33b12 !important; }

    .sentence-card {
        background: white !important;
        border-radius: 12px !important;
        padding: 14px !important;
        margin: 10px 0 !important;
        border-left: 5px solid #06D6A0 !important;
        box-shadow: 0 3px 12px rgba(0,0,0,0.06) !important;
        border: 1px solid #f0f0f0 !important;
    }
    .sentence-card.negative { border-left-color: #EF476F !important; }
    .sentence-card.neutral  { border-left-color: #FFD166 !important; }
    .sentence-card.positive { border-left-color: #06D6A0 !important; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# ENHANCED VADER PIPELINE (Upgraded)
# =========================================================
class EnhancedVADERPipeline:
    """
    Upgrades added:
    - Phrase activation for multi-word lexicon entries
    - True dominance (max |compound*weight|)
    - Robust sarcasm system:
        * sarcasm cues (yeah right / as if)
        * tail-not flip (... not!)
        * positive-clause then tail-not => negative flip
    - Full trace for viva-friendly explanations
    """

    def __init__(self):
        self.sia_base = SentimentIntensityAnalyzer()
        self.sia_enh = SentimentIntensityAnalyzer()

        # Thresholds (from your pipeline)
        self.thresholds = {
            "pos_thr": 0.30,
            "neg_thr": -0.05,
            "strong_neg_thr": -0.25,
            "strong_pos_thr": 0.45,
        }

        # Dominance blend
        self.alpha = 0.70  # 0=weighted avg only, 1=dominant only

        # Phrase replacement registry
        self.phrase_map = {}
        self._load_enhanced_lexicon()

        # Benchmark metrics (dataset-level; used as reference)
        self.benchmark_metrics = {
            "TextBlob": {"Accuracy": 0.502, "Macro F1": 0.471, "Negative F1": 0.349},
            "VADER (Base)": {"Accuracy": 0.540, "Macro F1": 0.530, "Negative F1": 0.485},
            "VADER (Enhanced)": {"Accuracy": 0.556, "Macro F1": 0.542, "Negative F1": 0.488},
        }

        self.color_scheme = {
            "models": {
                "TextBlob": "#EF476F",
                "VADER (Base)": "#118AB2",
                "VADER (Enhanced)": "#06D6A0",
            },
            "sentiments": {
                "negative": "#EF476F",
                "neutral": "#FFD166",
                "positive": "#06D6A0",
            },
        }

    # -------------------------
    # Lexicon phrase activation
    # -------------------------
    def _register_phrases(self, phrase_scores: dict):
        for phrase in phrase_scores.keys():
            if " " in phrase.strip():
                token = phrase.strip().lower().replace(" ", "_")
                pattern = r"(?i)\b" + re.escape(phrase.strip()) + r"\b"
                self.phrase_map[pattern] = f" {token} "

    def _apply_phrase_tokenization(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        # Multi-word phrases -> underscored tokens
        for pattern, repl in self.phrase_map.items():
            text = re.sub(pattern, repl, text)

        # Sarcasm/negation phrases -> tokens
        text = re.sub(r"(?i)\byeah right\b", " yeah_right ", text)
        text = re.sub(r"(?i)\bas if\b", " as_if ", text)
        text = re.sub(r"(?i)\bnot bad\b", " not_bad ", text)
        text = re.sub(r"(?i)\bnot too good\b", " not_too_good ", text)

        return text

    def _load_enhanced_lexicon(self):
        car_lexicon_phrases = {
            "fuel-efficient": 2.5, "fuel efficient": 2.5, "economical": 2.0,
            "overpriced": -3.0, "underpowered": -2.5, "noisy cabin": -2.5,
            "high maintenance": -3.0, "smooth ride": 2.6, "rough ride": -2.6,
            "cheap plastic": -2.4, "premium feel": 2.3, "responsive steering": 2.5,
            "sluggish engine": -3.0, "engine failure": -3.5, "poor reliability": -3.0,
            "frequent breakdown": -3.5, "jerks while shifting": -2.8,
            "body roll": -1.6, "drinks fuel": -2.2, "top speed": 1.0,
            "silent cabin": 2.3,
        }

        finance_lexicon_phrases = {
            "market crashed": -3.5, "market crash": -3.5, "bear market": -2.8,
            "bull market": 2.5, "profit warning": -3.5,
            "earnings beat expectations": 3.0, "missed estimates": -2.5,
            "strong quarter": 2.5, "weak quarter": -2.3,
            "volatile session": -1.8, "record profits": 3.0,
            "surged": 2.0, "plunged": -2.5,
        }

        general_lexicon = {
            "terrible": -3.5, "horrible": -3.2, "awful": -3.2, "sucks": -2.8,
            "unacceptable": -3.0, "dangerous": -3.0, "disaster": -3.2,
            "catastrophic": -3.3, "useless": -3.1, "pathetic": -3.0,
            "amazing": 3.0, "fantastic": 3.0, "brilliant": 2.8,
            "excellent": 2.8, "awesome": 2.5,
        }

        sarcasm_tokens = {
            "yeah_right": -2.3,      # slightly stronger
            "as_if": -2.0,
            "not_bad": 1.5,
            "not_too_good": -1.6,
        }

        # Register phrases for replacement
        self._register_phrases(car_lexicon_phrases)
        self._register_phrases(finance_lexicon_phrases)

        def to_token_lexicon(d):
            out = {}
            for k, v in d.items():
                kk = k.strip().lower()
                if " " in kk:
                    kk = kk.replace(" ", "_")
                out[kk] = v
            return out

        self.sia_enh.lexicon.update(to_token_lexicon(car_lexicon_phrases))
        self.sia_enh.lexicon.update(to_token_lexicon(finance_lexicon_phrases))
        self.sia_enh.lexicon.update({k.lower(): v for k, v in general_lexicon.items()})
        self.sia_enh.lexicon.update(sarcasm_tokens)

    # -------------------------
    # Sentence handling + weights
    # -------------------------
    def _simple_sent_tokenize(self, text: str):
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences if sentences else [text.strip()]

    def _compute_sentence_weight(self, sentence: str) -> float:
        tokens = sentence.split()
        n = len(tokens)

        # Length weight (min 1.0, max 3.0)
        len_weight = min(max(n / 8.0, 1.0), 3.0)

        # Emphasis
        exclam_weight = 1.0 + min(sentence.count("!"), 3) * 0.22
        caps_words = [w for w in tokens if w.isupper() and len(w) > 2]
        caps_weight = 1.0 + min(len(caps_words), 3) * 0.16

        # Contrast cue
        contrast = 1.12 if re.search(r"(?i)\bbut\b|\bhowever\b|\balthough\b", sentence) else 1.0

        return len_weight * exclam_weight * caps_weight * contrast

    # -------------------------
    # Sarcasm detection helpers
    # -------------------------
    def _has_sarcasm_cue(self, text_proc: str) -> bool:
        # tokenized cues
        return ("yeah_right" in text_proc.lower()) or ("as_if" in text_proc.lower())

    def _has_tail_not(self, raw_text: str) -> bool:
        # Matches: "... not!", "... not.", ", not!", "not!" at end
        return bool(re.search(r"(?i)(?:\.\.\.|,)?\s*not[.!?]*\s*$", raw_text.strip()))

    def _has_positive_clause(self, comps: np.ndarray) -> bool:
        return bool((comps >= 0.20).any())

    # -------------------------
    # Baselines
    # -------------------------
    def textblob_predict(self, text, return_scores=False):
        try:
            pol = TextBlob(str(text)).sentiment.polarity
            if pol >= 0.05:
                label = "positive"
            elif pol <= -0.05:
                label = "negative"
            else:
                label = "neutral"
            return (label, {"polarity": float(pol)}) if return_scores else label
        except:
            return ("neutral", {"polarity": 0.0}) if return_scores else "neutral"

    def vader_base_predict(self, text, return_scores=False):
        try:
            scores = self.sia_base.polarity_scores(str(text))
            c = float(scores["compound"])
            if c >= 0.05:
                label = "positive"
            elif c <= -0.05:
                label = "negative"
            else:
                label = "neutral"
            return (label, scores) if return_scores else label
        except:
            fallback = {"compound": 0.0, "neg": 0.0, "neu": 1.0, "pos": 0.0}
            return ("neutral", fallback) if return_scores else "neutral"

    # -------------------------
    # Enhanced VADER
    # -------------------------
    def enhanced_vader_predict(self, text, return_scores=False):
        """
        Decision logic (viva-friendly):
        1) tokenize phrases so multi-word lexicon is active
        2) compute sentence compounds + weights
        3) dominant sentence = max |compound*weight|
        4) final score = alpha*dominant + (1-alpha)*weighted_avg
        5) sarcasm: (tail_not + positive_clause) OR sarcasm cue => flip negative
        """
        try:
            raw_text = str(text)
            text_proc = self._apply_phrase_tokenization(raw_text)
            sentences = self._simple_sent_tokenize(text_proc)

            if not sentences:
                if return_scores:
                    return "neutral", {
                        "final_score": 0.0,
                        "weighted_avg": 0.0,
                        "dominant_sentence_index": 0,
                        "dominant_compound": 0.0,
                        "dominant_weighted_compound": 0.0,
                        "alpha": float(self.alpha),
                        "dominance_rule": "empty_text",
                        "sentence_scores": [],
                        "num_sentences": 0,
                        "sarcasm_badge": False,
                        "sarcasm_reasons": [],
                    }
                return "neutral"

            comps, weights, weighted_comps = [], [], []
            sentence_details = []

            for s in sentences:
                vs = self.sia_enh.polarity_scores(s)
                comp = float(vs["compound"])
                w = float(self._compute_sentence_weight(s))
                wc = comp * w

                comps.append(comp)
                weights.append(w)
                weighted_comps.append(wc)

                sentence_details.append(
                    {
                        "sentence": s if len(s) <= 180 else s[:180] + "...",
                        "compound": comp,
                        "weight": w,
                        "weighted_compound": wc,
                        "scores": vs,
                    }
                )

            comps = np.array(comps, dtype=float)
            weights = np.array(weights, dtype=float)
            weighted_comps = np.array(weighted_comps, dtype=float)

            weighted_avg = float(np.average(comps, weights=weights)) if len(comps) else 0.0
            dom_idx = int(np.argmax(np.abs(weighted_comps))) if len(weighted_comps) else 0
            dominant_comp = float(comps[dom_idx]) if len(comps) else 0.0
            dominant_weighted = float(weighted_comps[dom_idx]) if len(weighted_comps) else 0.0

            dominance_rule = "blend_dominant_vs_weighted_avg"
            sarcasm_reasons = []
            sarcasm_badge = False

            # Start with blend
            final_score = float(self.alpha * dominant_comp + (1 - self.alpha) * weighted_avg)

            # Hard dominance (rare, but for strong clauses)
            if dominant_weighted <= self.thresholds["strong_neg_thr"]:
                final_score = dominant_comp
                dominance_rule = "strong_negative_weighted_dominance"
            elif dominant_weighted >= self.thresholds["strong_pos_thr"]:
                final_score = dominant_comp
                dominance_rule = "strong_positive_weighted_dominance"

            # Apply thresholds
            if final_score >= self.thresholds["pos_thr"]:
                label = "positive"
            elif final_score <= self.thresholds["neg_thr"]:
                label = "negative"
            else:
                label = "neutral"

            # -------------------------
            # Sarcasm upgrade (badge + flip)
            # -------------------------
            has_cue = self._has_sarcasm_cue(text_proc)
            has_tail = self._has_tail_not(raw_text)
            has_pos_clause = self._has_positive_clause(comps)

            if has_cue:
                sarcasm_badge = True
                sarcasm_reasons.append("Sarcasm cue detected (yeah right / as if)")

            if has_tail:
                sarcasm_badge = True
                sarcasm_reasons.append("Sarcasm tail detected (... not!)")

            # Flip rule:
            # If tail-not exists and earlier positive clause exists -> strong negative flip.
            # If sarcasm cue exists and overall is positive/neutral -> flip to negative.
            if has_tail and has_pos_clause:
                flip_mag = max(0.45, abs(dominant_comp), abs(final_score))
                final_score = -flip_mag
                label = "negative"
                dominance_rule = dominance_rule + " + sarcasm_tail_flip"
                sarcasm_reasons.append("Flip applied: positive clause + tail-not")

            elif has_cue and label in ["positive", "neutral"]:
                # softer flip if only cue exists
                flip_mag = max(0.30, abs(final_score))
                final_score = -flip_mag
                label = "negative"
                dominance_rule = dominance_rule + " + sarcasm_cue_flip"
                sarcasm_reasons.append("Flip applied: sarcasm cue overrides")

            if return_scores:
                return label, {
                    "final_score": float(final_score),
                    "weighted_avg": float(weighted_avg),
                    "dominant_sentence_index": int(dom_idx),
                    "dominant_compound": float(dominant_comp),
                    "dominant_weighted_compound": float(dominant_weighted),
                    "alpha": float(self.alpha),
                    "dominance_rule": dominance_rule,
                    "sentence_scores": sentence_details,
                    "num_sentences": int(len(sentences)),
                    "sarcasm_badge": bool(sarcasm_badge),
                    "sarcasm_reasons": sarcasm_reasons,
                    "raw_text": raw_text,
                    "text_proc": text_proc,
                }

            return label

        except Exception:
            return self.vader_base_predict(text, return_scores=return_scores)

    def analyze_text(self, text, return_detailed=False):
        tb_label, tb_scores = self.textblob_predict(text, return_scores=True)
        vb_label, vb_scores = self.vader_base_predict(text, return_scores=True)
        ve_label, ve_scores = self.enhanced_vader_predict(text, return_scores=True)

        res = {
            "text": text[:220] + "..." if len(str(text)) > 220 else text,
            "TextBlob": tb_label,
            "VADER_Base": vb_label,
            "VADER_Enhanced": ve_label,
            "textblob_score": float(tb_scores.get("polarity", 0.0)),
            "vader_base_score": float(vb_scores.get("compound", 0.0)),
            "vader_enhanced_score": float(ve_scores.get("final_score", 0.0)),
        }

        if return_detailed:
            res.update(
                {
                    "textblob_details": tb_scores,
                    "vader_base_details": vb_scores,
                    "vader_enhanced_details": ve_scores,
                }
            )
        return res


# =========================================================
# METRICS (for LIVE evaluation if user provides expected label)
# =========================================================
LABELS = ["negative", "neutral", "positive"]

def negative_f1_multiclass(y_true, y_pred):
    """
    Compute F1 for the 'negative' class in a multiclass setting.
    Returns 0 if not computable.
    """
    try:
        # average=None gives per-class F1 in the order of LABELS
        f1s = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
        idx = LABELS.index("negative")
        return float(f1s[idx])
    except Exception:
        return 0.0

def compute_live_metrics(df_eval: pd.DataFrame):
    """
    df_eval columns: expected, pred_textblob, pred_vader_base, pred_vader_enh
    returns dict of metrics per model
    """
    out = {}
    if df_eval is None or len(df_eval) == 0:
        return out

    y_true = df_eval["expected"].tolist()

    for model_key, col in [
        ("TextBlob", "pred_textblob"),
        ("VADER (Base)", "pred_vader_base"),
        ("VADER (Enhanced)", "pred_vader_enh"),
    ]:
        y_pred = df_eval[col].tolist()
        acc = float(accuracy_score(y_true, y_pred))
        macro = float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))
        negf1 = negative_f1_multiclass(y_true, y_pred)
        out[model_key] = {"Accuracy": acc, "Macro F1": macro, "Negative F1": negf1}

    return out


# =========================================================
# UI HELPERS
# =========================================================
def create_header_shell(max_width=True):
    if max_width:
        st.markdown("<div class='app-shell'>", unsafe_allow_html=True)

def close_header_shell(max_width=True):
    if max_width:
        st.markdown("</div>", unsafe_allow_html=True)

def create_wow_header(presenter_mode=False):
    st.markdown("<h1 class='main-title'>🚀 ENHANCED VADER</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-title'>Multi-Domain Lexicon Intelligence · Sentence Dominance · Sarcasm Handling · Explainable Decisions</p>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 1.6])
    with c1:
        st.markdown('<span class="badge badge-purple">🧠 Explainable</span>', unsafe_allow_html=True)
    with c2:
        st.markdown('<span class="badge badge-blue">⚡ Phrase-Aware</span>', unsafe_allow_html=True)
    with c3:
        st.markdown('<span class="badge badge-green">🧾 Dominance</span>', unsafe_allow_html=True)
    with c4:
        st.markdown('<span class="badge badge-yellow">🎛 Tuned Thresholds</span>', unsafe_allow_html=True)
    with c5:
        st.markdown('<span class="badge badge-live">📊 Live + Benchmark + Live-Eval</span>', unsafe_allow_html=True)

    if presenter_mode:
        st.info("Presenter Mode is ON: cleaner layout, focused storytelling for viva.")


def sentiment_badge(label: str):
    if label == "positive":
        return '<span class="badge badge-green">POSITIVE</span>'
    if label == "negative":
        return '<span class="badge badge-red">NEGATIVE</span>'
    return '<span class="badge badge-yellow">NEUTRAL</span>'


def create_sentence_breakdown(sentence_details, dominant_index: int):
    for i, sent in enumerate(sentence_details, 1):
        score = sent["compound"]
        w = sent["weight"]
        wc = sent["weighted_compound"]

        sentiment = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
        card_class = f"sentence-card {sentiment}"

        is_dom = (i - 1) == dominant_index
        dom_tag = " ✅ DOMINANT" if is_dom else ""

        st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Sentence {i}{dom_tag}:** {sent['sentence']}")
        with col2:
            st.markdown(sentiment_badge(sentiment), unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.write(f"**Compound:** {score:.3f}")
        with c2:
            st.write(f"**Weight:** {w:.2f}×")
        with c3:
            st.write(f"**Weighted:** {wc:.3f}")

        cols = st.columns(4)
        with cols[0]:
            st.metric("Neg", f"{sent['scores']['neg']:.3f}")
        with cols[1]:
            st.metric("Neu", f"{sent['scores']['neu']:.3f}")
        with cols[2]:
            st.metric("Pos", f"{sent['scores']['pos']:.3f}")
        with cols[3]:
            st.metric("Comp", f"{sent['scores']['compound']:.3f}")

        st.markdown("</div>", unsafe_allow_html=True)


def live_score_chart(result, analyzer):
    st.markdown("### 📌 Live Score Comparison (Current Text)")
    df = pd.DataFrame(
        {
            "Model": ["TextBlob", "VADER (Base)", "VADER (Enhanced)"],
            "Score": [
                result["textblob_score"],
                result["vader_base_score"],
                result["vader_enhanced_score"],
            ],
        }
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=df["Model"],
                y=df["Score"],
                text=[f"{v:.3f}" for v in df["Score"]],
                textposition="auto",
                marker_color=[
                    analyzer.color_scheme["models"]["TextBlob"],
                    analyzer.color_scheme["models"]["VADER (Base)"],
                    analyzer.color_scheme["models"]["VADER (Enhanced)"],
                ],
            )
        ]
    )
    fig.update_layout(
        title="Live Polarity / Compound Score (Updates Per Input)",
        yaxis_title="Score (TextBlob polarity, VADER compound)",
        yaxis_range=[-1, 1],
        height=360,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


def benchmark_metrics_chart(analyzer):
    st.markdown("### 🏁 Benchmark Performance (Pipeline Results)")
    st.caption("Dataset-level benchmark (e.g., n≈5,055). Shown here as a reference during live demo.")
    df = pd.DataFrame([{ "Model": m, **vals } for m, vals in analyzer.benchmark_metrics.items()])

    fig = go.Figure()
    metrics = ["Accuracy", "Macro F1", "Negative F1"]
    colors = ["#4ECDC4", "#FF6B6B", "#95E1D3"]

    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric,
            x=df["Model"],
            y=df[metric],
            text=[f"{v:.3f}" for v in df[metric]],
            textposition="auto",
            marker_color=color,
        ))

    fig.update_layout(
        barmode="group",
        height=420,
        title="Accuracy, Macro-F1, Negative-F1 (Benchmark)",
        yaxis_title="Score",
        yaxis_range=[0, 0.7],
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, use_container_width=True)


def live_eval_metrics_chart(metrics_dict, analyzer, title="📈 Live Evaluation Metrics (Your Test Set)"):
    """
    metrics_dict = {model: {Accuracy, Macro F1, Negative F1}}
    """
    if not metrics_dict:
        st.info("Add a few labelled test examples (Expected Label) to see LIVE accuracy/F1 charts.")
        return

    df = pd.DataFrame([
        {"Model": m, **vals}
        for m, vals in metrics_dict.items()
    ])

    fig = go.Figure()
    metrics = ["Accuracy", "Macro F1", "Negative F1"]
    colors = ["#4ECDC4", "#FF6B6B", "#95E1D3"]

    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric,
            x=df["Model"],
            y=df[metric],
            text=[f"{v:.3f}" for v in df[metric]],
            textposition="auto",
            marker_color=color,
        ))

    fig.update_layout(
        barmode="group",
        height=420,
        title=title,
        yaxis_title="Score",
        yaxis_range=[0, 1],
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, use_container_width=True)


def explainability_panel(result):
    details = result.get("vader_enhanced_details", {}) or {}
    if not details:
        return

    final_score = float(details.get("final_score", 0.0))
    weighted_avg = float(details.get("weighted_avg", 0.0))
    dom_idx = int(details.get("dominant_sentence_index", 0))
    dom_comp = float(details.get("dominant_compound", 0.0))
    dom_wcomp = float(details.get("dominant_weighted_compound", 0.0))
    alpha = float(details.get("alpha", 0.7))
    rule = details.get("dominance_rule", "unknown")

    sarcasm_badge = bool(details.get("sarcasm_badge", False))
    sarcasm_reasons = details.get("sarcasm_reasons", [])

    st.markdown("### 🔬 Enhanced VADER Explainability (Author View)")
    cols = st.columns(5)
    cols[0].metric("Final Score", f"{final_score:.3f}")
    cols[1].metric("Weighted Avg", f"{weighted_avg:.3f}")
    cols[2].metric("Dominant Comp", f"{dom_comp:.3f}")
    cols[3].metric("Dominant Weighted", f"{dom_wcomp:.3f}")
    cols[4].metric("Alpha", f"{alpha:.2f}")

    if sarcasm_badge:
        st.markdown('<span class="badge badge-fire">🔥 Sarcasm Triggered</span>', unsafe_allow_html=True)
        for r in sarcasm_reasons[:3]:
            st.write(f"• {r}")
    else:
        st.markdown('<span class="badge badge-blue">No sarcasm rule triggered</span>', unsafe_allow_html=True)

    st.info(f"**Rule Path:** `{rule}`  |  **Dominant Sentence:** {dom_idx + 1} (1-based)")

    with st.expander("Show thresholds + tokenized text (debug/validation)", expanded=False):
        st.json({"thresholds": details.get("thresholds", None) or "see code thresholds"})
        st.write("**Tokenized Text (phrases + sarcasm tokens):**")
        st.code(details.get("text_proc", ""), language="text")


# =========================================================
# SESSION STATE INIT
# =========================================================
if "eval_rows" not in st.session_state:
    st.session_state.eval_rows = []  # list of dicts

if "history" not in st.session_state:
    st.session_state.history = []  # store recent analyses (optional)


# =========================================================
# TABS
# =========================================================
def create_single_analysis_tab(analyzer: EnhancedVADERPipeline, presenter_mode=False):
    st.markdown("## 🔍 Live Sentiment Analysis")
    st.markdown("---")

    examples = {
        "🎯 Select an example...": "",
        "🐦 Sarcasm Tail Not": "Yeah right, like this product is gonna last more than a week. Amazing quality... not!",
        "🚗 Car Review (Mixed)": "The engine performance is absolutely terrible and unreliable. However, the seats are surprisingly comfortable and the fuel economy is excellent.",
        "💰 Finance News (Complex)": "Market crashed by 15% today due to economic concerns. However, analysts remain optimistic about long-term recovery prospects.",
        "😠 Strong Negative": "This is the worst service I've ever experienced. Absolutely unacceptable and a complete waste of money!",
        "😊 Strong Positive": "Absolutely fantastic product! Exceeded all expectations and the customer service was brilliant!",
    }

    left, right = st.columns([2.2, 1.2])

    with left:
        selected_example = st.selectbox("Choose an example text:", list(examples.keys()))
        text = st.text_area(
            "**Enter your text for analysis:**",
            value=examples[selected_example],
            height=180 if not presenter_mode else 210,
            placeholder="Type or paste your text here...",
        )

        # Controls
        st.markdown("### ⚙️ Controls (Viva-friendly)")
        c1, c2 = st.columns(2)
        with c1:
            analyzer.alpha = st.slider(
                "Sentence dominance strength (alpha)",
                min_value=0.0,
                max_value=1.0,
                value=float(analyzer.alpha),
                step=0.05,
                help="0 = pure weighted average; 1 = pure dominant sentence. Recommended: 0.65–0.80",
            )
        with c2:
            show_debug = st.toggle("Show debug/validation info", value=False)

    with right:
        st.markdown("### 🧭 What makes Enhanced VADER different?")
        st.write("✅ Phrase-aware lexicon (multi-word terms activate)")
        st.write("✅ Sentence dominance + emphasis weighting")
        st.write("✅ Sarcasm handling (cues + tail-not flip)")
        st.write("✅ Explainability trace (what fired & why)")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Live evaluation helper
        st.markdown("### 🧪 Live Evaluation (for your supervisor)")
        st.write("Optional: select the expected label, then add to evaluation set.")
        expected = st.selectbox("Expected Label (optional)", ["(skip)"] + LABELS)
        add_eval = st.button("➕ Add this test to Live Evaluation Set", use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    analyze_btn = st.button("🚀 ANALYZE SENTIMENT", type="primary", use_container_width=True)
    if not analyze_btn:
        # Show charts even when not analyzing, to look professional and complete
        if not presenter_mode:
            benchmark_metrics_chart(analyzer)
        return

    if not text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
        return

    with st.spinner("🤖 Analyzing..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.005)
            progress_bar.progress(i + 1)
        result = analyzer.analyze_text(text, return_detailed=True)

    # Store to history
    st.session_state.history.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "text": text,
            "TextBlob": result["TextBlob"],
            "VADER_Base": result["VADER_Base"],
            "VADER_Enhanced": result["VADER_Enhanced"],
            "enh_score": result["vader_enhanced_score"],
        }
    )
    st.session_state.history = st.session_state.history[-30:]

    # Cards
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 📊 TextBlob")
        st.markdown(f"<div style='display:flex; justify-content:center;'>{sentiment_badge(result['TextBlob'])}</div>", unsafe_allow_html=True)
        st.metric("Polarity", f"{result['textblob_score']:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 📊 VADER (Base)")
        st.markdown(f"<div style='display:flex; justify-content:center;'>{sentiment_badge(result['VADER_Base'])}</div>", unsafe_allow_html=True)
        st.metric("Compound", f"{result['vader_base_score']:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="best-model-card">', unsafe_allow_html=True)
        st.markdown("### 🏆 VADER (Enhanced)")
        st.markdown(f"<div style='display:flex; justify-content:center;'>{sentiment_badge(result['VADER_Enhanced'])}</div>", unsafe_allow_html=True)
        st.metric("Final Score", f"{result['vader_enhanced_score']:.3f}")
        st.markdown(f"<div style='opacity:0.95;'>Alpha: <b>{analyzer.alpha:.2f}</b></div>", unsafe_allow_html=True)

        # Sarcasm badge in top card
        details = result.get("vader_enhanced_details", {}) or {}
        if details.get("sarcasm_badge", False):
            st.markdown('<span class="badge badge-fire">🔥 Sarcasm Triggered</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Charts
    live_score_chart(result, analyzer)

    if not presenter_mode:
        benchmark_metrics_chart(analyzer)

    # Explainability panel
    explainability_panel(result)

    # Sentence breakdown
    details = result.get("vader_enhanced_details", {}) or {}
    sentence_scores = details.get("sentence_scores", [])
    dom_idx = int(details.get("dominant_sentence_index", 0))

    with st.expander("🧾 Sentence-Level Breakdown (click to open)", expanded=presenter_mode):
        if sentence_scores:
            create_sentence_breakdown(sentence_scores, dominant_index=dom_idx)
        else:
            st.info("No sentence breakdown available.")

    # Add to evaluation set
    if add_eval and expected in LABELS:
        st.session_state.eval_rows.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "text": text,
                "expected": expected,
                "pred_textblob": result["TextBlob"],
                "pred_vader_base": result["VADER_Base"],
                "pred_vader_enh": result["VADER_Enhanced"],
            }
        )
        st.success("Added to live evaluation set ✅")

    # Live evaluation metrics (dynamic; computed from your labelled tests)
    st.markdown("## 📈 Live Evaluation Metrics (from labelled tests)")
    df_eval = pd.DataFrame(st.session_state.eval_rows) if len(st.session_state.eval_rows) else pd.DataFrame()
    if len(df_eval):
        metrics_dict = compute_live_metrics(df_eval)
        live_eval_metrics_chart(metrics_dict, analyzer, title="Live Accuracy, Macro-F1, Negative-F1 (Your Test Set)")

        with st.expander("See evaluation table + confusion matrix", expanded=False):
            st.dataframe(df_eval, use_container_width=True)

            # Confusion matrix for Enhanced (as the focus)
            y_true = df_eval["expected"].tolist()
            y_pred = df_eval["pred_vader_enh"].tolist()
            cm = confusion_matrix(y_true, y_pred, labels=LABELS)
            cm_df = pd.DataFrame(cm, index=[f"True:{l}" for l in LABELS], columns=[f"Pred:{l}" for l in LABELS])
            st.write("**Confusion Matrix — Enhanced VADER (Live Eval)**")
            st.dataframe(cm_df, use_container_width=True)
    else:
        st.info("No labelled tests yet. Select Expected Label and add a few examples to compute live Accuracy/F1.")

    # Optional debug view
    if show_debug:
        st.markdown("## 🧪 Debug / Validation")
        st.write("**Raw text:**")
        st.code(details.get("raw_text", ""), language="text")
        st.write("**Tokenized text:**")
        st.code(details.get("text_proc", ""), language="text")
        st.write("**Rule Path:**", details.get("dominance_rule", "unknown"))


def create_batch_analysis_tab(analyzer: EnhancedVADERPipeline):
    st.markdown("## 📊 Batch File Analysis")
    st.markdown("---")

    st.info(
        "Upload a CSV/TXT for batch predictions.\n\n"
        "If you include `gold_label` (negative/neutral/positive), the app computes metrics on your uploaded file."
    )

    uploaded_file = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"])
    if not uploaded_file:
        return

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            content = StringIO(uploaded_file.getvalue().decode("utf-8"))
            lines = [line.strip() for line in content if line.strip()]
            df = pd.DataFrame({"text": lines})

        st.success(f"✅ Loaded {len(df):,} records.")
        st.dataframe(df.head(10), use_container_width=True)

        text_col = st.selectbox("Select text column:", df.columns.tolist(), index=0)
        has_gold = "gold_label" in df.columns

        if has_gold:
            st.success("✅ Detected `gold_label` — full metrics will be computed.")
        else:
            st.warning("No `gold_label` column — predictions only.")

        if st.button("🚀 ANALYZE BATCH", type="primary", use_container_width=True):
            results = []
            progress = st.progress(0)
            texts = df[text_col].astype(str).tolist()

            for i, t in enumerate(texts):
                results.append(analyzer.analyze_text(t))
                progress.progress((i + 1) / len(texts))

            out = pd.DataFrame(results)
            st.success("🎉 Batch analysis complete.")
            st.dataframe(out.head(50), use_container_width=True)

            # Metrics if gold exists
            if has_gold:
                gold = df["gold_label"].astype(str).str.lower().str.strip()
                gold = gold.replace({"pos": "positive", "neg": "negative", "neu": "neutral"})
                valid = gold.isin(LABELS)
                gold = gold[valid]
                out_valid = out.loc[valid].copy()

                # Compute metrics for each model
                perf = {}
                for model_key, col in [
                    ("TextBlob", "TextBlob"),
                    ("VADER (Base)", "VADER_Base"),
                    ("VADER (Enhanced)", "VADER_Enhanced"),
                ]:
                    y_true = gold.tolist()
                    y_pred = out_valid[col].tolist()
                    perf[model_key] = {
                        "Accuracy": float(accuracy_score(y_true, y_pred)),
                        "Macro F1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
                        "Negative F1": negative_f1_multiclass(y_true, y_pred),
                    }

                st.markdown("### 📈 Batch Metrics (Your Uploaded Data)")
                live_eval_metrics_chart(perf, analyzer, title="Batch Accuracy, Macro-F1, Negative-F1 (Your File)")

            csv = out.to_csv(index=False)
            st.download_button(
                label="⬇️ DOWNLOAD RESULTS CSV",
                data=csv,
                file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def create_performance_tab(analyzer: EnhancedVADERPipeline):
    st.markdown("## 📈 Performance Metrics (Reference)")
    st.markdown("---")
    benchmark_metrics_chart(analyzer)

    st.markdown("### 🧠 Why improvements can be small but meaningful")
    st.write("• Small % gains can be statistically meaningful on large test sets.")
    st.write("• Negative F1 improvements matter most when the baseline fails on weak-negative and sarcasm cases.")
    st.write("• The key contribution here is interpretable improvement, not brute-force black-box accuracy.")


# =========================================================
# MAIN
# =========================================================
def main():
    analyzer = EnhancedVADERPipeline()

    # Sidebar: Viva controls + layout optimization
    with st.sidebar:
        st.markdown("## ⚙️ Viva Controls")
        presenter_mode = st.toggle("Presenter Mode", value=True)
        max_width = st.toggle("Max-Width Layout (recommended)", value=True)

        st.markdown("---")
        st.markdown("## 🧪 Live Eval Tools")
        if st.button("🧹 Clear Live Evaluation Set", use_container_width=True):
            st.session_state.eval_rows = []
            st.success("Cleared.")

        if st.button("🕘 Clear History", use_container_width=True):
            st.session_state.history = []
            st.success("Cleared.")

        st.markdown("---")
        st.markdown("## ℹ️ Notes")
        st.write("Accuracy/F1 require labelled samples. Use Live Eval to compute them during demo.")

    create_header_shell(max_width=max_width)
    create_wow_header(presenter_mode=presenter_mode)
    close_header_shell(max_width=max_width)

    tab1, tab2, tab3 = st.tabs(["🔍 Live Analysis", "📊 Batch Analysis", "📈 Performance"])

    with tab1:
        create_single_analysis_tab(analyzer, presenter_mode=presenter_mode)

        # Optional: show a small, neat history panel (good for viva storytelling)
        with st.expander("🕘 Recent Analyses (for demo narrative)", expanded=False):
            if st.session_state.history:
                st.dataframe(pd.DataFrame(st.session_state.history[::-1]).head(12), use_container_width=True)
            else:
                st.info("No history yet. Run a few examples during your demo.")

    with tab2:
        create_batch_analysis_tab(analyzer)

    with tab3:
        create_performance_tab(analyzer)


if __name__ == "__main__":
    main()
