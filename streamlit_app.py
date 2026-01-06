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
# CUSTOM CSS (kept from your version)
# =========================================================
st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
        padding-top: 0 !important;
    }
    .main-title {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #667eea, #764ba2, #06D6A0) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
        padding-top: 0 !important;
        margin-top: 0 !important;
        text-shadow: none !important;
        filter: none !important;
        opacity: 1 !important;
    }
    .sub-title {
        font-size: 1.2rem !important;
        color: #555 !important;
        text-align: center;
        margin-bottom: 2rem !important;
        font-weight: 400 !important;
    }
    .metric-card {
        background: white !important;
        border-radius: 15px;
        padding: 25px;
        margin: 10px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        border: 2px solid #e8e8e8;
        transition: all 0.3s ease;
        height: 100%;
        position: relative;
        z-index: 1;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        border-color: #06D6A0;
    }
    .best-model-card {
        background: linear-gradient(135deg, #06D6A0 0%, #04b586 100%) !important;
        color: white !important;
        border-radius: 15px;
        padding: 25px;
        margin: 10px;
        box-shadow: 0 10px 30px rgba(6, 214, 160, 0.25);
        border: 2px solid #04b586;
        animation: pulse 2s infinite;
        height: 100%;
        position: relative;
        z-index: 1;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(6, 214, 160, 0.3); }
        70% { box-shadow: 0 0 0 10px rgba(6, 214, 160, 0); }
        100% { box-shadow: 0 0 0 0 rgba(6, 214, 160, 0); }
    }
    .divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #06D6A0);
        border-radius: 3px;
        margin: 30px 0;
        opacity: 0.8;
    }
    .badge {
        display: inline-block !important;
        padding: 4px 12px !important;
        border-radius: 20px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        margin: 2px !important;
        position: relative !important;
        z-index: 1 !important;
    }
    .badge-green { background: linear-gradient(135deg, #06D6A0 0%, #04b586 100%) !important; color: white !important; border: 1px solid #04b586 !important; }
    .badge-blue  { background: linear-gradient(135deg, #118AB2 0%, #0a6d8e 100%) !important; color: white !important; border: 1px solid #0a6d8e !important; }
    .badge-red   { background: linear-gradient(135deg, #EF476F 0%, #e6305a 100%) !important; color: white !important; border: 1px solid #e6305a !important; }
    .badge-yellow{ background: linear-gradient(135deg, #FFD166 0%, #f9c74f 100%) !important; color: #333 !important; border: 1px solid #f9c74f !important; }
    .badge-purple{ background: linear-gradient(135deg, #764ba2 0%, #5d3a7e 100%) !important; color: white !important; border: 1px solid #5d3a7e !important; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sentence cards */
    .sentence-card {
        background: white !important;
        border-radius: 10px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
        border-left: 4px solid #06D6A0 !important;
        box-shadow: 0 3px 10px rgba(0,0,0,0.06) !important;
        border: 1px solid #f0f0f0 !important;
    }
    .sentence-card.negative { border-left-color: #EF476F !important; }
    .sentence-card.neutral  { border-left-color: #FFD166 !important; }
    .sentence-card.positive { border-left-color: #06D6A0 !important; }

    .legend-container {
        background: white !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #e8e8e8;
        box-shadow: 0 2px 5px rgba(0,0,0,0.04);
    }
    .legend-title {
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
        font-size: 1.1rem;
    }
    .legend-item {
        display: flex;
        align-items: center;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 4px;
        margin-right: 10px;
        border: 1px solid #ddd;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# ENHANCED VADER (FIXED): phrase activation + true dominance + debuggable trace
# =========================================================
class EnhancedVADERPipeline:
    """
    Fixes:
    1) Multi-word lexicon activation via phrase -> underscore tokenization
    2) True dominance driven by WEIGHTED sentence polarity (compound * weight)
    3) Live charts show per-text scores (not static dataset metrics)
    """

    def __init__(self):
        self.sia_base = SentimentIntensityAnalyzer()
        self.sia_enh = SentimentIntensityAnalyzer()

        # Thresholds from your pipeline
        self.thresholds = {
            "pos_thr": 0.30,
            "neg_thr": -0.05,
            "strong_neg_thr": -0.25,
            "strong_pos_thr": 0.45,
        }

        # Dominance blend (0=weighted avg only, 1=dominant sentence only)
        self.alpha = 0.70

        # Phrase replacement registry
        self.phrase_map = {}
        self._load_enhanced_lexicon()

        # Colors (kept)
        self.color_scheme = {
            "models": {
                "TextBlob": "#EF476F",
                "VADER (Base)": "#118AB2",
                "VADER (Enhanced)": "#06D6A0",
            },
            "metrics": {
                "Accuracy": "#4ECDC4",
                "Macro F1": "#FF6B6B",
                "Negative F1": "#95E1D3",
                "Positive F1": "#FFD166",
            },
            "sentiments": {
                "negative": "#EF476F",
                "neutral": "#FFD166",
                "positive": "#06D6A0",
            },
        }

    # -------------------------
    # Phrase activation utilities
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

        # Multi-word phrase -> underscored token
        for pattern, repl in self.phrase_map.items():
            text = re.sub(pattern, repl, text)

        # Sarcasm/negation phrases -> tokens
        text = re.sub(r"(?i)\byeah right\b", " yeah_right ", text)
        text = re.sub(r"(?i)\bas if\b", " as_if ", text)
        text = re.sub(r"(?i)\bnot bad\b", " not_bad ", text)
        text = re.sub(r"(?i)\bnot too good\b", " not_too_good ", text)

        return text

    def _load_enhanced_lexicon(self):
        # Your lexicons (same content, we will convert phrases to underscore keys)
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
            "yeah_right": -2.0,
            "as_if": -1.8,
            "not_bad": 1.5,
            "not_too_good": -1.5,
        }

        # Register multi-word phrases for replacement in text
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

        car_lexicon = to_token_lexicon(car_lexicon_phrases)
        finance_lexicon = to_token_lexicon(finance_lexicon_phrases)

        # Update enhanced lexicon
        self.sia_enh.lexicon.update(car_lexicon)
        self.sia_enh.lexicon.update(finance_lexicon)
        self.sia_enh.lexicon.update({k.lower(): v for k, v in general_lexicon.items()})
        self.sia_enh.lexicon.update(sarcasm_tokens)

    # -------------------------
    # Sentence handling + weighting
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

        # Length weight: 1..3
        len_weight = min(max(n / 8.0, 1.0), 3.0)

        exclam_weight = 1.0 + min(sentence.count("!"), 3) * 0.20
        caps_words = [w for w in tokens if w.isupper() and len(w) > 2]
        caps_weight = 1.0 + min(len(caps_words), 3) * 0.15

        # Contrast cue
        contrast = 1.10 if re.search(r"(?i)\bbut\b|\bhowever\b", sentence) else 1.0

        return len_weight * exclam_weight * caps_weight * contrast

    # -------------------------
    # Predictions
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

    def enhanced_vader_predict(self, text, return_scores=False):
        """
        True dominance:
        - Phrase tokenization so multi-word lexicon activates
        - Dominant sentence chosen by max |compound * weight|
        - Final score = alpha*dominant_compound + (1-alpha)*weighted_avg
        - Strong dominance triggers use WEIGHTED dominance
        """
        try:
            text_proc = self._apply_phrase_tokenization(str(text))
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
                        "sentence": s if len(s) <= 160 else s[:160] + "...",
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

            # Strong dominance based on WEIGHTED dominance
            if dominant_weighted <= self.thresholds["strong_neg_thr"]:
                final_score = dominant_comp
                label = "negative"
                dominance_rule = "strong_negative_weighted_dominance"
            elif dominant_weighted >= self.thresholds["strong_pos_thr"]:
                final_score = dominant_comp
                label = "positive"
                dominance_rule = "strong_positive_weighted_dominance"
            else:
                final_score = float(self.alpha * dominant_comp + (1 - self.alpha) * weighted_avg)

                if final_score >= self.thresholds["pos_thr"]:
                    label = "positive"
                elif final_score <= self.thresholds["neg_thr"]:
                    label = "negative"
                else:
                    label = "neutral"

            # Tail "not!" rule (lightweight)
            if re.search(r"(?i)\bnot[.!?]*\s*$", text_proc) and label == "positive":
                final_score = min(final_score, 0.0)
                label = "neutral" if final_score > self.thresholds["neg_thr"] else "negative"
                dominance_rule = dominance_rule + " + tail_not_rule"

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
                }

            return label

        except Exception:
            return self.vader_base_predict(text, return_scores=return_scores)

    def analyze_text(self, text, return_detailed=False):
        tb_label, tb_scores = self.textblob_predict(text, return_scores=True)
        vb_label, vb_scores = self.vader_base_predict(text, return_scores=True)
        ve_label, ve_scores = self.enhanced_vader_predict(text, return_scores=True)

        res = {
            "text": text[:200] + "..." if len(str(text)) > 200 else text,
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
# UI HELPERS
# =========================================================
def create_wow_header():
    st.markdown("<h1 class='main-title'>🚀 ENHANCED VADER</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Advanced Multi-Domain Sentiment Analysis</p>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="badge badge-purple">🧠 Explainable Rules</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="badge badge-green">⚡ Phrase-aware Lexicon</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="badge badge-blue">🧾 Sentence Dominance</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="badge badge-red">📈 Live Score Charts</div>', unsafe_allow_html=True)


def create_unified_legend(analyzer, chart_type="model_comparison"):
    if chart_type == "model_comparison":
        st.markdown("### 🤖 Model Legend")
        st.markdown('<div class="legend-container">', unsafe_allow_html=True)
        st.markdown('<div class="legend-title">Model Identification</div>', unsafe_allow_html=True)
        models = analyzer.color_scheme["models"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {models['TextBlob']};"></div>
                    <span><strong>TextBlob</strong></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {models['VADER (Base)']};"></div>
                    <span><strong>VADER (Base)</strong></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {models['VADER (Enhanced)']};"></div>
                    <span><strong>VADER (Enhanced)</strong></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


def create_sentence_breakdown(sentence_details, analyzer):
    for i, sent in enumerate(sentence_details, 1):
        score = sent["compound"]
        w = sent["weight"]
        wc = sent["weighted_compound"]

        sentiment = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
        card_class = f"sentence-card {sentiment}"
        badge_color = "green" if sentiment == "positive" else "red" if sentiment == "negative" else "yellow"

        st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Sentence {i}:** {sent['sentence']}")
        with col2:
            st.markdown(f"<span class='badge badge-{badge_color}'>{sentiment.upper()}</span>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.write(f"**Compound:** {score:.3f}")
        with c2:
            st.write(f"**Weight:** {w:.2f}×")
        with c3:
            st.write(f"**Weighted:** {wc:.3f}")

        # Detailed scores
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


def create_real_time_explanation(result, analyzer):
    details = result.get("vader_enhanced_details", {}) or {}
    final_score = float(details.get("final_score", 0.0))
    weighted_avg = float(details.get("weighted_avg", 0.0))
    dom_idx = int(details.get("dominant_sentence_index", 0))
    dom_comp = float(details.get("dominant_compound", 0.0))
    dom_wcomp = float(details.get("dominant_weighted_compound", 0.0))
    dominance_rule = details.get("dominance_rule", "unknown")

    st.markdown("## 🔬 Real-Time Enhanced VADER Decision Trace")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Final Score", f"{final_score:.3f}")
    with c2:
        st.metric("Weighted Avg", f"{weighted_avg:.3f}")
    with c3:
        st.metric("Dominant Comp", f"{dom_comp:.3f}")
    with c4:
        st.metric("Dominant Weighted", f"{dom_wcomp:.3f}")

    st.info(f"**Dominance Rule Applied:** `{dominance_rule}`")
    st.write(f"**Dominant Sentence Index:** {dom_idx + 1} (1-based)")

    # Thresholds shown
    with st.expander("🎛 Thresholds Used", expanded=False):
        st.json(analyzer.thresholds)


# =========================================================
# TABS
# =========================================================
def create_single_analysis_tab(analyzer: EnhancedVADERPipeline):
    st.markdown("## 🔍 Live Sentiment Analysis")
    st.markdown("---")

    examples = {
        "🎯 Select an example...": "",
        "🚗 Car Review (Mixed)": "The engine performance is absolutely terrible and unreliable. However, the seats are surprisingly comfortable and the fuel economy is excellent.",
        "💰 Finance News (Complex)": "Market crashed by 15% today due to economic concerns. However, analysts remain optimistic about long-term recovery prospects.",
        "🐦 Twitter (Sarcastic)": "Yeah right, like this product is gonna last more than a week. Amazing quality... not!",
        "😠 Strong Negative": "This is the worst service I've ever experienced. Absolutely unacceptable and a complete waste of money!",
        "😊 Strong Positive": "Absolutely fantastic product! Exceeded all expectations and the customer service was brilliant!",
        "🧠 Long Complex": "While the initial design and build quality are exceptional with premium materials used throughout, the software interface is frustratingly counter-intuitive and the battery life, though advertised as all-day, barely lasts through a morning of moderate use, which is disappointing given the high price point.",
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_example = st.selectbox("Choose an example text:", list(examples.keys()))
        text = st.text_area(
            "**Enter your text for analysis:**",
            value=examples[selected_example],
            height=160,
            placeholder="Type or paste your text here...",
        )

        st.markdown("### ⚙️ Enhanced VADER Controls")
        analyzer.alpha = st.slider(
            "Sentence dominance strength (alpha)",
            min_value=0.0,
            max_value=1.0,
            value=float(analyzer.alpha),
            step=0.05,
            help="0 = pure weighted average; 1 = pure dominant sentence. Recommended: 0.65–0.80",
        )

    with col2:
        st.markdown("### 🧭 What changes in Live mode?")
        st.write("- **Scores** update per text (dynamic).")
        st.write("- **Dataset metrics** are only in the Performance tab (static by design).")
        st.write("- Enhanced lexicon now **activates phrases** like `smooth ride`, `market crashed`.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.button("🚀 **ANALYZE SENTIMENT**", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
            return

        with st.spinner("🤖 Analyzing..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.006)
                progress_bar.progress(i + 1)

            result = analyzer.analyze_text(text, return_detailed=True)

        # Cards
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 📊 TextBlob")
            st.markdown(
                f"<h1 style='color:{analyzer.color_scheme['models']['TextBlob']}; font-size:3rem; text-align:center;'>{result['TextBlob'].upper()}</h1>",
                unsafe_allow_html=True,
            )
            st.write(f"**Polarity:** {result['textblob_score']:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 📊 VADER (Base)")
            st.markdown(
                f"<h1 style='color:{analyzer.color_scheme['models']['VADER (Base)']}; font-size:3rem; text-align:center;'>{result['VADER_Base'].upper()}</h1>",
                unsafe_allow_html=True,
            )
            st.write(f"**Compound:** {result['vader_base_score']:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="best-model-card">', unsafe_allow_html=True)
            st.markdown("### 🏆 VADER (Enhanced)")
            st.markdown(
                f"<h1 style='color:white; font-size:3.2rem; text-align:center;'>{result['VADER_Enhanced'].upper()}</h1>",
                unsafe_allow_html=True,
            )
            st.write(f"**Final Score:** {result['vader_enhanced_score']:.3f}")
            st.write(f"**Alpha:** {analyzer.alpha:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Legends
        create_unified_legend(analyzer, "model_comparison")

        # ✅ LIVE SCORE CHART (dynamic, per input text)
        st.markdown("## 📌 Live Score Comparison (Current Text)")

        score_df = pd.DataFrame(
            {
                "Model": ["TextBlob", "VADER (Base)", "VADER (Enhanced)"],
                "Score": [
                    result["textblob_score"],
                    result["vader_base_score"],
                    result["vader_enhanced_score"],
                ],
            }
        )

        fig_live = go.Figure(
            data=[
                go.Bar(
                    x=score_df["Model"],
                    y=score_df["Score"],
                    text=[f"{v:.3f}" for v in score_df["Score"]],
                    textposition="auto",
                )
            ]
        )
        fig_live.update_layout(
            title="Live Polarity / Compound Score (Updates Per Input)",
            yaxis_title="Score (TextBlob polarity, VADER compound)",
            yaxis_range=[-1, 1],
            height=360,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_live, use_container_width=True)

        # ✅ Explainability / Trace
        create_real_time_explanation(result, analyzer)

        details = result.get("vader_enhanced_details", {}) or {}
        sentence_scores = details.get("sentence_scores", [])

        if sentence_scores:
            st.markdown("### 🧾 Sentence-Level Breakdown (Enhanced VADER)")
            with st.expander("Show sentence scores + weights", expanded=True):
                create_sentence_breakdown(sentence_scores, analyzer)

            # Summary table
            rows = []
            dom_idx = int(details.get("dominant_sentence_index", 0))
            for i, sent in enumerate(sentence_scores):
                rows.append(
                    {
                        "Sentence #": i + 1,
                        "Text": sent["sentence"],
                        "Compound": round(sent["compound"], 3),
                        "Weight": round(sent["weight"], 2),
                        "Weighted": round(sent["weighted_compound"], 3),
                        "Dominant?": "✅" if i == dom_idx else "",
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


def create_batch_analysis_tab(analyzer: EnhancedVADERPipeline):
    st.markdown("## 📊 Batch File Analysis")
    st.markdown("---")

    st.info(
        "For batch *performance* (accuracy/F1), upload a CSV that contains a gold label column.\n\n"
        "- Minimal columns: `text`\n"
        "- For metrics: `text`, `gold_label` (gold_label in {negative, neutral, positive})"
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
            st.success("✅ Detected `gold_label` column — metrics will be computed.")
        else:
            st.warning("No `gold_label` column detected — will only generate predictions (no accuracy/F1).")

        if st.button("🚀 ANALYZE BATCH", type="primary", use_container_width=True):
            results = []
            progress = st.progress(0)
            for i, t in enumerate(df[text_col].astype(str).tolist()):
                results.append(analyzer.analyze_text(t))
                progress.progress((i + 1) / len(df))

            out = pd.DataFrame(results)
            st.success("🎉 Batch analysis complete.")
            st.dataframe(out.head(50), use_container_width=True)

            # Optional: compute metrics if gold exists
            if has_gold:
                gold = df["gold_label"].astype(str).str.lower().str.strip()
                # Basic sanity cleaning
                gold = gold.replace({"pos": "positive", "neg": "negative", "neu": "neutral"})
                valid = gold.isin(["negative", "neutral", "positive"])
                gold = gold[valid]
                out_valid = out.loc[valid].copy()

                def acc(y_true, y_pred):
                    return float((y_true.values == y_pred.values).mean()) if len(y_true) else 0.0

                # Simple per-model accuracy (since sklearn not needed here)
                acc_tb = acc(gold, out_valid["TextBlob"])
                acc_vb = acc(gold, out_valid["VADER_Base"])
                acc_ve = acc(gold, out_valid["VADER_Enhanced"])

                st.markdown("### 📈 Batch Accuracy (from your uploaded gold labels)")
                met1, met2, met3 = st.columns(3)
                met1.metric("TextBlob Accuracy", f"{acc_tb:.3f}")
                met2.metric("VADER (Base) Accuracy", f"{acc_vb:.3f}")
                met3.metric("VADER (Enhanced) Accuracy", f"{acc_ve:.3f}")

            # Download
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
    st.markdown("## 📈 Performance Metrics (Dataset-level)")
    st.markdown("---")

    create_unified_legend(analyzer, "model_comparison")

    # Your pipeline results (static by design)
    perf_df = pd.DataFrame(
        {
            "Model": ["TextBlob", "VADER (Base)", "VADER (Enhanced)"],
            "Accuracy": [0.502, 0.540, 0.556],
            "Macro F1": [0.471, 0.530, 0.542],
            "Negative F1": [0.349, 0.485, 0.488],
            "Positive F1": [0.512, 0.543, 0.561],
        }
    )

    st.markdown("### 🎯 Test Set Performance (n=5,055)")
    st.dataframe(perf_df, use_container_width=True)

    # Bar chart (static, correct place)
    st.markdown("### 📊 Performance Comparison (Static)")
    categories = ["Accuracy", "Macro F1", "Negative F1", "Positive F1"]

    fig = go.Figure()
    for i, row in perf_df.iterrows():
        fig.add_trace(
            go.Bar(
                name=row["Model"],
                x=categories,
                y=[row[c] for c in categories],
            )
        )
    fig.update_layout(
        barmode="group",
        height=450,
        yaxis_title="Score",
        plot_bgcolor="white",
        paper_bgcolor="white",
        title="Pipeline Metrics (Static Reference)",
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# MAIN
# =========================================================
def main():
    analyzer = EnhancedVADERPipeline()

    create_wow_header()
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔍 Live Analysis", "📊 Batch Analysis", "📈 Performance"])

    with tab1:
        create_single_analysis_tab(analyzer)

    with tab2:
        create_batch_analysis_tab(analyzer)

    with tab3:
        create_performance_tab(analyzer)


if __name__ == "__main__":
    main()
 
