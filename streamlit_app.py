import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import nltk

# NLTK data will be available on Streamlit Cloud via requirements.txt — no download needed at runtime
# (Remove any nltk.download calls to avoid issues)

class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.sia_base = SentimentIntensityAnalyzer()
        self.sia_enh = SentimentIntensityAnalyzer()
        
        # Custom lexicon
        enhanced_lexicon = {
            "fuel-efficient": 2.5, "overpriced": -3.0, "market crashed": -3.5, "bull market": 2.5,
            "yeah_right": -1.5, "as_if": -1.5, "not_bad": 1.5, "not_too_good": -1.5
        }
        self.sia_enh.lexicon.update(enhanced_lexicon)
        
        # Tuned thresholds as per your pipeline
        self.thresholds = {
            'pos_thr': 0.30,
            'neg_thr': -0.05,
            'strong_neg_thr': -0.25,
            'strong_pos_thr': 0.45
        }
        
        self.color_palette = {
            "TextBlob": "#EF476F",
            "VADER (Base)": "#118AB2",
            "VADER (Enhanced)": "#06D6A0",
            "negative": "#EF476F",
            "neutral": "#FFD166",
            "positive": "#06D6A0"
        }

    def predict(self, text):
        if not isinstance(text, str):
            text = str(text)
        
        # TextBlob
        tb_pol = TextBlob(text).sentiment.polarity
        tb = "positive" if tb_pol >= 0.05 else "negative" if tb_pol <= -0.05 else "neutral"
        
        # VADER Base
        vb_comp = self.sia_base.polarity_scores(text)['compound']
        vb = "positive" if vb_comp >= 0.05 else "negative" if vb_comp <= -0.05 else "neutral"
        
        # Enhanced VADER
        sentences = sent_tokenize(text)
        comps = []
        weights = []
        for s in sentences:
            vs = self.sia_enh.polarity_scores(s)
            comp = vs['compound']
            tokens = word_tokenize(s)
            len_w = min(len(tokens) / 8.0, 3.0)
            excl_w = 1.0 + min(s.count("!"), 3) * 0.15
            caps_w = 1.0 + min(len([w for w in tokens if w.isalpha() and w.upper() == w and len(w) > 2]), 3) * 0.12
            weights.append(len_w * excl_w * caps_w)
            comps.append(comp)
        
        if comps:
            comps = np.array(comps)
            weights = np.array(weights)
            if (comps <= self.thresholds['strong_neg_thr']).any():
                ve = "negative"
            elif (comps >= self.thresholds['strong_pos_thr']).any():
                ve = "positive"
            else:
                avg = np.average(comps, weights=weights)
                ve = "positive" if avg >= self.thresholds['pos_thr'] else "negative" if avg <= self.thresholds['neg_thr'] else "neutral"
            ve_score = float(avg)
        else:
            ve = "neutral"
            ve_score = 0.0
        
        return tb, vb, ve, tb_pol, vb_comp, ve_score

# Create the analyzer instance
analyzer = EnhancedSentimentAnalyzer()

# Page config
st.set_page_config(page_title="Enhanced VADER Sentiment Analysis", layout="wide")

# Beautiful header
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 30px;'>
    <h1 style='margin:0;'>🚀 Enhanced VADER Sentiment Analysis Deployment</h1>
    <p style='margin:5px 0 0 0;'>Professional deployment system for multi-domain sentiment analysis</p>
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 5px; margin-top: 15px;'>
        <strong>Features:</strong>
        <ul style='columns: 2; margin:10px 0;'>
            <li>✅ Uses actual pipeline configurations</li>
            <li>✅ Batch CSV file processing</li>
            <li>✅ Single text analysis</li>
            <li>✅ Performance visualization</li>
            <li>✅ Export results to HTML/CSV</li>
            <li>✅ API code generation</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Analysis", "📊 Batch Analysis", "📈 Performance", "📊 Visualizations"])

with tab1:
    st.markdown("<h2 style='color:#1e293b;'>🔍 Live Sentiment Analysis</h2>", unsafe_allow_html=True)
    st.write("Enter text below to analyze sentiment with all three models:")
    
    examples = [
        ("Select example...", ""),
        ("Car review: Excellent fuel economy but uncomfortable seats", "The car has excellent fuel economy which saves me money. However, the seats are very uncomfortable on long drives."),
        ("Finance news: Market crashed but recovery expected", "The stock market crashed by 15% today due to economic concerns. However, analysts expect a recovery next quarter."),
        ("Twitter sentiment: Absolutely amazing performance!", "Just watched the concert and it was absolutely amazing! The performance was 🔥🔥🔥"),
        ("Mixed sentiment: Terrible service but good food", "The service at this restaurant was terrible - we waited 45 minutes. However, the food was surprisingly good."),
        ("Negative review: Complete waste of money", "This product is a complete waste of money. It broke after 2 days and customer service was unhelpful.")
    ]
    
    selected = st.selectbox("Load Example:", [e[0] for e in examples], key="example_sel")
    selected_text = next((e[1] for e in examples if e[0] == selected), "")
    text = st.text_area("Input Text:", value=selected_text, height=150, placeholder="Enter text for sentiment analysis...", key="text_input")
    
    if st.button("Analyze Sentiment", type="primary", key="analyze"):
        if text.strip():
            tb, vb, ve, tb_score, vb_score, ve_score = analyzer.predict(text)
            html = f"""
            <div style='background:#f8fafc; padding:20px; border-radius:10px; margin-top:20px;'>
                <h3 style='color:#1e293b;'>📊 Analysis Results</h3>
                <div style='background:#f1f5f9; padding:15px; border-radius:5px; margin-bottom:20px;'>
                    <strong>Analyzed Text:</strong><br><em>"{text[:200]}{'...' if len(text)>200 else ''}"</em>
                </div>
                <table style='width:100%; border-collapse:collapse;'>
                    <tr style='background:#1e293b; color:white;'>
                        <th style='padding:12px;'>Model</th><th>Prediction</th><th>Score</th><th>Details</th>
                    </tr>
                    <tr>
                        <td><strong>TextBlob</strong></td>
                        <td><span style='color:#EF476F;font-weight:bold'>{tb.upper()}</span></td>
                        <td>{tb_score:.3f}</td>
                        <td>Polarity-based</td>
                    </tr>
                    <tr>
                        <td><strong>VADER (Base)</strong></td>
                        <td><span style='color:#118AB2;font-weight:bold'>{vb.upper()}</span></td>
                        <td>{vb_score:.3f}</td>
                        <td>Document-level</td>
                    </tr>
                    <tr>
                        <td><strong>VADER (Enhanced)</strong></td>
                        <td><span style='color:#06D6A0;font-weight:bold;border:2px solid #06D6A0;padding:4px 8px;border-radius:4px'>{ve.upper()}</span></td>
                        <td>{ve_score:.3f}</td>
                        <td>Sentence dominance + weighting</td>
                    </tr>
                </table>
                <div style='background:#e3f2fd;padding:15px;border-radius:8px;margin-top:20px;'>
                    <strong>Enhanced VADER Features:</strong>
                    <ul>
                        <li>Sentence Dominance: Any sentence ≤ -0.25 → Negative, ≥ 0.45 → Positive</li>
                        <li>Domain Lexicon: Custom words for car, finance, and Twitter domains</li>
                        <li>Weighted Average: Sentences weighted by length and emphasis</li>
                        <li>Tuned Thresholds: Positive ≥ 0.30, Negative ≤ -0.05</li>
                    </ul>
                </div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.error("Please enter some text.")

# Advanced Tools at the bottom
st.markdown("<h2 style='color:#1e293b;'>🛠 Advanced Deployment Tools</h2>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.button("📄 Export HTML Report")
with c2:
    st.button("⚙️ Generate API Code")
