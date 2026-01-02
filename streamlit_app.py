import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.sia_base = SentimentIntensityAnalyzer()
        self.sia_enh = SentimentIntensityAnalyzer()
        
        enhanced_lexicon = {
            "fuel-efficient": 2.5, "overpriced": -3.0, "market crashed": -3.5, "bull market": 2.5,
            "yeah_right": -1.5, "as_if": -1.5, "not_bad": 1.5, "not_too_good": -1.5
        }
        self.sia_enh.lexicon.update(enhanced_lexicon)
        
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

    def textblob_predict(self, text):
        polarity = TextBlob(text).sentiment.polarity
        if polarity >= 0.05:
            return "positive"
        elif polarity <= -0.05:
            return "negative"
        return "neutral"

    def vader_base_predict(self, text):
        c = self.sia_base.polarity_scores(text)["compound"]
        if c >= 0.05:
            return "positive"
        elif c <= -0.05:
            return "negative"
        return "neutral"

    def enhanced_vader_predict(self, text):
        text = self._preprocess_enh(text)
        sentences = self._simple_sent_tokenize(text)
        if not sentences:
            return "neutral"
        
        comps = []
        weights = []
        for s in sentences:
            vs = self.sia_enh.polarity_scores(s)
            comp = vs["compound"]
            w = self._compute_sentence_weight(s)
            comps.append(comp)
            weights.append(w)
        
        comps = np.array(comps, dtype=float)
        weights = np.array(weights, dtype=float)
        
        if (comps <= self.thresholds['strong_neg_thr']).any():
            return "negative"
        if (comps >= self.thresholds['strong_pos_thr']).any():
            return "positive"
        
        avg_score = float(np.average(comps, weights=weights))
        
        if avg_score >= self.thresholds['pos_thr']:
            return "positive"
        elif avg_score <= self.thresholds['neg_thr']:
            return "negative"
        else:
            return "neutral"

    def _preprocess_enh(self, text):
        if not isinstance(text, str):
            text = str(text)
        
        replacements = [
            (r"\byeah right\b", " yeah_right "),
            (r"\bas if\b", " as_if "),
            (r"\bnot bad\b", " not_bad "),
            (r"\bnot too good\b", " not_too_good "),
        ]
        for p, r in replacements:
            text = re.sub(p, r, text, flags=re.IGNORECASE)
        return text

    def _compute_sentence_weight(self, s):
        if not s.strip():
            return 1.0
        tokens = s.split()
        len_w = min(len(tokens)/8.0, 3.0)
        excl_w = 1.0 + min(s.count("!"), 3)*0.15
        caps_w = 1.0 + min(len([w for w in tokens if w.isupper() and len(w)>2]), 3)*0.12
        return len_w * excl_w * caps_w

    def _simple_sent_tokenize(self, text):
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

analyzer = EnhancedSentimentAnalyzer()

st.set_page_config(page_title="Enhanced VADER Sentiment Analysis", layout="wide")

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
            <li>✅ Visualizations for every analysis</li>
            <li>✅ Export results to CSV</li>
            <li>✅ API code generation</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

tab1 = st.tabs(["🔍 Single Analysis"])[0]

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
    
    selected = st.selectbox("Load Example:", [e[0] for e in examples])
    selected_text = next((e[1] for e in examples if e[0] == selected), "")
    text = st.text_area("Input Text:", value=selected_text, height=150, placeholder="Enter text for sentiment analysis...")
    
    if st.button("Analyze Sentiment", type="primary"):
        if text.strip():
            tb = analyzer.textblob_predict(text)
            vb = analyzer.vader_base_predict(text)
            ve = analyzer.enhanced_vader_predict(text)
            tb_score = TextBlob(text).sentiment.polarity
            vb_score = analyzer.sia_base.polarity_scores(text)['compound']
            ve_score = analyzer.sia_enh.polarity_scores(text)['compound']
            
            # Results table
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
                        <td>Custom lexicon + tuned thresholds</td>
                    </tr>
                </table>
                <div style='background:#e3f2fd;padding:15px;border-radius:8px;margin-top:20px;'>
                    <strong>Enhanced VADER Features:</strong>
                    <ul>
                        <li>Custom Lexicon: Domain-specific words for car, finance, sarcasm</li>
                        <li>Tuned Thresholds: Positive ≥ 0.30, Negative ≤ -0.05</li>
                        <li>Strong Dominance Rules Applied</li>
                    </ul>
                </div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
            
            # Visualizations for single text
            st.markdown("<h3 style='color:#1e293b;'>📊 Model Comparison Visualization</h3>", unsafe_allow_html=True)
            
            models = ['TextBlob', 'VADER (Base)', 'VADER (Enhanced)']
            scores = [tb_score, vb_score, ve_score]
            colors = ['#EF476F', '#118AB2', '#06D6A0']
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(models, scores, color=colors)
            ax.set_ylim(-1, 1)
            ax.set_ylabel('Compound Score')
            ax.set_title('Sentiment Score Comparison')
            st.pyplot(fig)
            
            # Pie chart for predictions
            pred_counts = pd.Series([tb, vb, ve]).value_counts()
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.pie(pred_counts.values, labels=pred_counts.index, colors=[self.color_palette[p.lower()] for p in pred_counts.index], autopct='%1.1f%%', startangle=90)
            ax2.set_title('Prediction Distribution')
            st.pyplot(fig2)
        else:
            st.error("Please enter some text.")

st.markdown("<h2 style='color:#1e293b;'>🛠 Advanced Deployment Tools</h2>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.button("📄 Export HTML Report")
with c2:
    st.button("⚙️ Generate API Code")
