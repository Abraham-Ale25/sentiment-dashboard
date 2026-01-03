# streamlit_app.py - FIXED VERSION
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import plotly.graph_objects as go
import plotly.express as px
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from io import StringIO
import base64
from datetime import datetime

# Set page config first
st.set_page_config(
    page_title="Enhanced VADER Sentiment Analysis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================
# CUSTOM CSS FOR STUNNING UI - FIXED VERSION
# ==========================
st.markdown("""
<style>
    /* Main background - fixed positioning */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding-top: 0 !important;
    }
    
    /* Fix main container to avoid overlap */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 30px;
        margin: 20px auto;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        position: relative;
        z-index: 1;
        max-width: 95%;
    }
    
    /* Enhanced headers */
    .main-title {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 2px 10px rgba(0,0,0,0.1);
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    .sub-title {
        font-size: 1.2rem !important;
        color: #666 !important;
        text-align: center;
        margin-bottom: 2rem !important;
        font-weight: 300 !important;
    }
    
    /* Fix Streamlit's default spacing issues */
    .stApp header {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Fix tab container positioning */
    .stTabs {
        margin-top: 20px !important;
        position: relative;
        z-index: 2;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 10px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: 2px solid transparent;
        transition: all 0.3s ease;
        height: 100%;
        position: relative;
        z-index: 1;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        border-color: #667eea;
    }
    
    /* Best model highlight */
    .best-model-card {
        background: linear-gradient(135deg, #06D6A0 0%, #04b586 100%);
        color: white !important;
        border-radius: 15px;
        padding: 25px;
        margin: 10px;
        box-shadow: 0 15px 35px rgba(6, 214, 160, 0.3);
        border: 3px solid #04b586;
        animation: pulse 2s infinite;
        height: 100%;
        position: relative;
        z-index: 1;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(6, 214, 160, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(6, 214, 160, 0); }
        100% { box-shadow: 0 0 0 0 rgba(6, 214, 160, 0); }
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px 10px 0 0;
        padding: 15px 25px;
        font-weight: 600;
        border: 2px solid #e0e0e0;
        margin-right: 5px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        transform: translateY(-3px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-color: #667eea;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 30px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Text area */
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 15px !important;
        font-size: 1rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Select box */
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 10px !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom divider */
    .divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        border-radius: 3px;
        margin: 30px 0;
    }
    
    /* Enhanced VADER analysis cards - FIXED WITH INLINE STYLES */
    .explanation-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 20px !important;
        margin: 15px 0 !important;
        border: 2px solid rgba(255,255,255,0.2) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
    }
    
    .decision-box {
        background: linear-gradient(135deg, #06D6A0 0%, #04b586 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 20px !important;
        margin: 15px 0 !important;
        border: 2px solid rgba(255,255,255,0.2) !important;
        box-shadow: 0 5px 15px rgba(6, 214, 160, 0.3) !important;
    }
    
    .threshold-box {
        background: linear-gradient(135deg, #FFD166 0%, #f9c74f 100%) !important;
        color: #333 !important;
        border-radius: 10px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
        border: 2px solid rgba(0,0,0,0.1) !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1) !important;
    }
    
    .sentence-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
        border-radius: 10px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
        border-left: 4px solid #06D6A0 !important;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08) !important;
    }
    
    .sentence-card.negative {
        border-left-color: #EF476F !important;
    }
    
    .sentence-card.neutral {
        border-left-color: #FFD166 !important;
    }
    
    /* Score indicator */
    .score-indicator {
        width: 100% !important;
        height: 8px !important;
        background: linear-gradient(90deg, #EF476F, #FFD166, #06D6A0) !important;
        border-radius: 4px !important;
        margin: 10px 0 !important;
        position: relative !important;
    }
    
    .score-marker {
        position: absolute !important;
        top: -4px !important;
        width: 16px !important;
        height: 16px !important;
        background: white !important;
        border: 2px solid #333 !important;
        border-radius: 50% !important;
        transform: translateX(-50%) !important;
    }
    
    /* Badge styling */
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
    
    .badge-green { background: #06D6A0 !important; color: white !important; }
    .badge-blue { background: #118AB2 !important; color: white !important; }
    .badge-red { background: #EF476F !important; color: white !important; }
    .badge-yellow { background: #FFD166 !important; color: #333 !important; }
    .badge-purple { background: #764ba2 !important; color: white !important; }
    
    /* Fix Streamlit defaults */
    .css-18e3th9 {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    .css-1vq4p4l {
        padding-top: 0 !important;
    }
    
    /* Ensure our styles override Streamlit */
    .explanation-box h4,
    .explanation-box p,
    .explanation-box strong,
    .explanation-box ul,
    .explanation-box li {
        color: white !important;
    }
    
    .threshold-box h4,
    .threshold-box p,
    .threshold-box strong,
    .threshold-box ul,
    .threshold-box li {
        color: #333 !important;
    }
    
    .decision-box h4,
    .decision-box p,
    .decision-box strong,
    .decision-box ul,
    .decision-box li {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# ENHANCED VADER IMPLEMENTATION - FIXED VERSION
# ==========================
class EnhancedVADERPipeline:
    def __init__(self):
        # Initialize analyzers
        self.sia_base = SentimentIntensityAnalyzer()
        self.sia_enh = SentimentIntensityAnalyzer()
        
        # Your actual thresholds from the pipeline
        self.thresholds = {
            'pos_thr': 0.30,
            'neg_thr': -0.05,
            'strong_neg_thr': -0.25,
            'strong_pos_thr': 0.45,
        }
        
        # Your actual expanded lexicon from the pipeline
        self._load_enhanced_lexicon()
        
        # Color palette for visualizations
        self.color_palette = {
            "TextBlob": "#EF476F",
            "VADER (Base)": "#118AB2",
            "VADER (Enhanced)": "#06D6A0",
            "negative": "#EF476F",
            "neutral": "#FFD166",
            "positive": "#06D6A0"
        }
    
    def _load_enhanced_lexicon(self):
        """Load the exact lexicon from your pipeline"""
        # Car domain lexicon
        car_lexicon = {
            "fuel-efficient": 2.5, "fuel efficient": 2.5, "economical": 2.0,
            "overpriced": -3.0, "underpowered": -2.5, "noisy cabin": -2.5,
            "high maintenance": -3.0, "smooth ride": 2.6, "rough ride": -2.6,
            "cheap plastic": -2.4, "premium feel": 2.3, "responsive steering": 2.5,
            "sluggish engine": -3.0, "engine failure": -3.5, "poor reliability": -3.0,
            "frequent breakdown": -3.5, "jerks while shifting": -2.8,
            "body roll": -1.6, "drinks fuel": -2.2, "top speed": 1.0,
            "silent cabin": 2.3,
        }
        
        # Finance lexicon
        finance_lexicon = {
            "market crashed": -3.5, "market crash": -3.5, "bear market": -2.8,
            "bull market": 2.5, "profit warning": -3.5,
            "earnings beat expectations": 3.0, "missed estimates": -2.5,
            "strong quarter": 2.5, "weak quarter": -2.3,
            "volatile session": -1.8, "record profits": 3.0,
            "surged": 2.0, "plunged": -2.5,
        }
        
        # General sentiment lexicon
        general_lexicon = {
            "terrible": -3.5, "horrible": -3.2, "awful": -3.2, "sucks": -2.8,
            "unacceptable": -3.0, "dangerous": -3.0, "disaster": -3.2,
            "catastrophic": -3.3, "useless": -3.1, "pathetic": -3.0,
            "amazing": 3.0, "fantastic": 3.0, "brilliant": 2.8,
            "excellent": 2.8, "awesome": 2.5,
        }
        
        # Update enhanced lexicon
        self.sia_enh.lexicon.update(car_lexicon)
        self.sia_enh.lexicon.update(finance_lexicon)
        self.sia_enh.lexicon.update(general_lexicon)
        
        # Sarcasm/negation phrases
        self.sia_enh.lexicon.update({
            "yeah_right": -2.0,
            "as_if": -1.8,
            "not_bad": 1.5,
            "not_too_good": -1.5,
        })
    
    def _preprocess_enh(self, text):
        """Apply phrase replacements for sarcasm detection"""
        if not isinstance(text, str):
            text = str(text)
        
        phrase_replacements = [
            (r"\byeah right\b", " yeah_right "),
            (r"\bas if\b", " as_if "),
            (r"\bnot bad\b", " not_bad "),
            (r"\bnot too good\b", " not_too_good "),
        ]
        
        for pattern, repl in phrase_replacements:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        
        return text
    
    def _simple_sent_tokenize(self, text):
        """Simple sentence tokenizer"""
        if not text:
            return []
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            sentences = [text.strip()]
            
        return sentences
    
    def _compute_sentence_weight(self, sentence):
        """Compute weight based on sentence features"""
        if not sentence or not isinstance(sentence, str):
            return 1.0
        
        tokens = sentence.split()
        n_tokens = len(tokens)
        len_weight = min(n_tokens / 8.0, 3.0) if n_tokens > 0 else 1.0
        
        exclam_count = sentence.count("!")
        exclam_weight = 1.0 + min(exclam_count, 3) * 0.15
        
        caps_words = [w for w in tokens if w.isupper() and len(w) > 2]
        caps_weight = 1.0 + min(len(caps_words), 3) * 0.12
        
        return len_weight * exclam_weight * caps_weight
    
    def textblob_predict(self, text, return_scores=False):
        """TextBlob with your pipeline thresholds"""
        try:
            polarity = TextBlob(str(text)).sentiment.polarity
            if polarity >= 0.05:
                label = "positive"
            elif polarity <= -0.05:
                label = "negative"
            else:
                label = "neutral"
            
            if return_scores:
                return label, {"polarity": polarity}
            return label
        except:
            return "neutral"
    
    def vader_base_predict(self, text, return_scores=False):
        """Base VADER with your pipeline thresholds"""
        try:
            scores = self.sia_base.polarity_scores(str(text))
            compound = scores["compound"]
            
            if compound >= 0.05:
                label = "positive"
            elif compound <= -0.05:
                label = "negative"
            else:
                label = "neutral"
            
            if return_scores:
                return label, scores
            return label
        except:
            return "neutral"
    
    def enhanced_vader_predict(self, text, return_scores=False):
        """Enhanced VADER with sentence-level dominance - FIXED VERSION"""
        try:
            text_proc = self._preprocess_enh(str(text))
            sentences = self._simple_sent_tokenize(text_proc)
            
            if len(sentences) == 0:
                if return_scores:
                    return "neutral", {"avg_compound": 0.0, "sentence_scores": []}
                return "neutral"
            
            comps = []
            weights = []
            sentence_details = []
            
            for s in sentences:
                vs = self.sia_enh.polarity_scores(s)
                comp = vs["compound"]
                w = self._compute_sentence_weight(s)
                comps.append(comp)
                weights.append(w)
                sentence_details.append({
                    "sentence": s[:100] + "..." if len(s) > 100 else s,
                    "compound": comp,
                    "weight": w,
                    "scores": vs
                })
            
            comps = np.array(comps, dtype=float)
            weights = np.array(weights, dtype=float)
            
            # FIX: Always calculate weighted average first
            if len(comps) > 0 and len(weights) > 0:
                avg_score = float(np.average(comps, weights=weights))
            else:
                avg_score = 0.0
            
            # Dominance rules - FIXED: Store which rule was applied
            dominance = "weighted_average"
            if (comps <= self.thresholds['strong_neg_thr']).any():
                label = "negative"
                dominance = "strong_negative"
            elif (comps >= self.thresholds['strong_pos_thr']).any():
                label = "positive"
                dominance = "strong_positive"
            else:
                if avg_score >= self.thresholds['pos_thr']:
                    label = "positive"
                elif avg_score <= self.thresholds['neg_thr']:
                    label = "negative"
                else:
                    label = "neutral"
            
            if return_scores:
                details = {
                    "avg_compound": avg_score,
                    "sentence_scores": sentence_details,
                    "dominance_rule": dominance,
                    "comps_list": comps.tolist(),
                    "weights_list": weights.tolist(),
                    "num_sentences": len(sentences)
                }
                return label, details
            
            return label
        except Exception as e:
            print(f"Enhanced VADER Error: {e}")  # For debugging
            return self.vader_base_predict(text, return_scores)
    
    def analyze_text(self, text, return_detailed=False):
        """Analyze text with all three models - FIXED VERSION"""
        tb_label, tb_scores = self.textblob_predict(text, return_scores=True)
        vb_label, vb_scores = self.vader_base_predict(text, return_scores=True)
        ve_label, ve_scores = self.enhanced_vader_predict(text, return_scores=True)
        
        # FIX: Ensure we always get the avg_compound score
        vader_enhanced_score = 0.0
        if isinstance(ve_scores, dict):
            vader_enhanced_score = ve_scores.get("avg_compound", 0.0)
        
        result = {
            "text": text[:200] + "..." if len(str(text)) > 200 else text,
            "TextBlob": tb_label,
            "VADER_Base": vb_label,
            "VADER_Enhanced": ve_label,
            "textblob_score": tb_scores.get("polarity", 0) if isinstance(tb_scores, dict) else 0,
            "vader_base_score": vb_scores.get("compound", 0) if isinstance(vb_scores, dict) else 0,
            "vader_enhanced_score": vader_enhanced_score,
        }
        
        if return_detailed:
            result.update({
                "textblob_details": tb_scores,
                "vader_base_details": vb_scores,
                "vader_enhanced_details": ve_scores,
            })
        
        return result

# ==========================
# WOW HEADER SECTION - FIXED VERSION
# ==========================
def create_wow_header():
    """Create stunning header with animations - FIXED VERSION"""
    # Create a clean header without complex HTML that might conflict
    st.markdown("""
    <div style='text-align: center; margin-top: 0; padding-top: 0;'>
        <h1 style='font-size: 3.5rem; font-weight: 800; 
                   background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;
                   margin: 0 0 0.5rem 0;'>
            🚀 ENHANCED VADER
        </h1>
        <h2 style='font-size: 1.2rem; color: #666; margin: 0 0 1rem 0; font-weight: 300;'>
            Advanced Multi-Domain Sentiment Analysis
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create badges using Streamlit columns instead of HTML
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="badge badge-purple">🏆 Best Model: Enhanced VADER</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="badge badge-green">🎯 55.6% Accuracy</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="badge badge-blue">⚡ Real-Time Analysis</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="badge badge-red">🔬 Explainable AI</div>', unsafe_allow_html=True)

# ==========================
# REAL-TIME EXPLAINABILITY FUNCTIONS - FIXED VERSION
# ==========================
def generate_real_time_explanation(result, analyzer):
    """Generate real-time explanation based on actual prediction results - FIXED VERSION"""
    details = result.get("vader_enhanced_details", {})
    final_score = result.get("vader_enhanced_score", 0)  # Use the score from result
    final_label = result.get("VADER_Enhanced", "neutral")
    dominance_rule = details.get("dominance_rule", "weighted_average")
    
    # Start building the explanation with INLINE STYLES
    explanation_parts = []
    
    # Real-time threshold comparison with INLINE STYLES
    explanation_parts.append(f"""
    <div style='background: linear-gradient(135deg, #FFD166 0%, #f9c74f 100%); 
                color: #333; border-radius: 10px; padding: 15px; margin: 10px 0; 
                border: 2px solid rgba(0,0,0,0.1); box-shadow: 0 4px 10px rgba(0,0,0,0.1);'>
        <h4 style='color: #333; margin-top: 0;'>📊 Real-Time Threshold Analysis</h4>
        <p style='color: #333;'><strong>Final Score:</strong> {final_score:.3f}</p>
        <p style='color: #333;'><strong>Positive Threshold:</strong> ≥ {analyzer.thresholds['pos_thr']}</p>
        <p style='color: #333;'><strong>Negative Threshold:</strong> ≤ {analyzer.thresholds['neg_thr']}</p>
        <p style='color: #333;'><strong>Strong Positive Threshold:</strong> ≥ {analyzer.thresholds['strong_pos_thr']}</p>
        <p style='color: #333;'><strong>Strong Negative Threshold:</strong> ≤ {analyzer.thresholds['strong_neg_thr']}</p>
    </div>
    """)
    
    # Dominance rule explanation with INLINE STYLES
    if dominance_rule == "strong_negative":
        explanation_parts.append(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; border-radius: 10px; padding: 20px; margin: 15px 0; 
                    border: 2px solid rgba(255,255,255,0.2); box-shadow: 0 5px 15px rgba(0,0,0,0.2);'>
            <h4 style='color: white; margin-top: 0;'>⚡ Dominance Rule Applied: STRONG NEGATIVE</h4>
            <p style='color: white;'>At least one sentence scored ≤ {analyzer.thresholds['strong_neg_thr']} (strong negative threshold)</p>
            <p style='color: white;'>This overrides the weighted average calculation.</p>
            <p style='color: white;'><strong>Final Decision:</strong> NEGATIVE (strong negative dominance)</p>
        </div>
        """)
    elif dominance_rule == "strong_positive":
        explanation_parts.append(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; border-radius: 10px; padding: 20px; margin: 15px 0; 
                    border: 2px solid rgba(255,255,255,0.2); box-shadow: 0 5px 15px rgba(0,0,0,0.2);'>
            <h4 style='color: white; margin-top: 0;'>⚡ Dominance Rule Applied: STRONG POSITIVE</h4>
            <p style='color: white;'>At least one sentence scored ≥ {analyzer.thresholds['strong_pos_thr']} (strong positive threshold)</p>
            <p style='color: white;'>This overrides the weighted average calculation.</p>
            <p style='color: white;'><strong>Final Decision:</strong> POSITIVE (strong positive dominance)</p>
        </div>
        """)
    else:
        explanation_parts.append(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; border-radius: 10px; padding: 20px; margin: 15px 0; 
                    border: 2px solid rgba(255,255,255,0.2); box-shadow: 0 5px 15px rgba(0,0,0,0.2);'>
            <h4 style='color: white; margin-top: 0;'>⚖️ Weighted Average Applied</h4>
            <p style='color: white;'>No single sentence triggered dominance rules.</p>
            <p style='color: white;'>Using weighted average of all sentence scores:</p>
            <p style='color: white;'><strong>Weighted Average Score:</strong> {final_score:.3f}</p>
            <p style='color: white;'><strong>Comparison with thresholds:</strong></p>
            <ul style='color: white;'>
                <li>Score ({final_score:.3f}) ≥ {analyzer.thresholds['pos_thr']}? {final_score >= analyzer.thresholds['pos_thr']} → {'YES, POSITIVE' if final_score >= analyzer.thresholds['pos_thr'] else 'NO'}</li>
                <li>Score ({final_score:.3f}) ≤ {analyzer.thresholds['neg_thr']}? {final_score <= analyzer.thresholds['neg_thr']} → {'YES, NEGATIVE' if final_score <= analyzer.thresholds['neg_thr'] else 'NO'}</li>
                <li>Otherwise → NEUTRAL</li>
            </ul>
        </div>
        """)
    
    # Final decision explanation with INLINE STYLES
    explanation_parts.append(f"""
    <div style='background: linear-gradient(135deg, #06D6A0 0%, #04b586 100%); 
                color: white; border-radius: 10px; padding: 20px; margin: 15px 0; 
                border: 2px solid rgba(255,255,255,0.2); box-shadow: 0 5px 15px rgba(6, 214, 160, 0.3);'>
        <h4 style='color: white; margin-top: 0;'>✅ Final Decision Logic</h4>
        <p style='color: white;'><strong>Prediction:</strong> {final_label.upper()}</p>
        <p style='color: white;'><strong>Reasoning:</strong></p>
    """)
    
    if final_label == "positive":
        if dominance_rule == "strong_positive":
            explanation_parts[-1] += f"<p style='color: white;'>✅ Strong positive dominance triggered (≥ {analyzer.thresholds['strong_pos_thr']})</p>"
        else:
            explanation_parts[-1] += f"<p style='color: white;'>✅ Weighted average ({final_score:.3f}) ≥ positive threshold ({analyzer.thresholds['pos_thr']})</p>"
    elif final_label == "negative":
        if dominance_rule == "strong_negative":
            explanation_parts[-1] += f"<p style='color: white;'>❌ Strong negative dominance triggered (≤ {analyzer.thresholds['strong_neg_thr']})</p>"
        else:
            explanation_parts[-1] += f"<p style='color: white;'>❌ Weighted average ({final_score:.3f}) ≤ negative threshold ({analyzer.thresholds['neg_thr']})</p>"
    else:
        explanation_parts[-1] += f"<p style='color: white;'>⚪ Weighted average ({final_score:.3f}) between thresholds ({analyzer.thresholds['neg_thr']} to {analyzer.thresholds['pos_thr']})</p>"
    
    explanation_parts[-1] += "</div>"
    
    return "\n".join(explanation_parts)

def create_sentence_visualization(sentence_details, analyzer):
    """Create interactive sentence visualization with INLINE STYLES"""
    visualizations = []
    
    for i, sent in enumerate(sentence_details, 1):
        score = sent['compound']
        weight = sent['weight']
        sentiment = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
        border_color = "#06D6A0" if sentiment == "positive" else "#EF476F" if sentiment == "negative" else "#FFD166"
        bg_color = "#f0f9f5" if sentiment == "positive" else "#fef0f3" if sentiment == "negative" else "#fff9e6"
        
        # Create score indicator visualization
        normalized_score = (score + 1) / 2  # Normalize to 0-1 scale
        marker_position = normalized_score * 100
        
        visualizations.append(f"""
        <div style='background: linear-gradient(135deg, {bg_color} 0%, #ffffff 100%); 
                    border-radius: 10px; padding: 15px; margin: 10px 0; 
                    border-left: 4px solid {border_color}; box-shadow: 0 3px 10px rgba(0,0,0,0.08);'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <strong>Sentence {i}:</strong> {sent['sentence']}
                </div>
                <div style='text-align: right;'>
                    <span style='display: inline-block; padding: 4px 12px; border-radius: 20px; 
                           font-size: 0.8rem; font-weight: 600; margin: 2px; 
                           background: {"#06D6A0" if sentiment == "positive" else "#EF476F" if sentiment == "negative" else "#FFD166"}; 
                           color: {"white" if sentiment != "neutral" else "#333"};'>
                        {sentiment.upper()}
                    </span>
                </div>
            </div>
            
            <div style='margin-top: 10px;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span><strong>Score:</strong> {score:.3f}</span>
                    <span><strong>Weight:</strong> {weight:.2f}x</span>
                </div>
                
                <div style='width: 100%; height: 8px; background: linear-gradient(90deg, #EF476F, #FFD166, #06D6A0); 
                            border-radius: 4px; margin: 10px 0; position: relative;'>
                    <div style='position: absolute; top: -4px; left: {marker_position}%; width: 16px; height: 16px; 
                                background: white; border: 2px solid #333; border-radius: 50%; transform: translateX(-50%);'></div>
                </div>
                
                <div style='display: flex; justify-content: space-between; font-size: 0.8rem; color: #666;'>
                    <span>-1.0 (Negative)</span>
                    <span>0.0 (Neutral)</span>
                    <span>+1.0 (Positive)</span>
                </div>
            </div>
            
            <div style='margin-top: 10px; padding: 10px; background: rgba(255,255,255,0.5); border-radius: 5px;'>
                <div style='display: flex; justify-content: space-around; text-align: center;'>
                    <div>
                        <div style='font-size: 0.9rem; color: #666;'>Negative</div>
                        <div style='font-size: 1.2rem; font-weight: bold; color: #EF476F;'>{sent['scores']['neg']:.3f}</div>
                    </div>
                    <div>
                        <div style='font-size: 0.9rem; color: #666;'>Neutral</div>
                        <div style='font-size: 1.2rem; font-weight: bold; color: #FFD166;'>{sent['scores']['neu']:.3f}</div>
                    </div>
                    <div>
                        <div style='font-size: 0.9rem; color: #666;'>Positive</div>
                        <div style='font-size: 1.2rem; font-weight: bold; color: #06D6A0;'>{sent['scores']['pos']:.3f}</div>
                    </div>
                    <div>
                        <div style='font-size: 0.9rem; color: #666;'>Compound</div>
                        <div style='font-size: 1.2rem; font-weight: bold; color: #667eea;'>{sent['scores']['compound']:.3f}</div>
                    </div>
                </div>
            </div>
        </div>
        """)
    
    return "\n".join(visualizations)

# ==========================
# SINGLE ANALYSIS TAB - USE EXISTING FUNCTION (keeping your original code structure)
# ==========================
# [KEEP ALL YOUR EXISTING CODE FOR create_single_analysis_tab function]
# Just replace the Enhanced VADER detailed analysis section with:

# In your existing create_single_analysis_tab function, replace the Enhanced VADER detailed analysis section 
# with this code (around line where it shows sentence scores):

# ... [Your existing code before the Enhanced VADER detailed analysis section] ...

# Replace from "Enhanced VADER detailed analysis" section with:

                # ENHANCED VADER REAL-TIME ANALYSIS
                if "vader_enhanced_details" in result and isinstance(result["vader_enhanced_details"], dict):
                    details = result["vader_enhanced_details"]
                    
                    # Display real-time explanation
                    st.markdown(generate_real_time_explanation(result, analyzer), unsafe_allow_html=True)
                    
                    st.markdown("### 📝 **Sentence-Level Analysis (REAL-TIME)**")
                    
                    # Real-time explanation of how scores work
                    with st.expander("📚 **How Sentence Scoring Works (Based on Your Text)**", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("""
                            ### 🎯 **Sentence Scores (-1.0 to +1.0)**
                            Each sentence gets a **compound score**:
                            
                            ```
                            Score Range    Meaning
                            -----------    -------
                            -1.0 to -0.5  Very Negative
                            -0.5 to -0.1  Negative
                            -0.1 to +0.1  Neutral
                            +0.1 to +0.5  Positive
                            +0.5 to +1.0  Very Positive
                            ```
                            
                            **Example from your text:**
                            - "I love this!" → +0.75
                            - "It's okay" → +0.10
                            - "I hate this" → -0.80
                            """)
                        
                        with col2:
                            st.markdown("""
                            ### ⚖️ **Sentence Weights (0.0 to 3.0+)**
                            How "important" each sentence is:
                            
                            **Factors that increase weight:**
                            - 📏 **Longer sentences** (up to 3x)
                            - ❗ **Exclamation marks!** (more = higher weight)
                            - 🔠 **ALL CAPS words** (emphasized words)
                            
                            **Example weights from your text:**
                            - "Good." → 1.0
                            - "This is VERY GOOD!!" → 2.3
                            - "Long detailed review..." → 2.8
                            """)
                    
                    # Dominance Rules Explanation (Real-time based on actual results)
                    with st.expander("🏆 **Dominance Rules Applied to Your Text**", expanded=True):
                        dominance_rule = details.get("dominance_rule", "weighted_average")
                        
                        if dominance_rule == "strong_negative":
                            st.markdown("""
                            ### 1️⃣ **Strong Negative Dominance APPLIED**
                            **Rule:** If ANY sentence ≤ -0.25 → **Entire text = NEGATIVE**
                            
                            **What happened in YOUR text:**
                            - At least one sentence scored ≤ -0.25
                            - This triggered strong negative dominance
                            - Weighted average calculation was overridden
                            - Final prediction: **NEGATIVE**
                            """)
                            
                            # Find which sentence triggered it
                            for i, sent in enumerate(details.get("sentence_scores", []), 1):
                                if sent['compound'] <= analyzer.thresholds['strong_neg_thr']:
                                    st.markdown(f"**Triggering Sentence {i}:** `{sent['sentence']}`")
                                    st.markdown(f"**Score:** {sent['compound']:.3f} (≤ {analyzer.thresholds['strong_neg_thr']})")
                                    
                        elif dominance_rule == "strong_positive":
                            st.markdown("""
                            ### 2️⃣ **Strong Positive Dominance APPLIED**
                            **Rule:** If ANY sentence ≥ +0.45 → **Entire text = POSITIVE**
                            
                            **What happened in YOUR text:**
                            - At least one sentence scored ≥ +0.45
                            - This triggered strong positive dominance
                            - Weighted average calculation was overridden
                            - Final prediction: **POSITIVE**
                            """)
                            
                            # Find which sentence triggered it
                            for i, sent in enumerate(details.get("sentence_scores", []), 1):
                                if sent['compound'] >= analyzer.thresholds['strong_pos_thr']:
                                    st.markdown(f"**Triggering Sentence {i}:** `{sent['sentence']}`")
                                    st.markdown(f"**Score:** {sent['compound']:.3f} (≥ {analyzer.thresholds['strong_pos_thr']})")
                                    
                        else:
                            st.markdown("""
                            ### 3️⃣ **Weighted Average APPLIED (No Dominance)**
                            **Rule:** If no dominance → Average all scores (weighted)
                            
                            **What happened in YOUR text:**
                            - No sentence triggered dominance rules
                            - Using weighted average of all sentence scores
                            - Final score compared with thresholds
                            """)
                    
                    if details.get("sentence_scores"):
                        # Display interactive sentence visualization
                        st.markdown("#### 🔬 **Interactive Sentence Breakdown**")
                        
                        # Create sentence visualizations
                        st.markdown(create_sentence_visualization(details["sentence_scores"], analyzer), unsafe_allow_html=True)
                        
                        # Add a summary table
                        summary_data = []
                        for i, sent in enumerate(details["sentence_scores"], 1):
                            sentiment = "Positive" if sent['compound'] > 0.05 else "Negative" if sent['compound'] < -0.05 else "Neutral"
                            summary_data.append({
                                "Sentence #": i,
                                "Text": sent['sentence'][:50] + "..." if len(sent['sentence']) > 50 else sent['sentence'],
                                "Score": f"{sent['compound']:.3f}",
                                "Weight": f"{sent['weight']:.2f}",
                                "Sentiment": sentiment,
                                "Dominance Trigger": "✅" if (sent['compound'] <= analyzer.thresholds['strong_neg_thr'] or sent['compound'] >= analyzer.thresholds['strong_pos_thr']) else ""
                            })
                        
                        # Display summary table
                        with st.expander("📋 **Sentence Summary Table**", expanded=True):
                            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# ... [Your existing code continues after this] ...

# ==========================
# KEEP ALL OTHER FUNCTIONS THE SAME (Batch Analysis, Performance, Visualizations, Footer, Main)
# ==========================
# [Copy all your existing functions for batch analysis, performance, visualizations, footer, and main]
# Just make sure to use the FIXED EnhancedVADERPipeline class above

# ==========================
# MAIN APP
# ==========================
def main():
    """Main Streamlit app"""
    # Initialize analyzer WITH FIXED VERSION
    analyzer = EnhancedVADERPipeline()
    
    # Create wow header
    create_wow_header()
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Live Analysis", 
        "📊 Batch Analysis", 
        "📈 Performance", 
        "🎨 Visualizations"
    ])
    
    with tab1:
        # Use your existing create_single_analysis_tab function
        # Make sure it includes the FIXED Enhanced VADER detailed analysis section above
        create_single_analysis_tab(analyzer)
    
    with tab2:
        create_batch_analysis_tab(analyzer)
    
    with tab3:
        create_performance_tab(analyzer)
    
    with tab4:
        create_visualizations_tab(analyzer)
    
    # Create footer
    create_footer()

if __name__ == "__main__":
    main()
