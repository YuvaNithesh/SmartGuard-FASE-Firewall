import streamlit as st
import numpy as np
import joblib
import time
import sys
import os

# Ensure the src folder is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from feature_engineering import FASEExtractor

# --- Page Config ---
st.set_page_config(page_title="SmartGuard AI Firewall", page_icon="🛡️", layout="wide")

# --- Initialize Session State for Metrics ---
if 'total_scanned' not in st.session_state:
    st.session_state.total_scanned = 0
    st.session_state.total_blocked = 0
    st.session_state.avg_latency = 0.0

# --- Load Model & Extractor ---
@st.cache_resource
def load_system():
    extractor = FASEExtractor()
    model = joblib.load('models/fase_lightgbm.pkl')
    return extractor, model

with st.spinner("Loading AI Firewall Models..."):
    extractor, classifier = load_system()

# --- Quick Category Heuristic ---
def guess_category(text):
    text = text.lower()
    if any(w in text for w in ['ignore', 'override', 'developer mode', 'hypothetically']): return "Jailbreak"
    if any(w in text for w in ['[', ']', 'command', 'print']): return "Prompt Injection"
    if any(w in text for w in ['idiot', 'hate', 'punch', 'kill']): return "Toxic"
    if any(w in text for w in ['ssn', 'address', 'password', 'credit card']): return "PII Leak"
    return "Malicious Intent"

# --- UI Layout ---
st.title("🛡️ SmartGuard AI Firewall")
st.markdown("Real-time LLM input guardrail. CPU-Optimized Semantic Ensemble.")

# Metrics Row
col1, col2, col3 = st.columns(3)
col1.metric("Prompts Scanned", st.session_state.total_scanned)
col2.metric("Threats Blocked", st.session_state.total_blocked)
col3.metric("Avg Latency (CPU)", f"{st.session_state.avg_latency:.2f} ms")

st.divider()

# Controls & Input
sidebar = st.sidebar
sidebar.header("Firewall Settings")
threshold = sidebar.slider("Strictness Threshold (Confidence)", 0.01, 0.99, 0.50, 0.01)

st.subheader("Test the Guardrail")
user_prompt = st.text_area("Enter a prompt intended for the LLM:", height=150)

if st.button("Submit to LLM", type="primary"):
    if user_prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        # Start timer for latency tracking
        start_time = time.time()
        
        # Extract & Predict
        features = extractor.transform([user_prompt])
        confidence_unsafe = classifier.predict(features)[0]
        
        # Calculate Latency
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Update metrics
        st.session_state.total_scanned += 1
        st.session_state.avg_latency = ((st.session_state.avg_latency * (st.session_state.total_scanned - 1)) + latency_ms) / st.session_state.total_scanned
        
        # --- RENDER THE OUTPUT ---
        st.markdown("---")
        st.markdown("### 🔍 Firewall Analysis Result")
        
        if confidence_unsafe >= threshold:
            st.session_state.total_blocked += 1
            st.error("🚨 **BLOCKED BY FIREWALL: Unsafe content detected.**")
            
            c1, c2, c3 = st.columns(3)
            c1.info(f"**Verdict:** UNSAFE")
            c2.warning(f"**Threat Confidence:** {confidence_unsafe:.1%}")
            c3.error(f"**Estimated Category:** {guess_category(user_prompt)}")
            
            st.code("Payload intercepted. The LLM did not receive this request.", language="bash")
        else:
            st.success("✅ **PASSED: Content is safe.**")
            
            c1, c2, c3 = st.columns(3)
            c1.info(f"**Verdict:** SAFE")
            c2.success(f"**Threat Confidence:** {confidence_unsafe:.1%}")
            c3.success(f"**Estimated Category:** Benign")
            
            st.code(f"LLM Processing Request: '{user_prompt}'", language="bash")
            
        st.caption(f"Processed in {latency_ms:.2f} milliseconds.")