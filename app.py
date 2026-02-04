import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Beta-Lactamase Predictor", page_icon="🧬", layout="centered")

# --- 1. SETUP & LOADING ---
@st.cache_resource
def load_resources():
    if not os.path.exists('final_thesis_model.h5'):
        return None, None, "Model file 'final_thesis_model.h5' not found."
    model = tf.keras.models.load_model('final_thesis_model.h5')
    
    if not os.path.exists('dataset.csv'):
        return None, None, "Dataset file 'dataset.csv' not found."
    df = pd.read_csv("dataset.csv")
    df = df.dropna(subset=['Sequence'])
    
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(df['Sequence'])
    
    return model, tokenizer, None

model, tokenizer, error_msg = load_resources()

if error_msg:
    st.error(f"⚠️ System Error: {error_msg}")
    st.stop()

# --- 2. PREDICTION LOGIC ---
def predict_dna(seq):
    seq_clean = str(seq).upper().strip()
    valid_chars = set("ATCGN")
    if not set(seq_clean).issubset(valid_chars):
        return None, "Error: Invalid DNA characters detected."
    if len(seq_clean) < 50:
        return None, "Error: Sequence is too short (<50 bp)."

    max_len = 1200 
    seq_idx = tokenizer.texts_to_sequences([seq_clean])
    seq_pad = pad_sequences(seq_idx, maxlen=max_len, padding='post', truncating='post')
    
    pred_probs = model.predict(seq_pad)[0]
    return pred_probs, None

# --- 3. UI LAYOUT ---
st.title("🧬 Beta-Lactamase Activity Predictor")
st.markdown("M.Tech Thesis Project: Deep Learning for Genomic Surveillance")

input_type = st.radio("Input Method:", ["Paste Sequence", "Upload FASTA File"], horizontal=True)
sequence = ""

if input_type == "Paste Sequence":
    sequence = st.text_area("Paste DNA Sequence:", height=150)
elif input_type == "Upload FASTA File":
    uploaded_file = st.file_uploader("Upload .fasta or .txt file", type=["fasta", "txt"])
    if uploaded_file:
        # 1. Read and Decode
        stringio = uploaded_file.getvalue().decode("utf-8")
        
        # 2. Robust Parsing
        lines = stringio.splitlines()
        seq_parts = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and headers
            if not line or line.startswith(">"):
                continue
            
            # Clean the line: Remove numbers and spaces, keep only letters
            # (NCBI files sometimes have numbers like "1 agct 60")
            clean_line = "".join(filter(str.isalpha, line))
            seq_parts.append(clean_line)
        
        # 3. Join and Validate
        sequence = "".join(seq_parts)
        
        if len(sequence) > 0:
            st.success(f"✅ Successfully loaded {len(sequence)} bp")
            # Show a tiny preview to confirm it worked
            st.caption(f"Preview: {sequence[:50]}...")
        else:
            st.error("⚠️ Error: File was empty or contained only headers. Please check the file format.")
if st.button("Analyze Resistance Profile", type="primary"):
    if not sequence:
        st.warning("Enter a sequence first.")
    else:
        with st.spinner("Analyzing..."):
            probs, error = predict_dna(sequence)
            if error:
                st.error(error)
            else:
                st.markdown("---")
                st.subheader("Results")
                classes = ['Penicillinase', 'Cephalosporinase', 'Carbapenemase', 'Monobactamase']
                detected = []
                
                for i, class_name in enumerate(classes):
                    score = float(probs[i]) # <--- FIXED: Converted to standard float
                    percent = int(score * 100)
                    
                    st.write(f"**{class_name}**")
                    if score > 0.5:
                        st.progress(score, text=f"⚠️ {percent}% (RESISTANT)")
                        detected.append(class_name)
                    else:
                        st.progress(score, text=f"{percent}% (Susceptible)")
                
                st.markdown("---")
                if detected:
                    st.error(f"🚨 **CRITICAL ALERT:** Resistance to {', '.join(detected)}")
                else:
                    st.success("✅ **SAFE:** No significant resistance detected.")
