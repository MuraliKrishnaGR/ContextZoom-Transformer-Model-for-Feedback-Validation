import streamlit as st
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np

# Load model and tokenizer
MODEL_PATH = "model"  
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

# Prediction function with threshold 0.7
def predict_value(text, reason, threshold=0.7):
    combined_text = text + " [SEP] " + reason
    encoding = tokenizer(combined_text, padding="max_length", truncation=True, max_length=128, return_tensors="tf")
    
    logits = model.predict(dict(encoding)).logits
    probs = tf.nn.softmax(logits, axis=1).numpy()
    
    prediction = 1 if probs[:, 1] > threshold else 0
    return prediction, probs[:, 1][0]

# Streamlit UI
st.set_page_config(page_title="Text-Reason Evaluator", layout="wide")

# Custom CSS for a unique futuristic design
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            color: white;
            font-family: 'Arial', sans-serif;
        }
        h1 {
            text-align: center;
            color: #00c6ff;
        }
        .stTextArea label, .stButton button {
            font-weight: bold;
            color: white;
        }
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: 1px solid white;
        }
        .stButton button {
            background: #00c6ff;
            color: black;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
            transition: 0.3s;
        }
        .stButton button:hover {
            background: #0072ff;
        }
    </style>
""", unsafe_allow_html=True)

# Page Title
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.title("🚀 Text & Reason Evaluator")
st.markdown("**Analyze if the provided text and reason are valuable!**")

# User inputs
text_input = st.text_area("📝 Enter the Text:", height=100)
reason_input = st.text_area("💡 Enter the Reason:", height=100)

# Predict Button
if st.button("🔍 Predict"):
    if text_input.strip() == "" or reason_input.strip() == "":
        st.warning("⚠️ Please enter both Text and Reason!")
    else:
        prediction, confidence = predict_value(text_input, reason_input)
        if prediction == 1:
            st.success("✅ This feedback is Valuable!")
            st.markdown(f"<b>📜 Text:</b> {text_input}", unsafe_allow_html=True)
            st.markdown(f"<b>🔍 Reason:</b> {reason_input}", unsafe_allow_html=True)
            st.markdown(f"<b>📊 Confidence Score:</b> {confidence:.2f}", unsafe_allow_html=True)
        else:
            st.error("❌ This feedback is Not Valuable.")

st.markdown('</div>', unsafe_allow_html=True)
