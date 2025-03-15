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
st.set_page_config(page_title="Text-Reason Evaluator", layout="centered")

st.title("📄 Text & Reason Evaluator")
st.markdown("**Check if the given text and reason are valuable!**")

# User inputs
text_input = st.text_area("Enter the Text:", height=100)
reason_input = st.text_area("Enter the Reason:", height=100)

# Predict Button
if st.button("Predict"):
    if text_input.strip() == "" or reason_input.strip() == "":
        st.warning("⚠️ Please enter both Text and Reason!")
    else:
        prediction, confidence = predict_value(text_input, reason_input)
        label = "This feedback is ✅ Valuable" if prediction == 1 else "This feedback is ❌ Not Valuable"
        st.success(f"**Prediction:** {label}")
