# My Zoom: A Transformer-Based Model for Contextual Feedback Validation

## Overview

**My Zoom** is a transformer-based model leveraging BERT for contextual feedback validation. The model classifies textual feedback into two categories, validating whether the provided reason aligns with the given text. It uses data augmentation techniques such as synonym replacement and negative sampling to enhance training efficiency and improve generalization.

## Features
- Utilizes **BERT (bert-base-uncased)** for text classification.
- Implements **data augmentation** through synonym replacement.
- Balances dataset using **downsampling and class weighting**.
- Implements **custom threshold-based prediction**.
- **Fine-tuned** using Adam optimizer and cross-entropy loss.
- Deployed using **Streamlit** for an interactive interface.

## Installation

### Prerequisites
Ensure you have Python installed (>=3.8). Install dependencies using:
```bash
pip install -r requirements.txt
```

### Required Libraries
The project relies on the following:
```bash
pip install tensorflow pandas numpy nltk transformers scikit-learn streamlit
```
Additionally, download the required NLTK dataset:
```python
import nltk
nltk.download('wordnet')
```

## Dataset
The model is trained on an Excel dataset consisting of two key fields:
- **text**: The original input statement.
- **reason**: The explanation or justification for the statement.
- **label**: Binary labels (0 or 1) indicating whether the reason is valid.

## Data Preprocessing
- **Cleaning**: Removing numbers, URLs, HTML tags, and special characters.
- **Augmentation**: Synonym replacement applied with a probability of 30%.
- **Negative Sampling**: Generates false examples by mismatching text and reason.
- **Balancing**: Ensures class balance using resampling techniques.
- **Tokenization**: Uses BERT tokenizer with a max length of 128.

## Model Training
- **Pretrained Model**: Fine-tunes `bert-base-uncased` for sequence classification.
- **Optimizer**: Adam with a learning rate of `1e-5`.
- **Loss Function**: SparseCategoricalCrossentropy with logits.
- **Batch Size**: 16 for efficient training.
- **Epochs**: 7 epochs with validation monitoring.
- **Class Weights**: Adjusted to balance class distribution.

Training command:
```python
history = model.fit(train_dataset, validation_data=val_dataset, epochs=7, class_weight=class_weights_dict)
```

## Model Evaluation
The model achieves an **accuracy of 91.22% on validation data**.

### Classification Report:
```plaintext
              precision    recall  f1-score   support
           0       0.86      0.73      0.79      3001
           1       0.77      0.88      0.82      3001
    accuracy                           0.81      6002
   macro avg       0.81      0.81      0.80      6002
weighted avg       0.81      0.81      0.80      6002
```

## Model Saving & Deployment

The model is saved to Google Drive for easy access:

## Streamlit Deployment
A **Streamlit app** is used for real-time feedback validation.

Run the app:
```bash
streamlit run app.py
```

## Prediction with Custom Threshold
The model allows threshold-based classification:



