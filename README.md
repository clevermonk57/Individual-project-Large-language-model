# Individual-project-Large-language-model
Large language model Assignment 
Multi-Class Sentiment Analysis using BERT
 Project Overview

This project explores multi-class sentiment analysis on Yelp reviews using a fine-tuned BERT model.
The task is to classify reviews into five sentiment categories (1–5 star ratings).

A traditional TF-IDF + Logistic Regression model is used as a baseline to evaluate the benefits of transformer-based contextual representations.

### Dataset

Yelp Review Full Dataset

Source: Hugging Face Datasets

Sentiment Classes: 5 (1–5 stars)

Subset Used:

🟢 40,000 training samples

🔵 10,000 test samples

🤖 Models Implemented
🔹 Baseline Model

TF-IDF + Logistic Regression

🔹 Transformer Model

BERT-base-uncased

Fine-tuned for multi-class classification

🧹 Data Preprocessing

The following preprocessing steps were applied:

BERT WordPiece tokenisation

Padding and truncation to 128 tokens

Attention mask generation

Conversion to PyTorch tensors

⚙️ Training Configuration
Setting	Value
Optimiser	AdamW
Loss Function	Cross-entropy
Batch Size	16
Epochs	2
📈 Evaluation Metrics

Model performance was measured using:

Accuracy

Macro Precision

Macro Recall

Macro F1-score

🧪 Results
Model	Accuracy	F1-score (Macro)
BERT	61.19%	60.97%
TF-IDF + Logistic Regression	56.51%	56.26%

✔️ BERT outperformed the baseline, showing that contextual embeddings capture sentiment more effectively than traditional bag-of-words methods.

⚠️ Limitations

Training limited due to computational constraints

Minimal hyperparameter tuning

Only one transformer model evaluated

🚀 Future Improvements

Evaluate other transformer models (RoBERTa, DistilBERT)

Increase training epochs

Perform extensive hyperparameter tuning

Use larger training subsets

🛠️ Tools & Libraries

Python

PyTorch

Hugging Face Transformers

Hugging Face Datasets

Scikit-learn

📚 Notes

This repository represents a student academic project demonstrating practical understanding of:

Natural Language Processing

Transformer models

Sentiment analysis
