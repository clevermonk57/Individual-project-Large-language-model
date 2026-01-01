# Individual-project-Large-language-model
Large language model Assignment 
Multi-Class Sentiment Analysis using BERT
Description

This project explores multi-class sentiment analysis on Yelp reviews using a fine-tuned BERT model. The task is to classify reviews into five sentiment categories (1–5 star ratings). The performance of BERT is compared with a traditional TF-IDF + Logistic Regression baseline to evaluate the benefit of transformer-based models for sentiment classification.

Dataset

Dataset: Yelp Review Full

Source: Hugging Face Datasets

Classes: 5 (1–5 stars)

Data used:

40,000 training samples

10,000 test samples

Models

Baseline Model: TF-IDF + Logistic Regression

Main Model: BERT-base-uncased (fine-tuned for classification)

Preprocessing

BERT WordPiece tokenisation

Padding and truncation to 128 tokens

Attention masks created

Converted to PyTorch tensors

Training Setup

Optimiser: AdamW

Loss function: Cross-entropy

Batch size: 16

Epochs: 2

Evaluation

The models were evaluated using:

Accuracy

Macro Precision

Macro Recall

Macro F1-score

Results
Model	Accuracy	F1-score (Macro)
BERT	61.19%	60.97%
TF-IDF + Logistic Regression	56.51%	56.26%

BERT outperformed the baseline model, showing that contextual embeddings help capture sentiment more effectively than traditional bag-of-words methods.

Limitations

Limited training due to computational constraints

Minimal hyperparameter tuning

Only one transformer model was tested

Future Improvements

Test other transformer models (RoBERTa, DistilBERT)

Increase training epochs

Perform hyperparameter tuning

Use larger training subsets

Tools & Libraries

Python

PyTorch

Hugging Face Transformers

Hugging Face Datasets

Scikit-learn

Notes

This project was completed as a student academic exercise to demonstrate practical understanding of NLP, transformer models, and sentiment analysis.
