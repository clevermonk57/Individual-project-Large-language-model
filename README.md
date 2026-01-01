# Individual-project-Large-language-model
Large language model Assignment 
Multi-Class Sentiment Analysis using BERT
Description

This project explores multi-class sentiment analysis on Yelp reviews using a fine-tuned BERT model. The task is to classify reviews into five sentiment categories (1–5 star ratings).
The performance of BERT is compared with a traditional TF-IDF + Logistic Regression baseline to evaluate the benefit of transformer-based models for sentiment classification.

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

Padding and truncation to a maximum of 128 tokens

Attention masks generation

Conversion to PyTorch tensors

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

The fine-tuned BERT model outperformed the baseline, demonstrating that contextual embeddings capture sentiment more effectively than traditional bag-of-words approaches.

Limitations

Limited training due to computational constraints

Minimal hyperparameter tuning

Only one transformer model was tested

Future Improvements

Experiment with other transformer models (RoBERTa, DistilBERT)

Increase the number of training epochs

Perform extensive hyperparameter tuning

Use larger subsets of the dataset

Tools & Libraries

Python

PyTorch

Hugging Face Transformers

Hugging Face Datasets

Scikit-learn

Notes

This project was completed as a student academic exercise to demonstrate practical understanding of NLP, transformer models, and sentiment analysis.
