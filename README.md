# Toxic_Comment_Classification
project focuses on building a robust multi-label toxic comment classification system using both traditional machine learning models and state-of-the-art transformer-based architectures like BERT.  The goal is to accurately classify user-generated text into multiple toxicity categories such as:  Toxic Severe Toxic Obscene Threat Insult Identity Hate.
A comprehensive Machine Learning-based Toxic Content Detection System that automatically identifies and classifies harmful online comments into multiple toxicity categories. The project combines Natural Language Processing (NLP), feature engineering, sentiment analysis, and machine learning techniques to build an intelligent content moderation solution.

📌 Overview

Online platforms generate millions of comments daily, making manual moderation nearly impossible. This project aims to automate content moderation by detecting:

Toxic Comments
Severe Toxicity
Obscene Language
Threats
Insults
Identity-Based Hate Speech

The system preprocesses raw text, extracts linguistic and behavioral features, generates TF-IDF representations, and prepares the data for multi-label classification models.

🚀 Features
Text Processing
URL removal
Mention removal
Special character cleaning
Lowercasing
Whitespace normalization
NLP Pipeline
Tokenization
Stop-word removal
Lemmatization using spaCy
Feature Engineering
Basic Features
Text Length
Word Count
Behavioral Features
Uppercase Character Count
Capitalization Ratio
Punctuation Count
Exclamation Count
Question Mark Count
Digit Count
Toxicity Features
Toxic Keyword Count
Toxic Keyword Ratio
Text Complexity Features
Average Word Length
Unique Word Ratio
Sentiment Analysis
Polarity Score using TextBlob
🏗️ Project Architecture
Raw Comments
      │
      ▼
Text Cleaning
      │
      ▼
Lemmatization
      │
      ▼
Feature Engineering
      │
      ├── Text Features
      ├── Behavioral Features
      ├── Toxicity Features
      ├── Complexity Features
      └── Sentiment Features
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Train/Test Split
      │
      ▼
Multi-Label Classification
      │
      ▼
Prediction
      │
      ▼
Content Moderation Decision
📊 Dataset

This project uses the widely adopted Jigsaw Toxic Comment Classification Dataset.

Target Labels
Label	Description
toxic	General toxic comments
severe_toxic	Extremely toxic comments
obscene	Obscene language
threat	Threatening content
insult	Personal attacks
identity_hate	Hate speech targeting identities
🛠️ Technologies Used
Programming Language
Python
Libraries
Pandas
NumPy
spaCy
TextBlob
Scikit-Learn
tqdm
Regular Expressions (re)
NLP
Lemmatization
Stop-word Removal
Sentiment Analysis
TF-IDF
📂 Project Structure
toxic_content_detection/
│
├── data/
│   ├── train.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│
├── models/
│   ├── toxic_model.pkl
│   ├── tfidf_vectorizer.pkl
│
├── outputs/
│   ├── processed_data.csv
│
├── requirements.txt
│
└── README.md
📈 Extracted Features
Feature	Purpose
text_length	Measures comment size
word_count	Counts words
uppercase_count	Detects shouting
punctuation_count	Measures punctuation usage
exclamation_count	Aggression signal
question_count	Question intensity
digit_count	Numeric content
caps_ratio	Uppercase proportion
toxic_word_count	Toxic vocabulary frequency
toxic_word_ratio	Toxicity density
avg_word_length	Linguistic complexity
unique_word_ratio	Spam/repetition detection
sentiment	Emotional polarity
1) using Random Forest & NLP
A machine learning-based content moderation system that automatically identifies and classifies toxic online comments into multiple harmful content categories. This project combines Natural Language Processing (NLP), feature engineering, TF-IDF vectorization, Random Forest classification, and threshold optimization to build an effective multi-label toxic comment detection pipeline.

📖 Project Overview

Online platforms receive millions of comments every day, making manual moderation difficult and expensive. This project aims to automate content moderation by detecting different forms of toxicity in user-generated text.

The model predicts six toxicity categories simultaneously:

Toxic
Severe Toxic
Obscene
Threat
Insult
Identity Hate

The system uses extensive text preprocessing, feature engineering, TF-IDF representations, and a One-vs-Rest Random Forest classifier for multi-label classification.

🎯 Objectives
Detect toxic comments automatically.
Classify multiple toxicity categories simultaneously.
Improve moderation efficiency.
Reduce harmful content exposure.
Build an interpretable NLP-based moderation system.
🏗️ Project Pipeline
Raw Comments
      │
      ▼
Text Cleaning
      │
      ▼
Lemmatization (spaCy)
      │
      ▼
Feature Engineering
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Train-Test Split
      │
      ▼
One-vs-Rest Random Forest
      │
      ▼
Probability Prediction
      │
      ▼
Threshold Optimization
      │
      ▼
Multi-Label Classification
      │
      ▼
Evaluation & Visualization
📂 Dataset

Dataset: Jigsaw Toxic Comment Classification Challenge

The dataset contains Wikipedia comments labeled with six toxicity categories.

Label	Description
toxic	General toxic content
severe_toxic	Extremely toxic content
obscene	Profanity and vulgar language
threat	Threatening comments
insult	Personal attacks
identity_hate	Hate speech targeting groups
🔍 Data Preprocessing

The preprocessing pipeline includes:

Text Cleaning
Lowercasing
URL removal
Mention removal
Special character removal
Extra whitespace removal
NLP Processing
Tokenization
Stop-word removal
Lemmatization using spaCy
Sentiment Analysis

Using TextBlob polarity scores.

⚙️ Feature Engineering

The system extracts several handcrafted features.

Basic Features
Feature	Description
text_length	Number of characters
word_count	Number of words
Behavioral Features
Feature	Description
uppercase_count	Uppercase characters
punctuation_count	Punctuation frequency
exclamation_count	Number of exclamation marks
question_count	Number of question marks
digit_count	Numeric content
caps_ratio	Ratio of uppercase characters
Toxicity Features
Feature	Description
toxic_word_count	Toxic keyword frequency
toxic_word_ratio	Toxic keyword density
Text Complexity Features
Feature	Description
avg_word_length	Average word length
unique_word_ratio	Vocabulary diversity
Sentiment Feature
Feature	Description
sentiment	TextBlob polarity score
🧠 Model Architecture
TF-IDF Vectorization
TfidfVectorizer(
    max_features=30000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)
Classifier
OneVsRestClassifier(
    RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
)
🎯 Threshold Optimization

Instead of using a fixed threshold of 0.5, the model automatically searches for the best threshold for each toxicity class.

Optimal thresholds:

toxic          : 0.35
severe_toxic   : 0.25
obscene        : 0.30
threat         : 0.10
insult         : 0.30
identity_hate  : 0.15

This improves recall for rare classes such as Threat and Identity Hate.

📊 Model Performance
Classification Report
Class	Precision	Recall	F1-Score
Toxic	0.74	0.70	0.72
Severe Toxic	0.21	0.55	0.30
Obscene	0.67	0.82	0.74
Threat	0.37	0.56	0.44
Insult	0.55	0.73	0.63
Identity Hate	0.18	0.59	0.28
Overall Metrics
Metric	Score
Micro F1	0.63
Macro F1	0.52
Weighted F1	0.66
ROC-AUC	0.963
📈 Visualizations

The project automatically generates:

Precision vs Recall Comparison

Shows precision and recall for each toxicity class.

F1 vs Threshold Curves

Visualizes optimal threshold selection.

Confusion Matrices

Per-label confusion matrices for:

Toxic
Severe Toxic
Obscene
Threat
Insult
Identity Hate
🛠️ Technologies Used
Programming Language
Python
Machine Learning
Scikit-Learn
Random Forest
One-vs-Rest Classification
NLP
spaCy
TextBlob
TF-IDF
Data Analysis
Pandas
NumPy
Visualization
Matplotlib
