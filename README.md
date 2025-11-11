💬 Twitter Sentiment Analyzer

This project is a **Twitter Sentiment Analysis App** built using **Python**, **Machine Learning**, and **Streamlit**.  
It analyzes tweets and classifies them into **Positive**, **Negative**, or **Neutral** sentiments.

---

 🚀 Features
- Classifies tweets into 3 categories: 😊 Positive, 😐 Neutral, 😡 Negative  
- Built using Logistic Regression and TF-IDF Vectorization  
- Real-time analysis using Streamlit web app  
- Confidence score and detailed insights for each tweet  
- Export results as CSV  
- Clean dark-themed interface with charts  

📦 Dataset Download & Setup

This project uses three main datasets to train and test the Twitter Sentiment Analyzer (3-Class) model.

🧠 1️⃣ Sentiment140 Dataset (Main Source)

📁 File Name: training.1600000.processed.noemoticon.csv

🧾 Description: Contains 1.6 million labeled tweets with sentiment values:

0 → 😡 Negative

4 → 😊 Positive

🌐 Download From:
👉 Kaggle – Sentiment140 Dataset

⚙️ 2️⃣ Generated Neutral Tweets

Because the Sentiment140 dataset has no neutral tweets,

2 → 😐 Neutral

📊 It contains around 800,000 factual & emotionless sentences such as:

🔄 3️⃣ Merged & Balanced Dataset

To train a model, both datasets are combined using the merge_datasets.py script.

📈 This merged dataset contains balanced samples across:

0 → 😡 Negative

2 → 😐 Neutral

4 → 😊 Positive

🎓 Project Overview & Workflow

The Twitter Sentiment Analyzer (3-Class) project combines Machine Learning, NLP, and Data Visualization to classify tweets as
😊 Positive, 😐 Neutral, or 😡 Negative.

🧭 Workflow Summary
🪄 Step 1: Data Collection

📥 Collects and combines two datasets:

Sentiment140 Dataset (1.6M tweets) — from Kaggle

Neutral Tweets (800K) — generated using generate_neutral_tweets.py

Creates a combined dataset:
📁 training_balanced.csv

🧹 Step 2: Data Preprocessing

Cleans and prepares the text using Python + Regular Expressions:

Removes URLs, mentions (@user), hashtags, emojis, and punctuation

Converts to lowercase

Removes duplicates and blank tweets

✅ Output: Cleaned tweet column (cleaned_text)

🧠 Step 3: Feature Extraction (TF-IDF)

Text data is converted into numerical vectors using TF-IDF Vectorization.

This helps the model understand which words are important based on frequency.

Example:

“I love this phone!” → [love, phone, amazing, happy...]

⚙️ Step 4: Model Training

Trains a Multiclass Logistic Regression model to classify sentiment:

0 → Negative

1 → Neutral

2 → Positive

Model learns from 2.4M+ tweets and saves:

sentiment_model_3class.pkl

📊 Step 5: Model Evaluation

Evaluates using:

✅ Accuracy

📋 Classification Report

🔢 Confusion Matrix

🌐 Step 6: Web App Integration

Uses Streamlit to create an interactive web dashboard (app.py):

Input multiple tweets

Predict sentiment in real time

Visualize results with:

Donut chart (sentiment distribution)

Confidence bar chart

Detailed tweet insights

Option to export results as CSV

🧩 Full System Workflow

📂 Datasets (Sentiment140 + Neutral Tweets)
        │
        ▼
🧹 Preprocessing (Cleaning & Normalizing Text)
        │
        ▼
🔤 TF-IDF Vectorization (Feature Extraction)
        │
        ▼
🤖 Logistic Regression (Model Training)
        │
        ▼
💾 sentiment_model_3class.pkl (Saved Model)
        │
        ▼
🌐 Streamlit App (User Input → Real-time Sentiment Output)

🎯 End Goal

A fully working AI-powered Twitter Sentiment Analysis Dashboard
that predicts emotions from tweets instantly and visually.
