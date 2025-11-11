"""
=========================================================
SENTIMENT140 - 3 CLASS SENTIMENT MODEL TRAINING (FINAL)
=========================================================

Classes:
- 0 → Negative 😡
- 1 → Neutral 😐
- 2 → Positive 😊

Dataset:
- training_balanced.csv (merged Sentiment140 + neutral_tweets.csv)

Output:
- sentiment_model_3class.pkl (includes model + vectorizer)
"""

import pandas as pd
import numpy as np
import re
import string
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Enable tqdm progress bars
tqdm.pandas()

# ==========================================================
# SENTIMENT MODEL TRAINER CLASS
# ==========================================================
class SentimentModelTrainer:
    def __init__(self, dataset_path, model_output_path="sentiment_model_3class.pkl"):
        self.dataset_path = dataset_path
        self.model_output_path = model_output_path
        self.vectorizer = None
        self.model = None
        self.df = None

    # -------------------- LOAD DATA --------------------
    def load_data(self):
        print("📂 Loading dataset...")
        columns = ["target", "id", "date", "flag", "user", "text"]

        try:
            self.df = pd.read_csv(
                self.dataset_path,
                encoding="latin-1",
                names=columns,
                header=None
            )
            print(f"✅ Loaded dataset: {len(self.df):,} tweets")
            print(f"📊 Shape: {self.df.shape}")
            print(f"\n📈 Raw label distribution:\n{self.df['target'].value_counts()}")
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ File not found: {self.dataset_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")

    # -------------------- CLEAN TEXT --------------------
    def clean_text(self, text):
        """Clean tweet text (light cleaning to preserve neutral content)."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)        # Remove URLs
        text = re.sub(r"@\w+", "", text)                  # Remove mentions
        text = re.sub(r"#", "", text)                     # Keep hashtag words
        text = re.sub(r"[^\x00-\x7F]+", " ", text)        # Remove non-ASCII
        text = re.sub(r"\s+", " ", text).strip()          # Normalize spaces
        return text

    # -------------------- PREPROCESS --------------------
    def preprocess_data(self):
        print("\n🧹 Preprocessing data...")
        self.df = self.df[["target", "text"]]

        print("🔄 Mapping sentiment labels (0, 2, 4 → 0, 1, 2)...")
        self.df["target"] = pd.to_numeric(self.df["target"], errors="coerce")
        self.df["target"] = self.df["target"].map({0: 0, 2: 1, 4: 2})

        before = len(self.df)
        self.df.dropna(subset=["target", "text"], inplace=True)
        print(f"🗑️ Removed {before - len(self.df):,} invalid or missing rows")

        print("🧼 Cleaning tweets (may take a few minutes)...")
        self.df["cleaned_text"] = self.df["text"].astype(str).progress_apply(self.clean_text)

        before = len(self.df)
        self.df = self.df[self.df["cleaned_text"].str.len() > 0]
        print(f"🗑️ Removed {before - len(self.df):,} empty tweets after cleaning")

        before = len(self.df)
        self.df = self.df[~((self.df["target"] == 1) & (self.df.duplicated(subset=["cleaned_text"])))]
        print(f"🗑️ Removed {before - len(self.df):,} duplicates (kept neutrals)")

        print(f"\n📊 Label distribution:\n{self.df['target'].value_counts()}")
        counts = self.df["target"].value_counts()

        max_count = counts.max()
        min_count = counts.min()

        if min_count / max_count < 0.5:
            print("⚙️ Oversampling neutral tweets to balance dataset...")
            neutral_df = self.df[self.df["target"] == 1]
            repeats = int(max_count / len(neutral_df))
            neutral_df = pd.concat([neutral_df] * repeats, ignore_index=True)
            self.df = pd.concat([
                self.df[self.df["target"] != 1],
                neutral_df
            ], ignore_index=True)
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            print("⚖️ Dataset already fairly balanced.")

        print(f"✅ Final label distribution:\n{self.df['target'].value_counts()}")

    # -------------------- FEATURE CREATION --------------------
    def create_features(self):
        print("\n🔢 Creating TF-IDF features...")
        X = self.df["cleaned_text"]
        y = self.df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"📊 Training samples: {len(X_train):,}")
        print(f"📊 Test samples: {len(X_test):,}")

        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.85,
            stop_words="english"
        )

        print("🔄 Fitting TF-IDF vectorizer...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        print(f"✅ TF-IDF Feature shape: {X_train_tfidf.shape}")
        return X_train_tfidf, X_test_tfidf, y_train, y_test

    # -------------------- TRAIN MODEL --------------------
    def train_model(self, X_train, y_train):
        print("\n🤖 Training Logistic Regression (multiclass)...")
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
            multi_class="multinomial",
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        print("✅ Model training complete!")

    # -------------------- EVALUATE --------------------
    def evaluate_model(self, X_test, y_test):
        print("\n📊 Evaluating model...")
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n🎯 Test Accuracy: {acc*100:.2f}%\n")
        print(classification_report(
            y_test, y_pred,
            labels=[0, 1, 2],
            target_names=["Negative", "Neutral", "Positive"],
            digits=4,
            zero_division=0
        ))
        return acc

    # -------------------- SAVE MODEL --------------------
    def save_model(self):
        print(f"\n💾 Saving model to: {self.model_output_path}")
        model_data = {
            "model": self.model,
            "vectorizer": self.vectorizer,
            "feature_names": self.vectorizer.get_feature_names_out()
        }
        joblib.dump(model_data, self.model_output_path)
        print("✅ Model saved successfully!")

    # -------------------- RUN PIPELINE --------------------
    def run_pipeline(self):
        print("="*65)
        print("🚀 SENTIMENT140 (BALANCED) — 3-CLASS MODEL TRAINING")
        print("="*65)

        self.load_data()
        self.preprocess_data()
        X_train, X_test, y_train, y_test = self.create_features()
        self.train_model(X_train, y_train)
        acc = self.evaluate_model(X_test, y_test)
        self.save_model()

        print("\n" + "="*65)
        print(f"✅ PIPELINE COMPLETE — FINAL ACCURACY: {acc*100:.2f}%")
        print(f"💾 Model saved as: {self.model_output_path}")
        print("="*65)

# ==========================================================
# MAIN FUNCTION
# ==========================================================
def main():
    DATASET_PATH = "training_balanced.csv"  # Your merged dataset
    MODEL_OUTPUT_PATH = "sentiment_model_3class.pkl"

    trainer = SentimentModelTrainer(DATASET_PATH, MODEL_OUTPUT_PATH)
    trainer.run_pipeline()

if __name__ == "__main__":
    main()
