# src/data_processing.py

import pandas as pd
import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji
from pathlib import Path

# ======================
# NLTK setup (safe)
# ======================
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ======================
# Paths
# ======================
RAW_PATH = Path("data/raw/complaints.csv")
PROCESSED_PATH = Path("data/processed/cleaned_complaints_sample.csv")

# ======================
# Product mapping (EXACT)
# ======================
mapping_rules = {
    "Personal loan": [
        "Payday loan, title loan, personal loan, or advance loan",
        "Student loan",
        "Payday loan, title loan, or personal loan",
        "Consumer Loan"
    ],
    "Money transfers": [
        "Money transfer, virtual currency, or money service",
        "Money transfers"
    ],
    "Savings account": [
        "Checking or savings account"
    ],
    "Credit card": [
        "Credit card or prepaid card",
        "Credit card"
    ]
}

PRODUCT_LOOKUP = {
    raw: mapped
    for mapped, raw_list in mapping_rules.items()
    for raw in raw_list
}

TARGET_PRODUCTS = list(mapping_rules.keys())

# ======================
# Text cleaning
# ======================
def clean_text(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove emails & phone numbers
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\b\d{10,}\b", "", text)

    # Remove emojis
    text = emoji.replace_emoji(text, replace="")

    # Remove xxx / xxxx / xxxxx patterns (case-insensitive)
    text = re.sub(r"\b[xX]{2,}\b", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Normalize
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize + lemmatize
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]

    return " ".join(tokens)

# ======================
# Main processing
# ======================
def process_complaints(df: pd.DataFrame, max_samples=100_000) -> pd.DataFrame:
    print("Mapping product categories...")
    df["Product_Category"] = df["Product"].map(PRODUCT_LOOKUP)

    # Keep only target products
    df = df[df["Product_Category"].isin(TARGET_PRODUCTS)].copy()

    # Remove empty narratives
    df = df.dropna(subset=["Consumer complaint narrative"])
    df = df[df["Consumer complaint narrative"].str.strip() != ""]

    print("Cleaning complaint text...")
    df["Complaint_Text"] = df["Consumer complaint narrative"].apply(clean_text)

    # Text length
    df["text_length"] = df["Complaint_Text"].str.len()

    # Remove extremely short complaints
    df = df[df["text_length"] >= 10]

    # Cap extremely long complaints
    df["Complaint_Text"] = df["Complaint_Text"].str.slice(0, 5000)

    # ======================
    # Stratified sampling (NO deprecated apply)
    # ======================
    print("Applying stratified sampling...")
    per_class = max_samples // len(TARGET_PRODUCTS)

    sampled_df = (
        df.groupby("Product_Category", group_keys=False)
          .sample(n=per_class, random_state=42)
    )

    return sampled_df

# ======================
# Pipeline
# ======================
def run_pipeline():
    print(f"Loading raw data from {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH, low_memory=False)

    processed_df = process_complaints(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving processed data to {PROCESSED_PATH} ...")
    processed_df.to_csv(PROCESSED_PATH, index=False)

    print("Pipeline completed successfully!")
    print("\nFinal distribution:")
    print(processed_df["Product_Category"].value_counts())

    return processed_df

# ======================
# Entry point
# ======================
if __name__ == "__main__":
    final_df = run_pipeline()
    print("\nSample output:")
    print(final_df.head())
