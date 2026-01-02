# src/vector_pipeline.py

import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm

# ======================
# File paths
# ======================
CLEANED_CSV_PATH = "data/processed/cleaned_complaints_sample.csv"
VECTOR_STORE_PATH = "vector_store/faiss_index"

# ======================
# Sampling configuration
# ======================
TOTAL_SAMPLE_SIZE = 12000  # You can adjust between 10k-15k
TARGET_PRODUCTS = ["Credit card", "Personal loan", "Savings account", "Money transfers"]

# ======================
# Chunking configuration
# ======================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ======================
# Embedding model
# ======================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ======================
# Functions
# ======================
def stratified_sample(df: pd.DataFrame, target_col: str, total_sample_size: int):
    """Perform stratified sampling across target categories."""
    n_categories = df[target_col].nunique()
    per_category = total_sample_size // n_categories
    sampled_df = (
        df.groupby(target_col, group_keys=False)
        .apply(lambda x: x.sample(min(len(x), per_category), random_state=42))
    )
    return sampled_df.reset_index(drop=True)

def create_chunks(df: pd.DataFrame, text_column: str):
    """Split complaint narratives into chunks with metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    metadata = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking texts"):
        complaint_text = row[text_column]
        complaint_id = row["Complaint ID"]
        product = row["Product_Category"]
        chunk_texts = text_splitter.split_text(complaint_text)
        
        for idx, chunk in enumerate(chunk_texts):
            chunks.append(chunk)
            metadata.append({
                "complaint_id": complaint_id,
                "product_category": product,
                "chunk_index": idx
            })
    return chunks, metadata

# ======================
# Main pipeline
# ======================
def main():
    os.makedirs("vector_store", exist_ok=True)
    
    print(f"Loading cleaned complaints from {CLEANED_CSV_PATH} ...")
    df = pd.read_csv(CLEANED_CSV_PATH)
    
    # Keep only target products
    df = df[df["Product_Category"].isin(TARGET_PRODUCTS)].reset_index(drop=True)
    
    print("Applying stratified sampling (~12k complaints)...")
    df_sample = stratified_sample(df, target_col="Product_Category", total_sample_size=TOTAL_SAMPLE_SIZE)
    
    print("Creating text chunks ...")
    chunks, metadata = create_chunks(df_sample, text_column="Complaint_Text")
    
    print(f"Total chunks created: {len(chunks)}")
    
    print(f"Generating embeddings using {EMBEDDING_MODEL_NAME} ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print("Creating FAISS vector store ...")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=metadata)
    
    print(f"Saving vector store to {VECTOR_STORE_PATH} ...")
    vectorstore.save_local(VECTOR_STORE_PATH)
    
    print("Vector store creation completed successfully!")

# ======================
# Run script
# ======================
if __name__ == "__main__":
    main()
