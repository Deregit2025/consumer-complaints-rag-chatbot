# tests/test_data_processing.py

import pytest
import pandas as pd
from src import data_processing as dp

# -------------------------
# Test Product Mapping
# -------------------------
def test_product_mapping():
    df = pd.DataFrame({"Product": ["Credit card", "Consumer Loan"]})
    df["Product_Category"] = df["Product"].map(dp.PRODUCT_LOOKUP)
    assert df["Product_Category"].tolist() == ["Credit card", "Personal loan"]

# -------------------------
# Test Text Cleaning
# -------------------------
def test_clean_text_removes_emails_and_x():
    raw = "Hello xxxx, contact me at test@example.com"
    cleaned = dp.clean_text(raw)
    assert "xxxx" not in cleaned
    assert "test" not in cleaned  # email removed

def test_clean_text_empty_input():
    assert dp.clean_text("") == ""
    assert dp.clean_text(None) == ""

# -------------------------
# Test Stratified Sampling
# -------------------------
def test_stratified_sampling_balance():
    df = pd.DataFrame({
        "Product": ["Credit card"]*50 + ["Personal Loan"]*50,
        "Consumer complaint narrative": ["Valid text"]*100
    })
    sampled = dp.process_complaints(df, max_samples=20)
    counts = sampled["Product_Category"].value_counts().to_dict()
    assert counts["Credit card"] == counts["Personal loan"]

# -------------------------
# Test Missing Columns
# -------------------------
def test_missing_columns():
    df = pd.DataFrame({"Product": ["Credit card"]})
    with pytest.raises(ValueError):
        dp.process_complaints(df)
