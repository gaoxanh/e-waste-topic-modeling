import json
import pandas as pd
from pathlib import Path


# =========================
# PATH
# =========================

INPUT_FILE = Path("data/Cell_Phones_and_Accessories_5.json")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOAD AMAZON JSON
# =========================

reviews = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        reviews.append(json.loads(line))


df = pd.DataFrame(reviews)

print("Original shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# =========================
# KEEP REQUIRED COLUMNS
# =========================

columns_to_keep = [
    "asin",
    "reviewText",
    "summary",
    "overall",
    "reviewTime",
    "unixReviewTime",
    "verified",
    "vote"
]

df = df[
    [col for col in columns_to_keep if col in df.columns]
]


# =========================
# REMOVE EMPTY REVIEW TEXT
# =========================

df["reviewText"] = df["reviewText"].fillna("").astype(str)

df = df[
    df["reviewText"].str.strip() != ""
]


# =========================
# CREATE YEAR
# =========================

df["review_date"] = pd.to_datetime(
    df["unixReviewTime"],
    unit="s",
    errors="coerce"
)

df["year"] = df["review_date"].dt.year


# =========================
# CREATE COMPLAINT CORPUS
# =========================

complaints = df[
    df["overall"].isin([1.0, 2.0])
].copy()


# =========================
# SAVE
# =========================

df.to_csv(
    OUTPUT_DIR / "amazon_cellphones_clean.csv",
    index=False
)

complaints.to_csv(
    OUTPUT_DIR / "amazon_complaints_1_2star.csv",
    index=False
)


# =========================
# SUMMARY
# =========================

print("\n========== SUMMARY ==========")

print("All reviews:", len(df))

print("1-2 star reviews:", len(complaints))

print(
    "Complaint percentage:",
    round(len(complaints) / len(df) * 100, 2),
    "%"
)

print("\nRating distribution:")
print(df["overall"].value_counts().sort_index())

print("\nYear distribution:")
print(df["year"].value_counts().sort_index())

print("\nSaved files:")
print(OUTPUT_DIR / "amazon_cellphones_clean.csv")
print(OUTPUT_DIR / "amazon_complaints_1_2star.csv")