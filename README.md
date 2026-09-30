# E-Waste Topic Modeling

## Live Demo: [https://gaoxanh-e-waste-topic-modeling.streamlit.app/](url)
## Overview

This project applies Data Mining and Natural Language Processing techniques to analyze negative consumer reviews of cell phones and accessories.

The main objective is to discover recurring complaint topics, analyze their temporal patterns, forecast short-term topic prevalence, and group products into complaint profiles. The results are then used to identify complaint patterns that may represent potential e-waste-related signals.

The project follows the pipeline:

Amazon Reviews
→ Negative Complaint Filtering
→ NLP Preprocessing
→ LDA Topic Modeling
→ Temporal Analysis & Forecasting
→ ASIN-level K-Means Clustering
→ Failure-related Profile Analysis
→ Streamlit Application

---

## Research Question

The project investigates the following questions:

1. What are the major complaint topics appearing in negative consumer reviews?
2. Which complaint topics are associated with potential failure-related patterns?
3. How do selected failure-related complaint topics change over time?
4. Can short-term topic prevalence be forecast from historical temporal patterns?
5. Can products be grouped into distinct complaint profiles based on their topic distributions?
6. What proportion of clustered ASINs and complaint reviews belongs to failure-related profiles?

---

## Dataset

The project uses the **Amazon Cell Phones and Accessories 5-core dataset**.

The dataset contains consumer reviews of products in the Cell Phones and Accessories category.

The original dataset contains approximately:

- 1.13 million reviews
- 48,000+ products
- Review ratings from 1 to 5 stars
- Review text, summary, product ASIN, verification status, votes, and review dates

For complaint analysis, reviews with ratings of **1 or 2 stars** are selected as a proxy for negative consumer experiences.

After cleaning and duplicate removal, the final complaint corpus contains:

**138,671 negative reviews**

After NLP preprocessing and removal of empty tokenized documents, the final LDA corpus contains:

**130,738 reviews**

> Note: A 1–2 star rating represents a negative consumer experience and is used as a complaint proxy. It does not necessarily prove that the product experienced a physical failure.

---

## Methodology

### 1. Data Exploration

The original Amazon review dataset is explored to understand:

- Dataset size
- Product coverage
- Rating distribution
- Review length
- Review dates
- Verified purchases
- Review votes
- Missing and duplicate review text

---

### 2. Complaint Filtering

Reviews with ratings of 1 or 2 stars are selected.

The text data is then cleaned by:

- Removing missing review text
- Removing duplicate review text
- Preserving review metadata
- Preparing review dates for temporal analysis

---

### 3. NLP Preprocessing

The complaint corpus is processed using NLTK.

The preprocessing pipeline includes:

1. Text normalization
2. HTML and URL removal
3. Tokenization
4. Contraction normalization
5. Negation preservation
6. POS tagging
7. WordNet lemmatization
8. Stopword removal

Negation terms such as:

- `no`
- `not`
- `never`
- `neither`
- `nor`

are preserved because they can change the meaning of a complaint.

The final processed tokens are used to construct the document-term matrix.

---

### 4. LDA Topic Modeling

Latent Dirichlet Allocation (LDA) is used to discover recurring complaint topics.

The final model contains:

**8 topics**

The identified topics are:

| Topic | Interpretation |
|---|---|
| T0 | Screen |
| T1 | Durability / Work-break |
| T2 | Case / Cover |
| T3 | Return / Order |
| T4 | Phone / Call |
| T5 | Charging / Battery |
| T6 | Fit / Quality |
| T7 | Mount / Holder |

The LDA model produces a topic probability distribution for each complaint review.

The dominant topic is determined by the topic with the highest probability for each review.

---

### 5. Temporal Analysis and Forecasting

Monthly topic prevalence is calculated from the LDA topic probabilities.

The temporal dataset covers the period from **2012 to 2018**.

For forecasting:

- 2012–2017 is used as the training period
- January–September 2018 is used as the test period
- October 2018 is excluded because it contains only one review

Lag features of 1, 2, and 3 months are constructed.

Short-term forecasting is evaluated for three selected topics:

- T0 – Screen
- T1 – Durability / Work-break
- T5 – Charging / Battery

The final models selected based on test RMSE are:

| Topic | Model |
|---|---|
| T0 – Screen | Linear Regression |
| T1 – Durability / Work-break | Linear Regression |
| T5 – Charging / Battery | Persistence Baseline |

The forecasting experiment is intended as a **short-term topic prevalence analysis**, not as a prediction of future e-waste volume.

---

### 6. ASIN-level K-Means Clustering

LDA topic probabilities are aggregated at the ASIN level.

ASINs with fewer than 3 complaint reviews are excluded from clustering to reduce instability caused by extremely small complaint samples.

This results in:

**13,932 ASINs**

Eight topic probability features are standardized and used for K-Means clustering.

The number of clusters is evaluated using silhouette scores.

The final model uses:

**K = 8**

The resulting profiles are:

| Cluster | Complaint Profile |
|---|---|
| C0 | Phone / Call |
| C1 | Case / Cover |
| C2 | Screen |
| C3 | Mount / Holder |
| C4 | Durability / Work-break |
| C5 | Charging / Battery |
| C6 | Return / Order |
| C7 | Fit / Quality |

---

### 7. Failure-related Profiles

Three clusters are considered failure-related complaint profiles for further analysis:

- **C2 – Screen**
- **C4 – Durability / Work-break**
- **C5 – Charging / Battery**

Among the ASINs retained for clustering:

**34.23%** belong to these three failure-related profiles.

These profiles account for:

**37.85%** of the complaint reviews represented in the clustering dataset.

These values should be interpreted as **complaint-profile signals**, not as direct measurements of e-waste generation.

---

## Results

The analysis identifies several recurring complaint patterns.

The overall topic prevalence shows that:

- Case / Cover complaints form one of the largest topic groups.
- Fit / Quality complaints also represent a substantial proportion of the corpus.
- Screen, Durability / Work-break, and Charging / Battery form the selected failure-related complaint profiles.

The temporal analysis shows that failure-related complaint topics exhibit different patterns over time.

The forecasting experiment demonstrates that short-term topic prevalence can be modeled using historical topic distributions, although the limited 2018 test period means that the forecasts should not be interpreted as long-term predictions.

The K-Means analysis provides an ASIN-level view of different complaint profiles and allows failure-related profiles to be isolated for further analysis.

---

## Streamlit Application

The project includes a Streamlit application for interactive exploration.

The application contains three main functions:

### 1. Project Overview

Displays:

- Number of complaint reviews
- Number of LDA topics
- Number of ASIN profiles
- Failure-related ASIN share
- Failure-related complaint-review share
- Overall topic distribution

### 2. ASIN Complaint Profile

Users can select an ASIN to view:

- Cluster assignment
- Complaint profile
- Failure-related status
- Topic probability distribution

### 3. Review Analyzer

Users can enter a negative review and the application:

1. Applies the same NLP preprocessing pipeline used during model training.
2. Transforms the review using the trained CountVectorizer.
3. Infers the topic distribution using the trained LDA model.
4. Identifies the dominant complaint topic.
5. Displays the topic probability profile.
6. Indicates whether the dominant topic belongs to a failure-related topic group.

---

## Project Structure

```text
e-waste-topic-modeling/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── processed/
│       ├── complaints_1_2star.pkl
│       ├── complaints_nlp.pkl
│       ├── complaints_nlp_clean.pkl
│       ├── complaints_topic_model.pkl
│       ├── monthly_topic_lag_features.pkl
│       ├── topic_forecasts.pkl
│       ├── forecast_summary.pkl
│       ├── asin_kmeans_profiles.pkl
│       ├── asin_kmeans_summary.pkl
│       ├── failure_cluster_summary.pkl
│       └── failure_profile_share.pkl
│
├── models/
│   ├── count_vectorizer.pkl
│   └── lda_model.pkl
│
├── notebooks/
│   ├── 01_explore_data.ipynb
│   ├── 01b_complaints_1_2star.ipynb
│   ├── 02_nlp_preprocessing.ipynb
│   ├── 03_topic_modeling.ipynb
│   ├── 04_prediction.ipynb
│   ├── 05_clustering.ipynb
│   └── 06_visualization_report.ipynb
│
└── src/
    └── 01_prepare_data.py# e-waste-topic-modeling
