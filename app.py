from pathlib import Path
import joblib
import streamlit as st
import pandas as pd

# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"


# ============================================================
# 2. LOAD MODEL ARTIFACTS
# ============================================================

vectorizer = joblib.load(
    MODEL_DIR / "count_vectorizer.pkl"
)

lda_model = joblib.load(
    MODEL_DIR / "lda_model.pkl"
)


# ============================================================
# 3. CHECK MODEL
# ============================================================

st.title("E-Waste Topic Modeling")

st.write("Model loaded successfully.")

st.write(
    f"Number of LDA topics: {lda_model.n_components}"
)

st.write(
    f"Number of vocabulary features: {len(vectorizer.vocabulary_)}"
)

# ============================================================
# 4. LOAD PROJECT RESULTS
# ============================================================

processed_dir = PROJECT_ROOT / "data" / "processed"

topic_df = joblib.load(
    processed_dir / "complaints_topic_model.pkl"
)

cluster_df = joblib.load(
    processed_dir / "asin_kmeans_profiles.pkl"
)

failure_share = joblib.load(
    processed_dir / "failure_profile_share.pkl"
)


# ============================================================
# 5. TOPIC LABELS
# ============================================================

topic_names = {
    0: "T0 - Screen",
    1: "T1 - Durability",
    2: "T2 - Case/Cover",
    3: "T3 - Return/Order",
    4: "T4 - Phone/Call",
    5: "T5 - Charging/Battery",
    6: "T6 - Fit/Quality",
    7: "T7 - Mount/Holder"
}


# ============================================================
# 6. OVERVIEW
# ============================================================

st.header("Project Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Complaint Reviews",
        f"{len(topic_df):,}"
    )

with col2:
    st.metric(
        "LDA Topics",
        "8"
    )

with col3:
    st.metric(
        "ASIN Profiles",
        f"{cluster_df['asin'].nunique():,}"
    )


st.subheader("Failure-Related Profiles")

failure_asin_share = failure_share.loc[
    failure_share["metric"] == "Failure-related ASIN share (%)",
    "value"
].iloc[0]

failure_review_share = failure_share.loc[
    failure_share["metric"] == "Failure-related complaint review share (%)",
    "value"
].iloc[0]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Failure-related ASIN Share",
        f"{failure_asin_share:.2f}%"
    )

with col2:
    st.metric(
        "Failure-related Complaint Share",
        f"{failure_review_share:.2f}%"
    )


st.subheader("Topic Distribution")

topic_prob_cols = [
    f"topic_{i}_prob"
    for i in range(8)
]

topic_prevalence = (
    topic_df[topic_prob_cols]
    .mean()
    .sort_values(ascending=False)
)

topic_chart = topic_prevalence.rename(
    lambda x: topic_names[int(x.split("_")[1])]
)

st.bar_chart(topic_chart)

# ============================================================
# 7. ASIN PROFILE
# ============================================================

st.divider()

st.header("ASIN Complaint Profile")

asin_options = sorted(
    cluster_df["asin"].dropna().unique()
)

selected_asin = st.selectbox(
    "Select an ASIN",
    asin_options
)

selected_row = cluster_df[
    cluster_df["asin"] == selected_asin
].iloc[0]

cluster_id = int(selected_row["cluster"])

# Failure-related clusters
failure_clusters = {
    2: "Screen-related",
    4: "Durability / Work-break",
    5: "Charging / Battery"
}

cluster_labels = {
    0: "Phone / Call",
    1: "Case / Cover",
    2: "Screen",
    3: "Mount / Holder",
    4: "Durability",
    5: "Charging / Battery",
    6: "Return / Order",
    7: "Fit / Quality"
}

# ASIN information
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "ASIN",
        selected_asin
    )

with col2:
    st.metric(
        "Cluster",
        f"C{cluster_id}"
    )

with col3:
    st.metric(
        "Profile",
        cluster_labels[cluster_id]
    )

# Failure-related status
if cluster_id in failure_clusters:
    st.warning(
        f"Failure-related profile: "
        f"{failure_clusters[cluster_id]}"
    )
else:
    st.info(
        "This ASIN is not assigned to one of the "
        "three failure-related profiles."
    )


# Topic profile
asin_topic_values = pd.Series(
    {
        topic_names[i]: selected_row[f"topic_{i}_prob"]
        for i in range(8)
    }
).sort_values(ascending=False)

st.subheader("Complaint Topic Profile")

st.bar_chart(asin_topic_values)

# ============================================================
# 8. REVIEW ANALYZER
# ============================================================

st.divider()

st.header("Review Analyzer")

st.write(
    "Enter a negative review to identify its dominant "
    "complaint topic using the trained LDA model."
)

review_text = st.text_area(
    "Review text",
    height=180,
    placeholder="Example: The battery stopped charging after a few weeks..."
)

analyze_button = st.button(
    "Analyze Review",
    type="primary"
)

if analyze_button:

    if not review_text.strip():
        st.warning("Please enter a review.")
    else:

        # Transform review using the trained vocabulary
        review_dtm = vectorizer.transform(
            [review_text]
        )

        # Get LDA topic probabilities
        topic_probabilities = lda_model.transform(
            review_dtm
        )[0]

        # Dominant topic
        dominant_topic_id = topic_probabilities.argmax()
        dominant_probability = topic_probabilities[
            dominant_topic_id
        ]

        dominant_topic_name = topic_names[
            dominant_topic_id
        ]

        # Display dominant topic
        st.subheader("Detected Complaint Topic")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Dominant Topic",
                dominant_topic_name
            )

        with col2:
            st.metric(
                "Topic Probability",
                f"{dominant_probability:.2%}"
            )

        # Failure-related signal
        failure_topic_ids = {0, 1, 5}

        if dominant_topic_id in failure_topic_ids:

            st.warning(
                "This review is associated with a "
                "failure-related complaint topic."
            )

        else:

            st.info(
                "This review is not dominated by one of "
                "the three failure-related complaint topics."
            )

        # Topic probability profile
        st.subheader("Topic Probability Profile")

        review_topic_df = pd.DataFrame({
            "Topic": [
                topic_names[i]
                for i in range(8)
            ],
            "Probability": topic_probabilities
        })

        review_topic_df = review_topic_df.sort_values(
            "Probability",
            ascending=False
        )

        st.bar_chart(
            review_topic_df.set_index("Topic")
        )