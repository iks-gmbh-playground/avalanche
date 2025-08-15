# import packages
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import plotly.express as px


# Helper function to get dataset path
def get_dataset_path():
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the CSV file
    csv_path = os.path.join(current_dir, "data", "customer_reviews.csv")
    return csv_path


# Text cleaning helper
def clean_text(text: str) -> str:
    """Clean text by removing punctuation, converting to lowercase, and stripping whitespaces."""
    if text is None:
        return ""
    # Lowercase and strip surrounding whitespace
    s = str(text).lower().strip()
    # Remove punctuation using regex: keep word characters and whitespace only
    s = re.sub(r"[^\w\s]", "", s)
    return s


st.title("Generative AI")
st.subheader("A prototyping app with help from Coursera")
st.write("This is your GenAI-powered data processing app.")

col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Ingest Dataset"):
        try:
            path = get_dataset_path()
            st.session_state["df"] = pd.read_csv(path)
            st.success("Dataset loaded successfully!")
        except Exception as e:
            st.error(f"Request failed: {e}")

with col2:
    if st.button("🧹 Parse Reviews"):
        if "df" in st.session_state:
            st.session_state["df"]["CLEANED_SUMMARY"] = st.session_state["df"]["SUMMARY"].apply(clean_text)
            st.success("Reviews parsed and cleaned successfully!")
        else:
            st.error("No dataset loaded. Please load a dataset first.")

if "df" in st.session_state:
    st.subheader("Filter by Product")
    selected_product = st.selectbox("Select a product", ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))
    if selected_product != "All Products":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == selected_product]
    else:
        filtered_df = st.session_state["df"]

    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head())

    st.subheader("Sentiment score by product")
    grouped = st.session_state["df"].groupby("PRODUCT")["SENTIMENT_SCORE"].mean()
    st.bar_chart(grouped)

    # Scatterplot: Sentiment Score vs. Review Length
    st.subheader("Sentiment vs. Review Length (scatter)")
    # Determine text column for length calculation
    text_col = None
    if "CLEANED_SUMMARY" in filtered_df.columns:
        text_col = "CLEANED_SUMMARY"
    elif "SUMMARY" in filtered_df.columns:
        text_col = "SUMMARY"

    if text_col is not None and "SENTIMENT_SCORE" in filtered_df.columns:
        plot_df = filtered_df.copy()
        plot_df["REVIEW_LENGTH"] = plot_df[text_col].astype(str).str.len()
        fig, ax = plt.subplots()
        ax.scatter(plot_df["REVIEW_LENGTH"], plot_df["SENTIMENT_SCORE"], alpha=0.5)
        ax.set_xlabel("Review Length (characters)")
        ax.set_ylabel("Sentiment Score")
        ax.set_title("Sentiment vs. Review Length")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Plotly scatter (interactive)
        st.subheader("Sentiment vs. Review Length (Plotly)")
        fig_px = px.scatter(
            plot_df,
            x="REVIEW_LENGTH",
            y="SENTIMENT_SCORE",
            color="PRODUCT" if "PRODUCT" in plot_df.columns else None,
            hover_data=[text_col, "PRODUCT"] if "PRODUCT" in plot_df.columns else [text_col],
            title="Sentiment vs. Review Length",
        )
        fig_px.update_layout(xaxis_title="Review Length (characters)", yaxis_title="Sentiment Score")
        st.plotly_chart(fig_px, use_container_width=True)
    else:
        st.info("Columns needed for scatterplot not found.")
