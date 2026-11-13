
import streamlit as st
import pandas as pd
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import re
import os

# Page Configuration
st.set_page_config(
    page_title="YouTube Comments Analysis",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 YouTube Comments Analysis")
st.subheader("Big Data Analytics - Sentiment Analysis")
st.write("Automatic YouTube Comments Analysis Dashboard")

# Automatically load CSV
file_path = "comments.csv"

if os.path.exists(file_path):

    df = pd.read_csv(file_path)

    # Identify comment column
    possible = ["comment", "comments", "text", "content"]

    col = next(
        (c for c in df.columns if c.lower() in possible),
        None
    )

    if col is None:
        st.error("CSV must contain a comment column.")
        st.stop()

    df = df.dropna(subset=[col]).copy()

    # Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(text):
        score = analyzer.polarity_scores(str(text))["compound"]

        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    df["Sentiment"] = df[col].apply(analyze_sentiment)

    # Dashboard Metrics
    st.header("📈 Dashboard Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Comments", len(df))
    c2.metric("Positive", (df["Sentiment"] == "Positive").sum())
    c3.metric("Negative", (df["Sentiment"] == "Negative").sum())
    c4.metric("Neutral", (df["Sentiment"] == "Neutral").sum())

    # Pie Chart
    st.header("📊 Sentiment Distribution")

    counts = df["Sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]

    fig = px.pie(
        counts,
        names="Sentiment",
        values="Count",
        title="Positive, Negative and Neutral Comments",
        color="Sentiment",
        color_discrete_map={
            "Positive": "green",
            "Negative": "red",
            "Neutral": "orange"
        }
    )

    st.plotly_chart(fig, width="stretch")

    # Bar Chart
    st.header("📊 Sentiment Count")

    fig2 = px.bar(
        counts,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        title="Sentiment Analysis Results"
    )

    st.plotly_chart(fig2, width="stretch")

    # Most Common Words
    st.header("🔤 Top 10 Common Words")

    words = re.findall(
        r'\b[a-zA-Z]{3,}\b',
        " ".join(df[col].astype(str)).lower()
    )

    common = Counter(words).most_common(10)

    if common:
        word_df = pd.DataFrame(
            common,
            columns=["Word", "Count"]
        )

        fig3 = px.bar(
            word_df,
            x="Word",
            y="Count",
            title="Most Frequently Used Words"
        )

        st.plotly_chart(fig3, width="stretch")

    # Display Comments
    st.header("💬 All Analyzed Comments")

    st.dataframe(df, width="stretch")

    # Download Results
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Analysis Report",
        data=csv,
        file_name="analyzed_comments.csv",
        mime="text/csv"
    )

    st.success("YouTube Comments Analysis Completed Successfully!")

else:
    st.error("comments.csv file not found. Please keep it in the project folder.")
