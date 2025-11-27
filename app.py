import streamlit as st
import pandas as pd
import plotly.express as px
import re
import time
import joblib

st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
)

# -------------------- LOAD MODEL --------------------
try:
    model_data = joblib.load("sentiment_model_3class.pkl")
    vectorizer = model_data["vectorizer"]
    clf = model_data["model"]
    st.success(" Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'sentiment_model_3class.pkl' exists.")
    st.stop()
except Exception as e:
    st.error(f" Error loading model: {e}")
    st.stop()

# -------------------- SESSION STATE --------------------
if "tweet_input" not in st.session_state:
    st.session_state.tweet_input = ""

# -------------------- STYLING --------------------
st.markdown("""
    <style>
        .main {
            background-color: #0f0f1a;
            color: white;
        }
        .stTextArea textarea {
            background-color: #1e1e2e;
            color: white;
            border: 1px solid #7b7bff;
        }
        .block-container {
            padding-top: 1rem;
        }
        .gradient-box {
            background: linear-gradient(to right, #7928CA, #FF0080);
            padding: 10px 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            font-weight: 600;
        }
        .tweet-box {
            background: linear-gradient(to right, #1e1e2e, #1c1c3a);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            color: white;
            border: 1px solid #7b7bff;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("<div class='gradient-box'><h2>💬 Twitter Sentiment Analyzer </h2></div>", unsafe_allow_html=True)

# -------------------- INPUT SECTION --------------------
st.subheader("Enter Tweets")

tweet_input_area = st.text_area(
    "Enter one or multiple tweets (each on a new line):",
    height=150,
    key="tweet_input"
)

# --- CLEAR BUTTON ---
def clear_text():
    st.session_state.tweet_input = ""

col1, col_spacer, col2 = st.columns([1, 4, 1])
with col1:
    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)
with col2:
    clear_btn = st.button("🧹 Clear Input", use_container_width=True, on_click=clear_text)

# -------------------- CLEANING FUNCTION --------------------
def clean_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+|www\S+", "", tweet)
    tweet = re.sub(r"@\w+|#", "", tweet)
    tweet = re.sub(r"[^a-z\s]", "", tweet)
    tweet = re.sub(r"\s+", " ", tweet).strip()
    return tweet

# -------------------- SENTIMENT MAPPING --------------------
sentiment_labels = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

color_map = {
    "Positive": "limegreen",
    "Negative": "red",
    "Neutral": "gray"
}

explanations = {
    "Positive": "This tweet expresses positivity — happiness, appreciation, or optimism.",
    "Negative": "This tweet conveys negativity — anger, sadness, or frustration.",
    "Neutral": "This tweet is neutral — factual or emotionless in tone."
}

# -------------------- ANALYSIS --------------------
if analyze_btn:
    if st.session_state.tweet_input.strip() == "":
        st.warning(" Please enter at least one tweet to analyze.")
    else:
        tweets = st.session_state.tweet_input.strip().split("\n")
        data = []
        progress_text = "Analyzing tweets... Please wait."
        progress_bar = st.progress(0, text=progress_text)

        for i, tweet in enumerate(tweets):
            cleaned = clean_tweet(tweet)
            if cleaned:
                X_new = vectorizer.transform([cleaned])
                probs = clf.predict_proba(X_new)[0]
                pred = clf.predict(X_new)[0]
                confidence = round(max(probs) * 100, 2)
                sentiment = sentiment_labels.get(pred, "Unknown")
                data.append({
                    "Tweet": tweet,
                    "Sentiment": sentiment,
                    "Confidence": confidence
                })
            progress_bar.progress((i + 1) / len(tweets), text=progress_text)
            time.sleep(0.05)

        progress_bar.empty()

        if not data:
            st.warning("⚠️ No valid tweets found after cleaning.")
        else:
            df = pd.DataFrame(data)

            # -------------------- RESULTS SUMMARY --------------------
            st.markdown("<div class='gradient-box'><h3> Analysis Results</h3></div>", unsafe_allow_html=True)
            total = len(df)
            pos = df[df["Sentiment"] == "Positive"].shape[0]
            neg = df[df["Sentiment"] == "Negative"].shape[0]
            neu = df[df["Sentiment"] == "Neutral"].shape[0]

            col1, col2, col3 = st.columns(3)
            col1.metric("😊 Positive", f"{pos} Tweets", f"{(pos/total)*100:.1f}%")
            col2.metric("😐 Neutral", f"{neu} Tweets", f"{(neu/total)*100:.1f}%")
            col3.metric("😡 Negative", f"{neg} Tweets", f"{(neg/total)*100:.1f}%")

            # -------------------- CHARTS --------------------
            st.markdown("<div class='gradient-box'><h3> Sentiment Charts</h3></div>", unsafe_allow_html=True)
            col_chart1, col_chart2 = st.columns(2)

            sentiment_counts = df["Sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]

            hover_texts = df.groupby("Sentiment")["Tweet"].apply(
                lambda x: "<br><br>".join(x.str.slice(0, 70) + '...')
            ).reset_index()
            hover_texts.columns = ["Sentiment", "Example Tweets"]
            chart_data = pd.merge(sentiment_counts, hover_texts, on="Sentiment")

            # Donut chart
            donut_fig = px.pie(
                chart_data,
                names="Sentiment",
                values="Count",
                hole=0.5,
                color="Sentiment",
                color_discrete_map=color_map,
                title="Sentiment Distribution (Hover to View Example Tweets)",
                hover_data=["Example Tweets"]
            )
            donut_fig.update_traces(
                textinfo="label+percent",
                hovertemplate="<b>%{label} (%{percent})</b><br>%{customdata[0]}<extra></extra>"
            )
            col_chart1.plotly_chart(donut_fig, use_container_width=True)

            # Confidence bar chart
            conf_avg = df.groupby("Sentiment")["Confidence"].mean().reset_index()
            conf_fig = px.bar(
                conf_avg,
                x="Sentiment",
                y="Confidence",
                color="Sentiment",
                text=conf_avg["Confidence"].apply(lambda x: f"{x:.1f}%"),
                color_discrete_map=color_map,
                title="Average Confidence per Sentiment (%)"
            )
            conf_fig.update_traces(textposition="outside")
            conf_fig.update_layout(yaxis_range=[0, 105])
            col_chart2.plotly_chart(conf_fig, use_container_width=True)

            # -------------------- DETAILED RESULTS --------------------
            st.markdown("<div class='gradient-box'><h3> Detailed Tweet Analysis</h3></div>", unsafe_allow_html=True)
            st.markdown("---")

            for i, row in df.iterrows():
                insight = explanations.get(row["Sentiment"], "No insight available.")
                st.markdown(
                    f"""
                    <div class='tweet-box'>
                        <b>Tweet {i+1}:</b> {row['Tweet']}<br>
                        <b>Sentiment:</b> {row['Sentiment']} | 
                        <b>Confidence:</b> {row['Confidence']}%<br>
                        {insight}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # -------------------- EXPORT RESULTS --------------------
            st.markdown("<div class='gradient-box'><h3>📤 Export Results</h3></div>", unsafe_allow_html=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Sentiment Results as CSV",
                data=csv,
                file_name="sentiment_results.csv",
                mime="text/csv"
            )

            st.success("✅ Analysis Complete!")

# -------------------- FOOTER --------------------
st.markdown("""
<hr style="border:1px solid #444;">
<div style='text-align: center; color: #aaa; font-size: 14px; padding: 20px 0;'>
    <p style="font-size: 16px; margin-bottom: 10px;">
        <b>💬 Twitter Sentiment Analyzer — (AI, ML & NLP)</b>
    </p>
    <p style="margin-bottom: 15px;">Built with 🐍 <b>Python</b> | ⚙️ <b>Streamlit</b> | 🤖 <b>Scikit-learn</b></p>
📧 <a href="mailto:albinmathew452@gmail.com" style="color:#1DA1F2;">Contact Me</a> | 
🌐 <a href="https://github.com/albinmathew90" style="color:#1DA1F2;">GitHub</a> 
</div>
""", unsafe_allow_html=True)
