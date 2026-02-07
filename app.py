import streamlit as st
import joblib
import numpy as np
import textstat
from urllib.parse import urlparse
from newspaper import Article
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup


st.title("🛡 AI Misinformation & Fraud Detection System")

menu = st.sidebar.selectbox("Choose Module", [
    "Fake News",
    "Claim Verification",
    "Fake Profile",
    "Awareness Mode"
])

# ================= FAKE NEWS =================
if menu == "Fake News":
    model = joblib.load("fake_news_module/fake_news_model.pkl")
    vectorizer = joblib.load("fake_news_module/vectorizer.pkl")

    def highlight_fake_words(text):
        suspicious_words = ["shocking", "miracle", "secret", "guaranteed", "exposed"]
        for word in suspicious_words:
            text = text.replace(word, f"**:red[{word}]**")
        return text

    text = st.text_area("Enter News Text")

    if st.button("Analyze"):
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.success("Real News")
        else:
            st.error("Fake News")
            st.markdown("### 🔍 Suspicious Words Highlighted")
            st.markdown(highlight_fake_words(text))


# ================= CLAIM VERIFICATION =================
elif menu == "Claim Verification":
    from duckduckgo_search import DDGS
    st.title("🔎 Claim Verification Assistant")

    claim = st.text_input("Enter a claim")

    if st.button("Search Evidence"):
        st.write("### 🔗 Related Sources")

        try:
            found = False
            with DDGS() as ddgs:
                results = ddgs.news(claim, max_results=5)

                for r in results:
                    st.write(f"[{r['title']}]({r['url']})")
                    found = True

                if not found:
                    results = ddgs.text(claim, max_results=5)
                    for r in results:
                        st.write(f"[{r['title']}]({r['href']})")
                        found = True

            if not found:
                st.warning("No results found. Try rephrasing the claim.")

        except Exception:
            st.error("⚠ Search temporarily blocked (rate limit). Please try again in a few seconds.")


# ================= FAKE PROFILE =================
elif menu == "Fake Profile":
    model = joblib.load("profile_module/profile_model.pkl")

    followers = st.number_input("Followers")
    following = st.number_input("Following")
    posts = st.number_input("Posts")
    bio_len = st.number_input("Bio Description Length")

    if st.button("Check Profile"):
        data = np.array([[followers, following, posts, bio_len]])
        pred = model.predict(data)[0]
        proba = model.predict_proba(data)[0][1]

        st.warning("Fake/Bot Account" if pred == 1 else "Genuine Account")
        st.write(f"Fake Probability: {proba*100:.2f}%")


# ================= AWARENESS MODE =================
elif menu == "Awareness Mode":
    st.title("🧠 Awareness Mode – Learn to Spot Fake Content")

    st.write("""
    ### 📰 Fake News Signs
    - Sensational headlines  
    - No trusted sources  
    - Emotional manipulation  

    ### 👤 Fake Profile Signs
    - Few posts, many followers  
    - Random username  
    - No real photos  

    ### 🤖 AI Text Signs
    - Very perfect grammar  
    - Repetitive sentence structure  
    - Overly neutral tone  
    """)
