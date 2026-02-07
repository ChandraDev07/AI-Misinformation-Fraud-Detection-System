import streamlit as st
import joblib
import numpy as np
import textstat
from urllib.parse import urlparse
from newspaper import Article
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup

# ---------- PAGE CONFIG (MOBILE + DESKTOP) ----------
st.set_page_config(
    page_title="AI Misinformation & Fraud Detection",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- MAIN TITLE ----------
st.markdown(
    "<h2 style='text-align:center;'>🛡️ AI Misinformation & Fraud Detection System</h2>",
    unsafe_allow_html=True
)

st.divider()

# ---------- SIDEBAR MENU ----------
menu = st.sidebar.radio(
    "Choose Module",
    ["Fake News", "Claim Verification", "Fake Profile", "Awareness Mode"]
)

# ================= FAKE NEWS =================
if menu == "Fake News":
    st.markdown("<h3 style='text-align:center;'>📰 Fake News Verification</h3>", unsafe_allow_html=True)

    model = joblib.load("fake_news_module/fake_news_model.pkl")
    vectorizer = joblib.load("fake_news_module/vectorizer.pkl")

    def highlight_fake_words(text):
        suspicious_words = ["shocking", "miracle", "secret", "guaranteed", "exposed"]
        for word in suspicious_words:
            text = text.replace(word, f"**:red[{word}]**")
        return text

    text = st.text_area("Paste news text here", height=180)

    if st.button("🔍 Analyze News", use_container_width=True):
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.success("✅ This appears to be REAL news")
        else:
            st.error("🚨 This appears to be FAKE news")
            st.markdown("#### 🔍 Suspicious Words Highlighted")
            st.markdown(highlight_fake_words(text))


# ================= CLAIM VERIFICATION =================
elif menu == "Claim Verification":
    st.markdown("<h3 style='text-align:center;'>🔎 Claim Verification</h3>", unsafe_allow_html=True)

    claim = st.text_input("Enter a claim to verify")

    if st.button("🌐 Search Evidence", use_container_width=True):
        st.markdown("#### 🔗 Related Sources")

        try:
            found = False
            with DDGS() as ddgs:
                results = ddgs.news(claim, max_results=5)

                for r in results:
                    st.markdown(f"- [{r['title']}]({r['url']})")
                    found = True

                if not found:
                    results = ddgs.text(claim, max_results=5)
                    for r in results:
                        st.markdown(f"- [{r['title']}]({r['href']})")
                        found = True

            if not found:
                st.warning("No results found. Try rephrasing the claim.")

        except Exception:
            st.error("⚠️ Search temporarily blocked. Please try again later.")


# ================= FAKE PROFILE =================
elif menu == "Fake Profile":
    st.markdown("<h3 style='text-align:center;'>👤 Fake Profile Detection</h3>", unsafe_allow_html=True)

    model = joblib.load("profile_module/profile_model.pkl")

    followers = st.number_input("Followers", min_value=0)
    following = st.number_input("Following", min_value=0)
    posts = st.number_input("Posts", min_value=0)
    bio_len = st.number_input("Bio Description Length", min_value=0)

    if st.button("🕵️ Check Profile", use_container_width=True):
        data = np.array([[followers, following, posts, bio_len]])
        pred = model.predict(data)[0]
        proba = model.predict_proba(data)[0][1]

        st.warning("🚨 Fake/Bot Account" if pred == 1 else "✅ Genuine Account")
        st.info(f"Fake Probability: {proba*100:.2f}%")


# ================= AWARENESS MODE =================
elif menu == "Awareness Mode":
    st.markdown("<h3 style='text-align:center;'>🧠 Awareness Mode</h3>", unsafe_allow_html=True)

    st.info("📰 **Fake News Signs**")
    st.write("- Sensational headlines\n- No trusted sources\n- Emotional manipulation")

    st.info("👤 **Fake Profile Signs**")
    st.write("- Few posts, many followers\n- Random usernames\n- No real photos")

    st.warning("🚨 **Fake Claim Signs**")
    st.write(
        "- No reliable source\n"
        "- Emotional or sensational language\n"
        "- Urgency to share"
    )

    st.success("✅ **Tip:** Think before you believe. Verify before you share.")
