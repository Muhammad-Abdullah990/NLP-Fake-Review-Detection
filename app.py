import os
import pickle
import numpy as np
import streamlit as st
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
CSV_FILE = "fake_reviews_dataset.csv"
INDEX_FILE = "reviews.index"
DATA_FILE = "reviews.pkl"

st.set_page_config(page_title="🕵️ Fake Review Detection", layout="centered")
st.title("🕵️ Fake Review Detection (RAG + NLP)")

# -----------------------------
# Load embedding model (cached)
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)

# -----------------------------
# Build index if missing
# -----------------------------
def build_index():
    st.info("Building search index for the first time... this may take a moment.")

    if not os.path.exists(CSV_FILE):
        st.error(f"❌ {CSV_FILE} not found in repo root")
        st.stop()

    df = pd.read_csv(CSV_FILE)

    if "review" in df.columns:
        texts = df["review"].astype(str).tolist()
    else:
        texts = df.iloc[:, 0].astype(str).tolist()

    model = load_embedding_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(DATA_FILE, "wb") as f:
        pickle.dump(texts, f)

    st.success("Index built successfully!")
    return index, texts


# -----------------------------
# Load index & reviews
# -----------------------------
if not os.path.exists(INDEX_FILE) or not os.path.exists(DATA_FILE):
    index, reviews = build_index()
else:
    index = faiss.read_index(INDEX_FILE)
    with open(DATA_FILE, "rb") as f:
        reviews = pickle.load(f)

# -----------------------------
# Retrieval
# -----------------------------
def retrieve(query, k=5):
    model = load_embedding_model()
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    return [reviews[i] for i in I[0]]

# -----------------------------
# Sentiment classifier (CPU)
# -----------------------------
@st.cache_resource(show_spinner="Loading classifier...")
def load_classifier():
    return pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )

classifier = load_classifier()

# -----------------------------
# Fake review detection logic
# -----------------------------
def detect(review):
    docs = retrieve(review)
    context = "\n".join(docs)

    input_text = f"{review}\n\nContext:\n{context}"
    result = classifier(input_text)[0]

    label = "Fake" if result["label"] == "NEGATIVE" else "Genuine"
    score = result["score"]

    return label, score, docs

# -----------------------------
# UI
# -----------------------------
text = st.text_area("Enter a review to analyze", height=160)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing review..."):
            label, score, docs = detect(text)

        st.subheader("Result")
        st.markdown(
            f"### **{label}**  \nConfidence: **{score*100:.2f}%**"
        )

        st.subheader("Retrieved Similar Reviews")
        for r in docs:
            st.write("•", r)
