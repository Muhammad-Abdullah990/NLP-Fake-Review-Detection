import streamlit as st
import numpy as np
import faiss
import pickle
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

import os
import subprocess

# Check if index exists, if not, run the build script
if not os.path.exists("reviews.index") or not os.path.exists("reviews.pkl"):
    st.info("Building search index for the first time... this may take a moment.")
    subprocess.run(["python", "build_index.py"])
    st.success("Index built successfully!")

# -----------------------------
# Load FAISS + reviews
# -----------------------------
import os
import faiss
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_FILE = "reviews.index"
DATA_FILE = "reviews.pkl"
CSV_FILE = "Dataset.csv"

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

model = load_model()

def build_index():
    st.info("Building search index for the first time... this may take a moment.")

    df = pd.read_csv(CSV_FILE)
    texts = df["review"].astype(str).tolist()

    embeddings = model.encode(texts, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(DATA_FILE, "wb") as f:
        pickle.dump(texts, f)

    st.success("Index built successfully!")
    return index, texts


if not os.path.exists(INDEX_FILE) or not os.path.exists(DATA_FILE):
    index, reviews = build_index()
else:
    index = faiss.read_index(INDEX_FILE)
    with open(DATA_FILE, "rb") as f:
        reviews = pickle.load(f)

# -----------------------------
# FAISS embeddings function
# -----------------------------
from transformers import AutoTokenizer, AutoModel
def embed_text(text_list):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

def retrieve(query, k=5):
    q_emb = embed_text([query])
    D, I = index.search(np.array(q_emb), k)
    return [reviews[i] for i in I[0]]

# -----------------------------
# Load DistilBERT sentiment classifier (binary)
# -----------------------------
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1  # CPU
)

# -----------------------------
# Fake review detection
# -----------------------------
def detect(review):
    docs = retrieve(review)
    context = "\n".join(docs)

    # Combine review + context
    input_text = f"{review}\nContext:\n{context}"

    result = classifier(input_text)[0]
    label_raw = result["label"]
    score = result["score"]

    # Map sentiment to fake review categories
    if label_raw == "NEGATIVE":
        mapped_label = "Fake"
    else:  # POSITIVE
        mapped_label = "Genuine"

    return mapped_label, score, docs

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🕵️ Fake Review Detection")
st.title("🕵️ Fake Review Detection (RAG + DistilBERT)")

text = st.text_area("Enter a review to analyze", height=150)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter a review first!")
    else:
        with st.spinner("Analyzing..."):
            label, score, docs = detect(text)

        # Color-coded labels
        color_map = {
            "Very Genuine": "#28a745",
            "Genuine": "#7bc67e",
            "Moderate": "#ffc107",
            "Fake": "#fd7e14",
            "Very Fake": "#dc3545"
        }

        st.subheader("Result")
        st.markdown(
            f"<h3 style='color:{color_map.get(label,'black')}'>{label} ({score*100:.1f}%)</h3>",
            unsafe_allow_html=True
        )

        st.subheader("Retrieved Similar Reviews (RAG Context)")
        for r in docs:
            st.write("-", r)


