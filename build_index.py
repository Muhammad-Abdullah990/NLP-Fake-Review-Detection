import pandas as pd
import numpy as np
import faiss
import pickle
from transformers import AutoTokenizer, AutoModel
import torch

# -----------------------------
# Load dataset
# -----------------------------
print("Loading dataset...")
df = pd.read_csv("fake_reviews_dataset.csv")  # Ensure your reviews are in first column
reviews = df.iloc[:, 0].astype(str).tolist()

# -----------------------------
# Load HuggingFace MiniLM model for embeddings
# -----------------------------
print("Loading embedding model...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Function to compute embeddings
# -----------------------------
def embed_text(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling of last hidden states
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# -----------------------------
# Compute embeddings
# -----------------------------
print("Encoding reviews...")
embeddings = embed_text(reviews)

# -----------------------------
# Build FAISS index
# -----------------------------
print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# -----------------------------
# Save index and review list
# -----------------------------
print("Saving index and reviews...")
faiss.write_index(index, "reviews.index")
with open("reviews.pkl", "wb") as f:
    pickle.dump(reviews, f)

print("DONE! Your FAISS index and reviews are ready.")
