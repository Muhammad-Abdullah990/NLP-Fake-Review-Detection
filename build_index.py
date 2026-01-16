import os
import pickle
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

CSV_FILE = "fake_reviews_dataset.csv"
INDEX_FILE = "reviews.index"
DATA_FILE = "reviews.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"{CSV_FILE} not found in repo root")

    print("Loading dataset...")
    df = pd.read_csv(CSV_FILE)

    # Use first column OR a column named 'review'
    if "review" in df.columns:
        texts = df["review"].astype(str).tolist()
    else:
        texts = df.iloc[:, 0].astype(str).tolist()

    print(f"Loaded {len(texts)} reviews")

    print("Loading SentenceTransformer model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding reviews...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("Saving index & review text...")
    faiss.write_index(index, INDEX_FILE)
    with open(DATA_FILE, "wb") as f:
        pickle.dump(texts, f)

    print("✅ Index build complete!")


if __name__ == "__main__":
    main()
