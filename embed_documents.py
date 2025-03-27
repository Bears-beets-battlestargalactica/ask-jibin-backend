from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import faiss
from pathlib import Path

# Load a lightweight embedding model
model = SentenceTransformer("sentence-transformers/paraphrase-albert-small-v2")

# Embed function
def embed(text):
    return model.encode([text])[0]

# Load documents
def load_docs(folder):
    chunks = []
    for file in Path(folder).glob("*.txt"):
        with open(file, "r") as f:
            text = f.read()
            chunks.append({"filename": file.name, "text": text})
    return chunks

docs = load_docs("knowledge_base/")
texts = [doc["text"] for doc in docs]

# Generate embeddings
embeddings = [embed(t) for t in texts]

# Save FAISS index
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))
faiss.write_index(index, "jibin_index.faiss")
