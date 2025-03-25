# embed_documents.py
import openai, faiss
import os
from pathlib import Path

openai.api_key = os.getenv("OPENAI_API_KEY")

def load_docs(folder):
    chunks = []
    for file in Path(folder).glob("*.txt"):
        with open(file, "r") as f:
            text = f.read()
            chunks.append({"filename": file.name, "text": text})
    return chunks

docs = load_docs("knowledge_base/")
texts = [doc["text"] for doc in docs]

# Get embeddings
embeddings = [openai.Embedding.create(input=t, model="text-embedding-ada-002")["data"][0]["embedding"] for t in texts]

# Build FAISS index
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

# Save index and metadata
faiss.write_index(index, "jibin_index.faiss")
with open("metadata.txt", "w") as f:
    for doc in docs:
        f.write(f"{doc['filename']}\n")
