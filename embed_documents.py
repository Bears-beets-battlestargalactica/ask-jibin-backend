from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss
from pathlib import Path

# Load a local Hugging Face model
model_name = "thenlper/gte-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()

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
