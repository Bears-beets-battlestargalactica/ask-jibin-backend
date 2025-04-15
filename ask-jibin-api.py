from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from flask_cors import CORS


# Load lightweight sentence-transformer model
model = SentenceTransformer("sentence-transformers/paraphrase-albert-small-v2")

# Load FAISS index
index = faiss.read_index("jibin_index.faiss")

# Load documents for metadata (optional improvement)
def load_docs(folder):
    from pathlib import Path
    chunks = []
    for file in Path(folder).glob("*.txt"):
        with open(file, "r") as f:
            text = f.read()
            chunks.append({"filename": file.name, "text": text})
    return chunks

docs = load_docs("knowledge_base/")
texts = [doc["text"] for doc in docs]

# Get context using embedding similarity
def get_context(query):
    q_embed = model.encode([query])[0]
    _, I = index.search(np.array([q_embed]).astype("float32"), k=3)
    return "\n\n".join([texts[i] for i in I[0]])

# Set up Flask app
app = Flask(__name__)
CORS(app,origins="*")

@app.route("/chat", methods=["POST"])
def chat():
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    query = request.json["message"]
    context = get_context(query)

    response = client.chat.completions.create(
        model="mistralai/mixtral-8x7b-instruct",  # Free OpenRouter-supported model
        messages=[
            {"role": "system", "content": f"You are JibinBot. Funny, smart, and always helpful. Use this context: {context}"},
            {"role": "user", "content": query}
        ]
    )

    return jsonify({"response": response.choices[0].message.content})

# For Render deployment (bind to proper port)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
