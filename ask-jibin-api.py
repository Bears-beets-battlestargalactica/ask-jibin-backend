from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime
import faiss
import numpy as np
import pickle

# Load FAISS index (this is lightweight) new
index = faiss.read_index("jibin_index.faiss")

# Load mapping between index positions and actual context strings
with open("jibin_docs.pkl", "rb") as f:
    docs = pickle.load(f)  # list of strings

# Embed placeholder - assume embeddings were created with GTE
def get_context(query):
    return "Jibin specializes in cybersecurity and machine learning."  # temporary fallback
    # Note: replace with embedding if needed externally

# Flask setup
app = Flask(__name__)
CORS(app, origins=["https://bears-beets-battlestargalactica.github.io"])

@app.route("/chat", methods=["POST"])
def chat():
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    query = request.json["message"]
    print(f"[{datetime.now()}] 📥 Received prompt: {query}")

    context = get_context(query)

    try:
        response = client.chat.completions.create(
            model="openrouter/openchat-3.5-0106",
            messages=[
                {"role": "system", "content": f"You are JibinBot. Use this context: {context}"},
                {"role": "user", "content": query}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return jsonify({"response": response.choices[0].message.content})

    except Exception as e:
        print(f"[{datetime.now()}] ❌ Error: {e}")
        return jsonify({"response": "An error occurred, please try again."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
