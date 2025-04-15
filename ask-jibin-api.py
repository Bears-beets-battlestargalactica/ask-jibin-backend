from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import torch
import os
from datetime import datetime

# Load model and tokenizer for embedding
model_name = "thenlper/gte-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load FAISS index
index = faiss.read_index("jibin_index.faiss")

# Load context from query
def get_context(query):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    q_embed = outputs.last_hidden_state.mean(dim=1)[0].numpy()
    _, I = index.search(np.array([q_embed]).astype("float32"), k=3)
    return "Jibin is a cybersecurity and ML specialist..."  # Optional: update with retrieved file info

# Flask setup
app = Flask(__name__)

# ✅ Secure CORS config for GitHub Pages
CORS(app, origins=["https://bears-beets-battlestargalactica.github.io"])

@app.route("/chat", methods=["POST"])
def chat():
    from openai import OpenAI

    # Set up OpenRouter API
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    # Read user message
    query = request.json["message"]
    print(f"[{datetime.now()}] 📥 Prompt received: {query}")

    context = get_context(query)

    try:
        # Send prompt to OpenRouter
        response = client.chat.completions.create(
            model="openrouter/openchat-3.5-0106",  #  Faster model
            messages=[
                {"role": "system", "content": f"You are JibinBot. Helpful, witty, and informative. Use this context: {context}"},
                {"role": "user", "content": query}
            ],
            max_tokens=200,  #  Prevents overlong replies
            temperature=0.7   #  Slight creativity
        )
        return jsonify({"response": response.choices[0].message.content})
    
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Error: {e}")
        return jsonify({"response": "Sorry, something went wrong. Please try again later."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
