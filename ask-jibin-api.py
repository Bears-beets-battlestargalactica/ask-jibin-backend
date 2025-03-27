from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import torch

# Load the same Hugging Face model used in embedding
model_name = "thenlper/gte-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load FAISS index
index = faiss.read_index("jibin_index.faiss")

# Fake context for now (you can use metadata file later)
def get_context(query):
    # Embed query
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    q_embed = outputs.last_hidden_state.mean(dim=1)[0].numpy()

    # Search in FAISS
    _, I = index.search(np.array([q_embed]).astype("float32"), k=3)

    # For now, just return static context (can be updated to match doc names)
    return "Jibin is a cybersecurity and ML specialist..."

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    from openai import OpenAI
    import os

    # You can still use OpenRouter for chat
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    query = request.json["message"]
    context = get_context(query)

    response = client.chat.completions.create(
        model="mistralai/mixtral-8x7b-instruct",  # Free & good
        messages=[
            {"role": "system", "content": f"You are JibinBot. Funny, smart, and always helpful. Use this context: {context}"},
            {"role": "user", "content": query}
        ]
    )

    return jsonify({"response": response.choices[0].message.content})

if __name__ == "__main__":
    app.run()
