from flask import Flask, request, jsonify
import openai, faiss, numpy as np
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
index = faiss.read_index("jibin_index.faiss")

def get_context(query):
    q_embed = openai.Embedding.create(input=query, model="text-embedding-ada-002")["data"][0]["embedding"]
    _, I = index.search(np.array([q_embed]).astype("float32"), k=3)
    # For now, simulate context
    return "Jibin is a cybersecurity and ML specialist..."

@app.route("/chat", methods=["POST"])
def chat():
    query = request.json["message"]
    context = get_context(query)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are JibinBot. Funny, smart, and always helpful. Use this context: {context}"},
            {"role": "user", "content": query}
        ]
    )
    return jsonify({"response": response["choices"][0]["message"]["content"]})

if __name__ == "__main__":
    app.run()

