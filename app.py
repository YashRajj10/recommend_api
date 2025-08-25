from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

WORDPRESS_API = "https://webmind.keytools.in/wp-json/wp/v2/posts?per_page=100"

def fetch_posts():
    response = requests.get(WORDPRESS_API)
    response.raise_for_status()
    posts = response.json()
    # Return list of dicts: id, title, rendered content
    processed = [{
        "id": p["id"],
        "title": p["title"]["rendered"],
        "content": p["content"]["rendered"]
    } for p in posts]
    return processed

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    current_post_id = data.get("post_id")
    posts = fetch_posts()
    curr_post = next((p for p in posts if p["id"] == current_post_id), None)
    if not curr_post:
        return jsonify({"error": "Post not found"}), 404
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([p["content"] for p in posts])
    curr_vector = vectorizer.transform([curr_post["content"]])
    similarities = cosine_similarity(curr_vector, tfidf_matrix).flatten()
    similar_indices = similarities.argsort()[::-1]
    recommendations = []
    for idx in similar_indices:
        if posts[idx]["id"] != current_post_id:
            recommendations.append({"id": posts[idx]["id"], "title": posts[idx]["title"]})
        if len(recommendations) == 3:
            break
    return jsonify({"recommended": recommendations})

