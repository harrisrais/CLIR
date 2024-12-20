from flask import Flask, request, render_template
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Initialize Flask app
app = Flask(__name__)

# Load dataset
data = pd.read_csv('allfootball.csv', encoding='ISO-8859-9')
data['content'] = data['title'] + " " + data['desc']

# Load multilingual sentence embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Precompute embeddings for both titles and content
title_embeddings = model.encode(data['title'].tolist(), show_progress_bar=True)
content_embeddings = model.encode(
    data['content'].tolist(), show_progress_bar=True)


# Initialize translator using deep-translator
def translate_query(query):
    """Translate the input query into English using deep-translator."""
    try:
        return GoogleTranslator(source='auto', target='en').translate(query)
    except Exception as e:
        print(f"Translation failed: {e}")
        return query


# Function to embed query
def embed_query(query):
    """Generate an embedding vector for the query."""
    return model.encode([query])[0]


# Function to search for results
def search(query_embedding, title_embeddings,
           content_embeddings, top_k=5, title_weight=1):
    title_similarity = cosine_similarity(
        [query_embedding], title_embeddings)[0]
    content_similarity = cosine_similarity(
        [query_embedding], content_embeddings)[0]

    # Compute weighted similarity
    combined_similarity = title_weight * title_similarity + \
        (1 - title_weight) * content_similarity

    # Get top-k indices
    indices = combined_similarity.argsort()[-top_k:][::-1]
    results = data.iloc[indices]
    return results, combined_similarity[indices]


# Function to rerank results based on contextual similarity
def rerank_results(query, top_k_results, model):
    query_embedding = model.encode([query])[0]
    result_embeddings = model.encode(
        top_k_results['content'].tolist(), show_progress_bar=False)

    # Compute similarity between query and result contents
    contextual_similarities = cosine_similarity(
        [query_embedding], result_embeddings)[0]

    # Add contextual similarity scores for reranking
    top_k_results['contextual_similarity'] = contextual_similarities

    # Re-rank results by contextual similarity
    reranked_results = top_k_results.sort_values(
        by='contextual_similarity', ascending=False)
    return reranked_results


# Routes
@app.route('/')
def index():
    """Render the main search page."""
    return render_template('index.html')


@app.route('/', methods=['POST'])
def search_articles():
    query = request.form['query']
    translated_query = translate_query(query)  # Translate the query
    query_embedding = embed_query(translated_query)  # Embed the query

    # Initial search
    initial_results, scores = search(
        query_embedding, title_embeddings, content_embeddings)

    # Prepare the top-k results for reranking
    top_k_results = initial_results.copy()
    top_k_results['score'] = scores

    # Re-rank results
    reranked_results = rerank_results(query, top_k_results, model)

    # Pass final results to the template
    response = reranked_results[['title', 'desc', 'url']].to_dict(
        orient='records')
    return render_template(
        'results.html', query=query,
        translated_query=translated_query, results=response)


if __name__ == '__main__':
    app.run(debug=True)
