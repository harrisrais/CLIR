# Multilingual Search Engine

A Flask-based multilingual search engine that translates user queries, ranks results based on their similarity to the query, and reranks them using contextual relevance. The application supports real-time query translation and leverages deep learning embeddings for semantic search.

---

## Features
- **Multilingual Query Translation:** Automatically translates user queries into English using `deep-translator`.
- **Semantic Search:** Uses the `sentence-transformers` library to compute embeddings and perform semantic similarity search.
- **Weighted Search:** Combines title and content similarities for accurate results.
- **Contextual Reranking:** Reranks search results based on contextual relevance.
- **Responsive Web Interface:** Built using Flask, the app provides a user-friendly search experience.

---

## Requirements

### Programming Language
- Python 3.9 or higher

### Libraries and Tools
- `Flask`: To build and run the web application.
- `pandas`: For data manipulation and preprocessing.
- `deep-translator`: To translate multilingual queries to English.
- `sentence-transformers`: For generating embeddings for both queries and documents.
- `scikit-learn`: For calculating cosine similarity between embeddings.

### Dataset
- A CSV file (`allfootball.csv`) containing the following fields:
  - `title`: Title of the article
  - `desc`: Description of the article
  - `url`: URL to the article

---

## Installation

### Step 1: Clone the Repository
```bash
$ git clone https://github.com/your-username/your-repo-name.git
$ cd your-repo-name
```

### Step 2: Install Dependencies
Use `pip` to install the required libraries.
```bash
$ pip install -r requirements.txt
```

### Step 3: Prepare Dataset
Ensure the `allfootball.csv` file is present in the root directory and formatted as described in the **Dataset** section.

---

## Usage

### Running the Application
Start the Flask server with the following command:
```bash
$ python app.py
```

The application will run on `http://127.0.0.1:5000/` by default.

### How to Use
1. Enter your query in any language on the search bar.
2. Submit the query to get the top-ranked results based on semantic similarity.
3. View the results along with their descriptions and URLs.

---

## Application Architecture

1. **Query Translation:**
   - Translates the input query into English using `deep-translator`.

2. **Query Embedding:**
   - Generates a semantic embedding for the translated query using `sentence-transformers`.

3. **Initial Search:**
   - Calculates the similarity of the query embedding with precomputed embeddings of titles and content.
   - Combines title and content similarities to generate initial results.

4. **Reranking:**
   - Contextually reranks the top results based on similarity to the query's context.

5. **Results Display:**
   - Outputs the ranked results with their titles, descriptions, and URLs in a user-friendly interface.

---

## File Structure
```plaintext
.
├── app.py                # Main application script
├── templates/
│   ├── index.html        # Home page template
│   └── results.html      # Search results template
├── static/               # Static files (CSS, JavaScript, etc.)
├── allfootball.csv       # Dataset
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Key Functions

1. `translate_query(query)`
   - Translates the input query into English.

2. `embed_query(query)`
   - Generates an embedding vector for the query.

3. `search(query_embedding, title_embeddings, content_embeddings, top_k=5, title_weight=1)`
   - Performs initial search by combining title and content similarities.

4. `rerank_results(query, top_k_results, model)`
   - Reranks results using contextual similarity.

---

## Future Improvements
- **Support for Additional Languages:** Extend translation capabilities for more languages.
- **Real-Time Dataset Updates:** Allow dynamic addition of new articles.
- **Pagination:** Implement pagination for displaying large result sets.
- **Performance Optimization:** Leverage GPUs for faster embedding generation.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contribution
Feel free to fork the repository and submit pull requests. Contributions are always welcome!

---

## Contact
For any questions or suggestions, contact [your email/contact information].
