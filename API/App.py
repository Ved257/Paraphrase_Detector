import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from http.client import HTTPException


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/plagarism_api', methods=['POST'])
def plagarism_api():
    
    data = pd.read_csv('C:/Users/91897/OneDrive/Desktop/Dev/python-codes/NLP/train.csv')
    pd.set_option('display.max_colwidth',-1)
    data.head()
    
    def remove_stopwords_from_sentences(sentence_list):
        stop_words = set(stopwords.words('english'))
        filtered_sentences = []
        for sentence in sentence_list:
            word_tokens = word_tokenize(sentence)
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            filtered_sentences.append(' '.join(filtered_sentence))
        return filtered_sentences
    train = remove_stopwords_from_sentences(data.Sentence1)
    

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(train)

    input = request.get_json(force=True)
    print(list(input.values()))
    inputs = (list(input.values()))
    doc1 = inputs[0]
    doc2 = inputs[1]

    tfidf_doc1 = tfidf_vectorizer.transform([doc1])
    tfidf_doc2 = tfidf_vectorizer.transform([doc2])

    similarity = cosine_similarity(tfidf_doc1, tfidf_doc2)
    output = similarity[0][0]
    return jsonify(output)
    
    # Output the similarity
    print(f"Similarity between '{doc1}' and '{doc2}': {similarity[0][0]}")
    
    
    
@app.errorhandler(HTTPException)
def handle_exception(e):
    return jsonify({"message": e.description}), e.code


if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=8000)
