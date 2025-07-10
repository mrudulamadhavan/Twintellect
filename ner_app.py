# ner_extended_app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForTokenClassification
import pickle
import spacy
import os

# Optional: Twitter API
import requests
import re

st.set_page_config(page_title="Tweet NER App", layout="wide")
st.title("🔍 Named Entity Recognition on Tweets (Multi-Domain)")

# Load tokenizer and model
@st.cache_resource
def load_bert_model():
    model = TFAutoModelForTokenClassification.from_pretrained("output/NER_pretrained")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

@st.cache_resource
def load_lstm_crf_model():
    from tensorflow_addons.layers import CRF
    from keras.models import load_model
    model = build_lstm_crf_model()  # Define this like in your training script
    model.load_weights("twitter_ner_crf.h5")
    return model

@st.cache_resource
def load_crf_tokenizer():
    with open("crf_tokenizer.pkl", "rb") as f:
        return pickle.load(f)

# ID to label map
tag2id = {
    'O': 0, 'B-company': 1, 'B-facility': 2, 'B-geo-location': 3, 'B-movie': 4, 'B-musicartist': 5,
    'B-other': 6, 'B-person': 7, 'B-product': 8, 'B-sportsteam': 9, 'B-tvshow': 10
}
id2tag = {v: k for k, v in tag2id.items()}

# Prediction logic
def get_bert_predictions(text):
    model, tokenizer = load_bert_model()
    tokens = tokenizer(text, return_tensors="tf", truncation=True)
    logits = model(**tokens).logits
    preds = tf.argmax(logits, axis=-1).numpy()[0]
    tokens_decoded = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    return [(token, id2tag.get(pred, "O")) for token, pred in zip(tokens_decoded, preds) if token not in ["[CLS]", "[SEP]"]]

def get_lstm_crf_predictions(text):
    model = load_lstm_crf_model()
    tokenizer = load_crf_tokenizer()
    seq = tokenizer.texts_to_sequences([[w.lower() for w in text.split()]])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=75, padding='post')
    y_pred, _, _, _ = model.predict(padded)
    pred_labels = [id2tag[i] for i in y_pred[0][:len(text.split())]]
    return list(zip(text.split(), pred_labels))

# Visualize with SpaCy
def visualize_with_spacy(predictions):
    nlp = spacy.blank("en")
    doc = nlp.make_doc(" ".join([w for w, _ in predictions]))
    ents = []
    start = 0
    for word, label in predictions:
        end = start + len(word)
        if label != "O":
            ents.append(spacy.tokens.Span(doc, doc.char_span(start, end).start, doc.char_span(start, end).end, label=label.split("-")[-1]))
        start = end + 1
    doc.ents = ents
    return doc

# Tweet input type
input_mode = st.radio("Input Type", ["Manual", "Upload File", "Twitter API"])

if input_mode == "Manual":
    tweet = st.text_area("Enter tweet", "Cristiano Ronaldo joined Al-Nassr Football Club in Saudi Arabia.")
    tweets = [tweet]

elif input_mode == "Upload File":
    uploaded = st.file_uploader("Upload a .txt or .csv with tweets", type=["txt", "csv"])
    if uploaded:
        if uploaded.name.endswith(".txt"):
            tweets = uploaded.read().decode().splitlines()
        else:
            df = pd.read_csv(uploaded)
            tweets = df.iloc[:, 0].dropna().tolist()

elif input_mode == "Twitter API":
    bearer = st.text_input("Twitter Bearer Token")
    query = st.text_input("Search Tweets (e.g. 'sports', 'movies')")
    count = st.slider("Number of Tweets", 1, 20, 5)
    tweets = []
    if bearer and query:
        headers = {"Authorization": f"Bearer {bearer}"}
        url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results={count}"
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            json_data = resp.json()
            tweets = [x["text"] for x in json_data.get("data", [])]
        else:
            st.error("API error or invalid Bearer Token")

# Model selection
model_type = st.selectbox("Choose NER model", ["BERT", "LSTM-CRF"])

# Analyze button
if st.button("Run NER on Tweets"):
    results = []
    for tweet in tweets:
        pred = get_bert_predictions(tweet) if model_type == "BERT" else get_lstm_crf_predictions(tweet)
        entities = [(w, tag) for w, tag in pred if tag != "O"]
        results.append({
            "Tweet": tweet,
            "Entities": ", ".join(f"{w} ({t})" for w, t in entities)
        })

        st.subheader(f"📝 Tweet:")
        st.markdown(f"```{tweet}```")

        st.markdown("### 🔍 Entities Found:")
        if entities:
            st.write(entities)
        else:
            st.info("No entities found.")

        # Display visualization
        try:
            doc = visualize_with_spacy(pred)
            html = spacy.displacy.render(doc, style="ent", jupyter=False)
            st.components.v1.html(html, height=200, scrolling=True)
        except:
            st.warning("Could not visualize entities with SpaCy.")

    # Export
    if results:
        df_out = pd.DataFrame(results)
        csv = df_out.to_csv(index=False).encode()
        st.download_button("📥 Download Results as CSV", csv, "annotated_tweets.csv", "text/csv")

