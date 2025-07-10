import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForTokenClassification
import numpy as np
import streamlit as st

# Domain-specific BERT checkpoints
BERT_DIR = {
    "General": "bert_models/general",
    "Medical": "bert_models/medical",
    "Legal": "bert_models/legal"
}

def load_models():
    # Load LSTM-CRF
    crf_model = tf.keras.models.load_model("crf_model", compile=False)
    with open("crf_model/tokenizer.json") as f:
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        crf_tokenizer = tokenizer_from_json(f.read())
    id2tag_crf = {i: tag for i, tag in enumerate(["O","B-person","I-person","B-location","I-location","B-company","I-company","B-product","I-product","B-movie","I-movie","B-tvshow","I-tvshow","B-sportsteam","I-sportsteam","B-other","I-other"])}
    # Load BERT models
    bert_models = { d: TFAutoModelForTokenClassification.from_pretrained(BERT_DIR[d]) for d in BERT_DIR }
    bert_tokenizers = { d: AutoTokenizer.from_pretrained(BERT_DIR[d]) for d in BERT_DIR }
    return crf_model, crf_tokenizer, id2tag_crf, bert_models, bert_tokenizers, id2tag_crf

def predict_entities(text, model_type, bert_models, crf_model, crf_tokenizer, bert_tokenizers, id2tag, domain=None):
    tokens, tags = [], []
    if model_type == "LSTM‑CRF":
        seq = crf_tokenizer.texts_to_sequences([text.split()])
        pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=75, padding='post')
        preds = crf_model.predict(pad)[0]
        tags = [id2tag[i] for i in np.argmax(preds, axis=-1)][: len(seq[0])]
        tokens = text.split()
    else:
        tokenizer = bert_tokenizers[domain]
        model = bert_models[domain]
        inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
        logits = model(**inputs).logits.numpy()[0]
        preds = np.argmax(logits, axis=-1)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        tags = [id2tag[i] for i in preds]
    return list(zip(tokens, tags))

def visualize_ner(text, entities):
    def color_of(tag):
        cols = {
            "person":"#f94144","location":"#f3722c","company":"#90be6d","product":"#277da1",
            "movie":"#9c89b8","tvshow":"#577590","sportsteam":"#43aa8b","other":"#f9c74f"
        }
        return cols.get(tag.split("-")[-1].lower(), "#eee")
    st.markdown("### Annotated Text")
    html = ""
    for tok, tag in entities:
        if tag=="O":
            html += tok + " "
        else:
            html += f"<span style='background:{color_of(tag)}'>{tok}</span> "
    st.markdown(html, unsafe_allow_html=True)
