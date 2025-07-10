import streamlit as st
from model_utils import load_models, predict_entities, visualize_ner

st.set_page_config(page_title="Domain‑Aware Twitter NER", layout="centered")
st.title("🔍 Domain‑Aware NER for Tweets")
st.markdown("Choose your model and domain to detect named entities in tweets.")

# Load models once
@st.cache_resource
def init():
    return load_models()

models = init()
crud = { "LSTM‑CRF": models[0:3], "BERT": models[3:] }

sentence = st.text_area("Enter tweet or sentence:", "Patient took Aspirin in New York.")

model_choice = st.selectbox("Select model:", ["LSTM‑CRF", "BERT"])
domain_choice = st.selectbox("Select domain:", ["General", "Medical", "Legal"])

if st.button("Run NER"):
    if sentence.strip() == "":
        st.warning("Please input text.")
    else:
        if model_choice == "LSTM‑CRF":
            crf_model, crf_tokenizer, id2tag = crud["LSTM‑CRF"]
            results = predict_entities(sentence, "LSTM‑CRF", None, crf_model, crf_tokenizer, None, id2tag)
        else:
            bert_models, bert_tokenizers, id2tag2 = crud["BERT"]
            results = predict_entities(sentence, "BERT", bert_models, None, None, bert_tokenizers, id2tag2, domain_choice)
        st.write(results)
        visualize_ner(sentence, results)
