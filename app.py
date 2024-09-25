import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keybert import KeyBERT
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re

# Download necessary NLTK resources
nltk.download('vader_lexicon')

# Set page title
st.set_page_config(page_title="DocuSense", layout="wide")

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Model and tokenizer loading
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)

# Explicitly specify the model to load on CPU
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint,
    torch_dtype=torch.float32
).to('cpu')  # Ensure the model is on CPU without offloading

# File loader and preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts = final_texts + text.page_content
    return final_texts

# LLM pipeline (summarization)
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50)
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

# Sentiment analysis
def sentiment_analysis(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores

# Keyword extraction using KeyBERT
def extract_keywords(text):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
    return keywords

# Topic modeling using LDA
def topic_modeling(text):
    # Preprocess text
    clean_text = re.sub(r'\W+', ' ', text.lower())
    
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    text_vectorized = vectorizer.fit_transform([clean_text])
    
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(text_vectorized)

    topics = lda.components_
    terms = vectorizer.get_feature_names_out()

    top_words_per_topic = []
    for topic in topics:
        top_words = [terms[i] for i in topic.argsort()[-10:]]
        top_words_per_topic.append(top_words)

    return top_words_per_topic

@st.cache_data
# Function to display the PDF of a given file 
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit app layout
def main():
    st.title("DocuSense — Document Summarization and Analysis App")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        if st.button("Summarize and Analyze"):
            col1, col2 = st.columns(2)
            filepath = "data/"+uploaded_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            with col1:
                st.info("File Uploaded! :)")
                displayPDF(filepath)

            with col2:
                # Summarization
                summary = llm_pipeline(filepath)
                st.info("Summarization Complete")
                st.success(summary)

                # Text preprocessing for analysis
                processed_text = file_preprocessing(filepath)

                # Sentiment analysis
                sentiment = sentiment_analysis(processed_text)
                st.info("Sentiment Analysis")
                st.markdown(f"**Negative:** {sentiment['neg']*100:.2f}%")
                st.markdown(f"**Neutral:** {sentiment['neu']*100:.2f}%")
                st.markdown(f"**Positive:** {sentiment['pos']*100:.2f}%")
                st.markdown(f"**Compound Score:** {sentiment['compound']:.4f}")

                # Keyword extraction
                keywords = extract_keywords(processed_text)
                st.info("Keyword Extraction")
                st.markdown("**Top Keywords:**")
                for keyword, score in keywords:
                    st.markdown(f"- **{keyword}** (Score: {score:.4f})")

                # Topic modeling
                topics = topic_modeling(processed_text)
                st.info("Topic Modeling")
                for i, topic in enumerate(topics):
                    st.markdown(f"**Topic {i + 1}:** {', '.join(topic)}")

if __name__ == "__main__":
    main()