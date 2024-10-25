import fitz  # PyMuPDF para PDFs
from docx import Document
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import spacy
import streamlit as st

# Baixar os pacotes necessários de NLP
nltk.download('stopwords')
nltk.download('punkt')
spacy.cli.download('en_core_web_sm')
nlp = spacy.load('en_core_web_sm')

# Funções de extração de texto
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

# Função de pré-processamento de texto
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    processed_text = [token.lemma_ for token in nlp(" ".join(tokens)) if token.is_alpha and token.lemma_ not in stop_words]
    return " ".join(processed_text)

# Configuração da interface do Streamlit
st.title("Analisador de Currículos")
st.write("Faça upload da descrição da vaga e dos currículos para analisar e ranquear os mais adequados.")

# Input para a descrição da vaga
job_description = st.text_area("Descrição da Vaga", "Digite aqui os requisitos e descrição da vaga.")

# Upload dos arquivos de currículos
uploaded_files = st.file_uploader("Faça upload dos currículos (PDF ou DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

# Botão para iniciar a análise
if st.button("Analisar Currículos"):
    if job_description and uploaded_files:
        # Pré-processamento da descrição da vaga
        cleaned_job_description = preprocess_text(job_description)
        
        # Processar e pré-processar cada currículo
        processed_resumes = {}
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            if file_name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            elif file_name.endswith(".docx"):
                resume_text = extract_text_from_docx(uploaded_file)
            processed_resumes[file_name] = preprocess_text(resume_text)
        
        # Preparar documentos para vetorização
        documents = [cleaned_job_description] + list(processed_resumes.values())
        
        # Vetorização com TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Calcular similaridade de cosseno entre currículos e a descrição da vaga
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Criar DataFrame com resultados e exibir o ranking dos Top 5
        ranking = pd.DataFrame({"Currículo": list(processed_resumes.keys()), "Similaridade": similarity_scores})
        ranking = ranking.sort_values(by="Similaridade", ascending=False).head(5)
        
        st.write("Top 5 currículos mais adequados:")
        st.table(ranking)
    else:
        st.warning("Por favor, insira uma descrição da vaga e faça upload dos currículos.")
