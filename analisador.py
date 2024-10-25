import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from collections import Counter
import streamlit as st

# Baixar o modelo de língua, se não estiver disponível
if not os.path.exists("pt_core_news_sm"):
    os.system('python -m spacy download pt_core_news_sm')

# Carregar o modelo de língua
nlp = spacy.load("pt_core_news_sm")

def carregar_curriculos(caminho):
    curriculos = []
    for arquivo in os.listdir(caminho):
        if arquivo.endswith('.txt') or arquivo.endswith('.pdf'):
            with open(os.path.join(caminho, arquivo), 'r', encoding='utf-8') as file:
                curriculos.append(file.read())
    return curriculos

def extrair_palavras_chave(texto, num_palavras=5):
    doc = nlp(texto)
    palavras = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    contagem = Counter(palavras)
    return contagem.most_common(num_palavras)

def calcular_pontuacao(curriculo, descricao_vaga):
    palavras_chave_curriculo = extrair_palavras_chave(curriculo)
    palavras_chave_vaga = extrair_palavras_chave(descricao_vaga)
    
    pontuacao = 0
    for palavra, _ in palavras_chave_vaga:
        if palavra in dict(palavras_chave_curriculo):
            pontuacao += 1  # Aumenta a pontuação se a palavra-chave estiver presente
    return pontuacao

def analisar_curriculos(curriculos, descricao_vaga):
    vectorizer = TfidfVectorizer()
    documentos = curriculos + [descricao_vaga]
    tfidf_matrix = vectorizer.fit_transform(documentos)
    
    similaridade = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    pontuacoes = [calcular_pontuacao(curriculo, descricao_vaga) for curriculo in curriculos]
    
    return similaridade.flatten(), pontuacoes

def salvar_resultados(resultados):
    df = pd.DataFrame(resultados, columns=["Currículo", "Similaridade", "Pontuação"])
    df.to_csv('resultados.csv', index=False)

# Configurar a interface Streamlit
st.title("Analisador de Currículos")

# Selecionar a pasta com os currículos
pasta_curriculos = st.text_input("Insira o caminho da pasta com currículos:")
descricao_vaga = st.text_area("Descrição da Vaga:")

if st.button("Analisar Currículos"):
    if not os.path.exists(pasta_curriculos):
        st.warning("A pasta especificada não existe.")
    else:
        curriculos = carregar_curriculos(pasta_curriculos)
        
        if not curriculos:
            st.warning("Nenhum currículo encontrado na pasta selecionada.")
        else:
            similaridade, pontuacoes = analisar_curriculos(curriculos, descricao_vaga)
            resultados = [(os.path.basename(arquivo), sim, pont) for arquivo, sim, pont in zip(os.listdir(pasta_curriculos), similaridade, pontuacoes)]
            resultados = sorted(resultados, key=lambda x: (x[1], x[2]), reverse=True)
            salvar_resultados(resultados)
            st.write("Resultados da Análise:")
            for curriculo, sim, pont in resultados:
                st.write(f"{curriculo} - Similaridade: {sim:.4f}, Pontuação: {pont}")
