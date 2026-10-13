from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import Normalizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

st.title("Modèle Word2Vec")

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_resources():
    vocab = pd.read_csv(BASE_DIR / "vocab.csv", keep_default_na=False)
    word2idx = {}
    idx2word = {}
    for i in range(len(vocab)):
        mot = vocab["word"][i]
        indice = int(vocab["index"][i])
        word2idx[mot] = indice
        idx2word[indice] = mot
    vocab_size = 10000

    embedding_dim = 300
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(vocab_size, activation='softmax'))
    model.build(input_shape=(None, 4))

    model.load_weights(BASE_DIR / "word2vec.h5")

    vectors = model.layers[0].trainable_weights[0].numpy()
    return vectors, word2idx, idx2word, vocab_size


vectors, word2idx, idx2word, vocab_size = load_resources()


def dot_product(vec1, vec2):
    return np.sum((vec1*vec2))

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2)/np.sqrt(dot_product(vec1, vec1)*dot_product(vec2, vec2))

def find_closest(word_index, vectors, number_closest):
    list1=[]
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def compare(index_word1, index_word2, index_word3, vectors, number_closest):
    list1=[]
    query_vector = vectors[index_word1] - vectors[index_word2] + vectors[index_word3]
    normalizer = Normalizer()
    query_vector =  normalizer.fit_transform([query_vector], 'l2')
    query_vector= query_vector[0]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def print_closest(word, number=10):
    index_closest_words = find_closest(word2idx[word], vectors, number + 1)
    rows = []
    for index_word in index_closest_words:
        indice = int(index_word[1])
        if indice == 0:
            continue
        rows.append({"Mot": idx2word[indice], "Similarité cosinus": float(index_word[0])})
    st.dataframe(rows[:number], hide_index=True)


st.header("Mots les plus proches")

word = st.text_input("Mot", value="zombie").lower().strip()
number = st.slider("Nombre de mots proches", min_value=1, max_value=50, value=10)

if not word:
    st.info("Saisis un mot pour afficher ses voisins.")
elif word not in word2idx:
    st.warning(f"Le mot ' {word} ' n'est pas dans le vocabulaire.")
elif word2idx[word] >= vocab_size:
    st.warning(f"Le mot ' {word} ' est trop rare, il n'a pas été appris par le modèle.")
else:
    print_closest(word, number)
