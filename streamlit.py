import streamlit as st
import numpy as np
import pickle

@st.cache_resource
def load_embeddings():
    # 讀取向量與字典
    vectors = np.load("vectors.npy")

    with open("word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)

    with open("idx2word.pkl", "rb") as f:
        idx2word = pickle.load(f)

    # 做 L2 正規化，讓 cosine similarity 變成單純的內積
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    vectors_norm = vectors / norms

    return vectors_norm, word2idx, idx2word


vectors, word2idx, idx2word = load_embeddings()


def most_similar(word, topn=10):
    word = word.lower()

    if word not in word2idx:
        return []

    idx = word2idx[word]

    # 取出目標向量
    target_vec = vectors[idx]

    # 計算 cosine similarity（因為已經正規化，可以直接內積）
    sims = vectors @ target_vec

    # 把自己那一格的相似度設成 -inf，不要被排進結果
    sims[idx] = -1.0

    # 取前 topn 個 index
    best_idx = np.argsort(sims)[::-1][:topn]

    results = []
    for i in best_idx:
        # 有些 index 可能沒有對應的字（保險處理）
        token = idx2word.get(int(i), None)
        if token is not None:
            results.append((token, float(sims[i])))

    return results


# ---------------------- Streamlit 介面 ----------------------

st.title("Model Word2Vec")

st.write("輸入一個英文單字，我會幫你找出向量空間裡最相似的單字。")

query = st.text_input("請輸入單字（英文）:")

topn = st.slider("顯示幾個相似單字", min_value=5, max_value=20, value=10)

if query:
    results = most_similar(query, topn=topn)

    if not results:
        st.warning("這個單字不在詞彙表裡，可能是太冷門或有打錯字 🥲")
    else:
        st.subheader(f"和 **{query}** 最接近的單字：")
        for w, score in results:
            st.write(f"- {w}  （cosine similarity = {score:.3f}）")