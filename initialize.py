import os
import hashlib
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

import constants as C
import utils as U


# .env読み込みは初回に一度
load_dotenv()


def create_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def set_page():
    st.set_page_config(
        page_title="Dance School Guide Chat | ダンススクール案内チャット",
        page_icon="💃",
        layout="wide"
    )


# ==== CSV読み込み ==================================================
@st.cache_data(show_spinner=False)
def load_csv(path: str, encoding: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=encoding)
    missing = [c for c in C.INDEX_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSVに必要な列がありません: {missing}")
    df[C.INDEX_COLS] = df[C.INDEX_COLS].fillna("")
    return df


# ==== コーパス構築（チャンク化） ====================================
@st.cache_data(show_spinner=False)
def build_corpus(df: pd.DataFrame) -> List[Dict]:
    corpus: List[Dict] = []
    for idx, row in df.iterrows():
        text = U.to_text(row)
        if not text:
            continue
        chunks = U.split_into_chunks(text, C.CHUNK_SIZE, C.CHUNK_OVERLAP)
        for j, ch in enumerate(chunks):
            corpus.append({"row_index": int(idx), "chunk_index": j, "text": ch})
    return corpus


def _hash_corpus(corpus: List[Dict]) -> str:
    h = hashlib.sha256()
    for item in corpus:
        h.update(item["text"].encode("utf-8"))
    return h.hexdigest()


# ==== 埋め込み（生成＆保持） =======================================
@st.cache_resource(show_spinner=False)
def _embed_corpus_cached(texts: List[str], model: str):
    """
    clientを引数に含めないことでcacheのキーが安定。
    内部でOpenAIクライアントを都度生成（環境変数使用）。
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.embeddings.create(model=model, input=texts)
    embeddings = np.array([d.embedding for d in resp.data], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    return embeddings / norms


def ensure_embeddings(corpus: List[Dict]):
    corpus_key = _hash_corpus(corpus)
    if "embeddings_key" not in st.session_state or st.session_state.embeddings_key != corpus_key:
        with st.spinner("資料をインデックス化しています…"):
            texts = [c["text"] for c in corpus]
            EMB = _embed_corpus_cached(texts, C.EMBED_MODEL)
        st.session_state.embeddings = EMB
        st.session_state.embeddings_key = corpus_key
    return st.session_state.embeddings


# ==== セッション初期化 =============================================
def ensure_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{"role": "...", "content": "..."}]
    if "pairs" not in st.session_state:
        st.session_state.pairs = []     # [{"user": "...", "assistant": "..."}]
    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False