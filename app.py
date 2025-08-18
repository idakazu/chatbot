import os
import re
import hashlib
from typing import List, Dict, Tuple, Generator

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv()

# ==== 固定設定 =====================================================
CSV_FILE = "dance_school.csv"          # 固定の参考資料CSV
ENCODING = "utf-8"
INDEX_COLS = ["name", "overview", "access"]
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 6
MIN_TOP_SIM = 0.20
USE_HISTORY_TURNS = 2
MAX_CONTEXT_CHARS = 4500
STREAM_ANSWER = True   # リアルタイム表示
# ==================================================================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Dance School Guide Chat | ダンススクール案内チャット", page_icon="💃", layout="wide")
st.title("💃 Multilingual Dance School Guide | 多言語ダンススクール案内")
st.caption("Ask me anything about dance schools in any language! | どの言語でもダンススクールについて聞いてください！🌍✨")

# ==== CSV読み込み ==================================================
@st.cache_data(show_spinner=False)
def load_csv(path: str, encoding: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding=encoding)

try:
    df = load_csv(CSV_FILE, ENCODING)
except Exception as e:
    st.error(f"CSVファイルの読み込みに失敗しました: {e}")
    st.stop()

missing = [c for c in INDEX_COLS if c not in df.columns]
if missing:
    st.error(f"CSVに必要な列がありません: {missing}")
    st.stop()

df[INDEX_COLS] = df[INDEX_COLS].fillna("")

# ==== チャンク化 ==================================================
SENT_SPLIT_RE = re.compile(r"(?<=[。．.!?！？])\s+|\n+")

def to_text(row: pd.Series) -> str:
    return " ".join(str(row[c]) for c in INDEX_COLS if c in row and str(row[c]).strip())

def split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    sents = [s.strip() for s in SENT_SPLIT_RE.split(text) if s and s.strip()]
    chunks, buf = [], ""
    for s in sents:
        if len(buf) + len(s) <= chunk_size:
            buf = f"{buf} {s}".strip()
        else:
            if buf:
                chunks.append(buf)
            if overlap > 0 and len(buf) > overlap:
                buf = buf[-overlap:] + " " + s
            else:
                buf = s
    if buf:
        chunks.append(buf)
    return chunks

@st.cache_data(show_spinner=False)
def build_corpus(df: pd.DataFrame) -> List[Dict]:
    corpus = []
    for idx, row in df.iterrows():
        text = to_text(row)
        if not text:
            continue
        chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for j, ch in enumerate(chunks):
            corpus.append({"row_index": int(idx), "chunk_index": j, "text": ch})
    return corpus

corpus = build_corpus(df)

# ==== 埋め込み =====================================================
def hash_corpus(corpus: List[Dict]) -> str:
    h = hashlib.sha256()
    for item in corpus:
        h.update(item["text"].encode("utf-8"))
    return h.hexdigest()

@st.cache_resource(show_spinner=False)
def embed_corpus(corpus: List[Dict], model: str):
    texts = [c["text"] for c in corpus]
    resp = client.embeddings.create(model=model, input=texts)
    embeddings = np.array([d.embedding for d in resp.data], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    return embeddings / norms

corpus_key = hash_corpus(corpus)
if "embeddings_key" not in st.session_state or st.session_state.embeddings_key != corpus_key:
    with st.spinner("資料をインデックス化しています…"):
        EMB = embed_corpus(corpus, EMBED_MODEL)
    st.session_state.embeddings = EMB
    st.session_state.embeddings_key = corpus_key
else:
    EMB = st.session_state.embeddings

# ==== 検索 =========================================================
def embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)

def topk_search(q: str, k: int) -> List[Tuple[int, float]]:
    qv = embed_query(q)
    sims = (EMB @ qv)
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]

def build_context(chosen: List[Tuple[int, float]], max_chars: int) -> str:
    parts, total = [], 0
    for idx, score in chosen:
        snippet = f"[{corpus[idx]['row_index']}-{corpus[idx]['chunk_index']} sim={score:.3f}]\n{corpus[idx]['text']}"
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n---\n\n".join(parts)

# ==== 言語判定と多言語対応 ==========================================
def detect_language_and_get_system_prompt(question: str) -> tuple[str, str]:
    """質問の言語を判定し、適切なシステムプロンプトを返す"""
    
    # OpenAI APIで言語判定
    try:
        detection_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user", 
                "content": f"""以下のテキストの言語を判定してください。回答は言語コードのみ（例: ja, en, zh, ko, es, fr, de, etc.）で返してください。

テキスト: {question}"""
            }],
            max_tokens=10,
            temperature=0.0
        )
        language_code = detection_response.choices[0].message.content.strip().lower()
    except:
        # エラーの場合は日本語をデフォルトとする
        language_code = "ja"
    
    # 言語別のシステムプロンプトと「分からない」メッセージ
    language_prompts = {
        "ja": {
            "system": "あなたは厳格なドキュメント回答アシスタントです。与えられたコンテキストに基づいて日本語で簡潔に回答してください。コンテキストに根拠が無い場合は、必ず『分かりません』とだけ答えてください。",
            "no_answer": "分かりません"
        },
        "en": {
            "system": "You are a strict document-based assistant. Answer concisely in English based only on the given context. If there is no evidence in the context, you must answer only 'I don't know'.",
            "no_answer": "I don't know"
        },
        "zh": {
            "system": "您是一个严格的文档回答助手。请仅基于给定的上下文用中文简洁回答。如果上下文中没有根据，您必须只回答'我不知道'。",
            "no_answer": "我不知道"
        },
        "ko": {
            "system": "당신은 엄격한 문서 기반 답변 어시스턴트입니다. 주어진 컨텍스트에만 기반하여 한국어로 간결하게 답변해주세요. 컨텍스트에 근거가 없으면 반드시 '모르겠습니다'라고만 답변해주세요.",
            "no_answer": "모르겠습니다"
        },
        "es": {
            "system": "Eres un asistente estricto basado en documentos. Responde de manera concisa en español basándote solo en el contexto dado. Si no hay evidencia en el contexto, debes responder solo 'No lo sé'.",
            "no_answer": "No lo sé"
        },
        "fr": {
            "system": "Vous êtes un assistant strict basé sur des documents. Répondez de manière concise en français en vous basant uniquement sur le contexte donné. S'il n'y a pas de preuve dans le contexte, vous devez répondre seulement 'Je ne sais pas'.",
            "no_answer": "Je ne sais pas"
        },
        "de": {
            "system": "Sie sind ein strenger dokumentenbasierter Assistent. Antworten Sie prägnant auf Deutsch, basierend nur auf dem gegebenen Kontext. Wenn es keine Belege im Kontext gibt, müssen Sie nur 'Ich weiß es nicht' antworten.",
            "no_answer": "Ich weiß es nicht"
        }
    }
    
    # 対応している言語の場合はそのプロンプトを、そうでなければ英語をデフォルトに
    if language_code in language_prompts:
        selected = language_prompts[language_code]
    else:
        selected = language_prompts["en"]
    
    return language_code, selected["system"], selected["no_answer"]

# ==== LLM呼び出し（ストリーミング対応・多言語対応） ===================
def stream_llm_answer(question: str, context: str, history_pairs: List[Dict]) -> Generator[str, None, None]:
    # 言語判定とシステムプロンプト取得
    language_code, system_prompt, no_answer_message = detect_language_and_get_system_prompt(question)
    
    messages = [{"role": "system", "content": system_prompt}]
    for h in history_pairs[-USE_HISTORY_TURNS:]:
        messages.append({"role": "user", "content": h.get("user", "")})
        if h.get("assistant"):
            messages.append({"role": "assistant", "content": h["assistant"]})
    messages.append({
        "role": "user",
        "content": f"### コンテキスト\n{context}\n\n### 質問\n{question}\n\n※根拠が無ければ『分かりません』と答えてください。"
    })

    if not STREAM_ANSWER:
        resp = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, temperature=0.0, max_tokens=400
        )
        return resp.choices[0].message.content.strip()

    # ストリーミング
    stream = client.chat.completions.create(
        model=CHAT_MODEL, messages=messages, temperature=0.0, max_tokens=400, stream=True
    )
    full = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full += delta
        yield delta  # 増分を返す
    yield f"__FINAL__{full}"

# ==== セッション状態（リロードで消える） =============================
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "...", "content": "...", "ts": "..."}]
if "pairs" not in st.session_state:
    st.session_state.pairs = []     # [{"user": "...", "assistant": "..."}]
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False  # 入力欄クリアフラグ

# ==== チャット履歴表示（LINE風） ====================================
st.markdown("---")
st.subheader("💬 Chat | チャット")

# カスタムCSS（LINE風スタイリング）
st.markdown("""
<style>
.user-message {
    background-color: #007AFF;
    color: white;
    padding: 10px 15px;
    border-radius: 20px;
    margin: 10px 0;
    margin-left: 20%;
    text-align: right;
    max-width: 70%;
    float: right;
    clear: both;
}
.assistant-message {
    background-color: #E5E5EA;
    color: black;
    padding: 10px 15px;
    border-radius: 20px;
    margin: 10px 0;
    margin-right: 20%;
    text-align: left;
    max-width: 70%;
    float: left;
    clear: both;
}
.message-spacer {
    height: 10px;
    clear: both;
}
</style>
""", unsafe_allow_html=True)

# チャット履歴表示エリア
if st.session_state.messages:
    # 既存の全メッセージを表示（古いものから順に）
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'''
            <div class="user-message">
                {msg["content"]}
            </div>
            <div class="message-spacer"></div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="assistant-message">
                {msg["content"]}
            </div>
            <div class="message-spacer"></div>
            ''', unsafe_allow_html=True)
else:
    # 初回表示時のウェルカムメッセージ
    st.info("👋 Hello! I'd love to hear anything about dance schools. | こんにちは！ダンススクールについて何でもお聞きください。")

# 入力欄（下部固定風）
st.markdown("---")
col1, col2 = st.columns([5, 1])
with col1:
    # 入力欄をクリアするために、clear_inputフラグによって動的にkeyを変更
    input_key = f"message_input_{len(st.session_state.messages)}"
    user_input = st.text_input("💭 Enter your message | メッセージを入力", value="", placeholder="Please feel free to ask me anything about dance schools... | ダンススクールについて何でも聞いてください...", key=input_key)
with col2:
    st.write(""); st.write("")
    ask_clicked = st.button("➤", help="送信")

# ==== パイプライン ==================================================
def answer_pipeline(question: str) -> str:
    """質問に対する回答を生成（多言語対応）"""
    # 言語判定
    language_code, system_prompt, no_answer_message = detect_language_and_get_system_prompt(question)
    
    top = topk_search(question, TOP_K)
    if not top or top[0][1] < MIN_TOP_SIM:
        return no_answer_message
    
    context = build_context(top, MAX_CONTEXT_CHARS)
    
    # LLMに回答生成を依頼
    try:
        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n参考資料:\n{context}"},
            {"role": "user", "content": question}
        ]
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # エラーメッセージも言語に応じて変更
        error_messages = {
            "ja": f"回答生成中にエラーが発生しました。😅\n{str(e)}",
            "en": f"An error occurred while generating the answer. 😅\n{str(e)}",
            "zh": f"生成回答时发生错误。😅\n{str(e)}",
            "ko": f"답변 생성 중 오류가 발생했습니다. 😅\n{str(e)}",
            "es": f"Ocurrió un error al generar la respuesta. 😅\n{str(e)}",
            "fr": f"Une erreur s'est produite lors de la génération de la réponse. 😅\n{str(e)}",
            "de": f"Ein Fehler ist bei der Antwortgenerierung aufgetreten. 😅\n{str(e)}"
        }
        return error_messages.get(language_code, error_messages["en"])

# ==== 実行（履歴を残す・リロードでクリア） =========================
if ask_clicked and user_input.strip():
    # ユーザーメッセージを履歴に追加
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 回答生成中の表示
    with st.spinner("💭 回答を考えています..."):
        answer = answer_pipeline(user_input)

    # アシスタントメッセージを履歴に追加
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.pairs.append({"user": user_input, "assistant": answer})
    
    # ページを再実行して最新の履歴を表示（入力欄は新しいkeyで自動的にクリアされる）
    st.rerun()

# クリアボタン
if st.button("New Project | 新しいチャットにする"):
    st.session_state.messages = []
    st.session_state.pairs = []
    st.rerun()