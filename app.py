import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 固定で参照するCSVファイル
CSV_FILE = "data.csv"
ENCODING = "utf-8"  # 必要に応じて "cp932" (Shift_JIS) などに変更

st.set_page_config(page_title="CSV QA Chatbot", page_icon="💬", layout="wide")

st.title("💬 固定CSVベース Q&A チャットボット")
st.caption("あらかじめ用意したCSVの内容をもとに、質問に自動回答します。分からない場合は「分からない」と答えます。")

# 回答のしきい値
threshold = st.sidebar.slider(
    "回答の確信度しきい値（コサイン類似度）",
    0.0, 1.0, 0.30, 0.01,
    help="この値より低い場合は「分からない」と回答します。"
)
top_k = st.sidebar.number_input("関連候補の表示件数 (Top-K)", 1, 10, 3)
show_sources = st.sidebar.checkbox("関連候補（ソース）を表示する", value=True)

# CSVの読み込み
try:
    df = pd.read_csv(CSV_FILE, encoding=ENCODING)
except Exception as e:
    st.error(f"CSVファイルの読み込みに失敗しました: {e}")
    st.stop()

st.success(f"CSVを読み込みました。行数: {len(df)}, 列: {list(df.columns)}")

# 検索対象列・回答列を指定（ここでは固定例）
index_cols = ["question"]   # 検索対象
answer_col = "answer"       # 回答に使う列

# インデックス作成
def build_index(df: pd.DataFrame, index_cols):
    texts = []
    for _, row in df.iterrows():
        parts = []
        for c in index_cols:
            val = row.get(c, "")
            if pd.isna(val):
                val = ""
            parts.append(str(val))
        texts.append(" ".join(parts))
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), min_df=1)
    X = vectorizer.fit_transform(texts)
    return vectorizer, X

vectorizer, X = build_index(df, index_cols)

def retrieve_and_answer(question: str):
    if not question.strip():
        return None
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, X)[0]
    order = np.argsort(-sims)
    best_idx = order[0]
    best_score = float(sims[best_idx])

    if best_score < threshold:
        return {
            "answer": "すみません、その質問の答えは分かりませんでした。",
            "score": best_score,
            "unknown": True,
            "candidates": [(int(i), float(sims[i])) for i in order[:top_k]]
        }

    ans = df.iloc[best_idx][answer_col] if answer_col in df.columns else str(df.iloc[best_idx].to_dict())
    return {
        "answer": str(ans),
        "score": best_score,
        "unknown": False,
        "candidates": [(int(i), float(sims[i])) for i in order[:top_k]]
    }

# セッションステート初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# チャットUI
st.markdown("---")
st.subheader("💬 チャット")

col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("質問を入力してください", value="")
with col2:
    st.write("")
    st.write("")
    ask_clicked = st.button("送信")

if st.button("🧹 履歴をクリア"):
    st.session_state.messages = []

# 質問処理
if ask_clicked and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})
    res = retrieve_and_answer(user_input)
    if res is None:
        st.session_state.messages.append({"role": "assistant", "content": "質問が空です。"})
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": res["answer"],
            "meta": res
        })

# 表示
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            meta = msg.get("meta")
            if meta and not meta.get("unknown", True) and show_sources:
                st.caption(f"確信度（類似度）: {meta['score']:.3f}")
                with st.expander("🔎 関連候補（上位）"):
                    rows = []
                    for idx, sc in meta["candidates"]:
                        row = df.iloc[idx]
                        rows.append({"row_index": idx, "similarity": round(sc, 4)} | row.to_dict())
                    st.dataframe(pd.DataFrame(rows))
            elif meta and meta.get("unknown", False):
                st.caption(f"最も近い候補の類似度: {meta['score']:.3f}（しきい値未満のため『分からない』）")