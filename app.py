import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===== 固定設定 =====
CSV_FILE = "Dance_School.csv"
ENCODING = "utf-8"   # 必要に応じて "cp932" などに変更
THRESHOLD = 0.30     # 類似度しきい値（下回れば「分からない」）
TOP_K = 5            # 内部で参照する候補件数（UIには出さない）
SHOW_SOURCES = False # デバッグ用（Trueで候補表示）
# コンテキスト（履歴）設定
CONTEXT_TURNS = 2    # 直近のユーザー質問を何件まで合成するか
CONTEXT_DECAY = 0.7  # 古い質問ほど重みを下げる(0.0-1.0)
USE_TOPIC_LOCK = True          # 最初に当たったトピックを優先
TOPIC_LOCK_COL = "category"    # トピックとして使う列名（CSVにあれば有効）

st.set_page_config(page_title="CSV QA Chatbot", page_icon="💬", layout="wide")
st.title("💬 Q&A チャットボット")
st.caption("自動で回答します。")

# ===== CSV読込 =====
try:
    df = pd.read_csv(CSV_FILE, encoding=ENCODING)
except Exception as e:
    st.error(f"CSVファイルの読み込みに失敗しました: {e}")
    st.stop()

# 検索対象列・回答列（固定）
INDEX_COLS = ["question"]
ANSWER_COL = "answer"
HAS_TOPIC = USE_TOPIC_LOCK and (TOPIC_LOCK_COL in df.columns)

# ===== インデックス作成 =====
def build_index(df: pd.DataFrame, index_cols):
    texts = []
    for _, row in df.iterrows():
        parts = []
        for c in index_cols:
            val = row.get(c, "")
            val = "" if pd.isna(val) else str(val)
            parts.append(val)
        texts.append(" ".join(parts))
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), min_df=1)
    X = vectorizer.fit_transform(texts)
    return vectorizer, X

vectorizer, X = build_index(df, INDEX_COLS)

# ===== 取得・回答 =====
def retrieve_and_answer(current_question: str, history_user_questions: list, topic_hint: str | None):
    """
    history_user_questions: 直近から古い順に並んだユーザー質問のリスト（現在の質問は含めない）
    topic_hint: 既定トピック（categoryなど）。優先的にそのトピックの候補を上位化
    """
    q = current_question.strip()
    if not q:
        return None

    # --- クエリベクトルを履歴と合成 ---
    q_vec = vectorizer.transform([q])
    if CONTEXT_TURNS > 0 and len(history_user_questions) > 0:
        # 直近の質問から順に最大CONTEXT_TURNS件、減衰重みで合成
        for i, prev_q in enumerate(history_user_questions[:CONTEXT_TURNS]):
            w = (CONTEXT_DECAY ** (i + 1))
            q_vec += w * vectorizer.transform([prev_q])

    sims = cosine_similarity(q_vec, X)[0]  # shape: (n_docs,)
    order = np.argsort(-sims)

    # --- トピック固定が有効なら、そのトピックに属する候補を少し優遇 ---
    if HAS_TOPIC and topic_hint:
        # 同トピック行に微小ボーナス付与（類似度の0.02上乗せなど）
        bonus = np.zeros_like(sims)
        same_topic_mask = (df[TOPIC_LOCK_COL].astype(str) == str(topic_hint))
        bonus[same_topic_mask.values] = 0.02  # 調整可
        sims_with_bonus = sims + bonus
        order = np.argsort(-sims_with_bonus)

    best_idx = int(order[0])
    best_score = float(sims[best_idx])

    # しきい値チェック
    if best_score < THRESHOLD:
        return {
            "answer": "すみません、その質問の答えは分かりませんでした。",
            "score": best_score,
            "unknown": True,
            "candidates": [(int(i), float(sims[i])) for i in order[:TOP_K]],
            "topic_hint": topic_hint  # 変更なし
        }

    # 回答取得
    if ANSWER_COL in df.columns:
        ans = df.iloc[best_idx][ANSWER_COL]
    else:
        ans = str(df.iloc[best_idx].to_dict())

    # 新しいトピックヒント（最初の確信あるヒットで固定）
    new_topic_hint = topic_hint
    if HAS_TOPIC and topic_hint is None:
        maybe_topic = df.iloc[best_idx][TOPIC_LOCK_COL]
        if pd.notna(maybe_topic):
            new_topic_hint = str(maybe_topic)

    return {
        "answer": str(ans),
        "score": best_score,
        "unknown": False,
        "candidates": [(int(i), float(sims[i])) for i in order[:TOP_K]],
        "topic_hint": new_topic_hint
    }

# ===== セッション状態 =====
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role": "user"/"assistant", "content": str}
if "topic_hint" not in st.session_state:
    st.session_state.topic_hint = None  # 例: category名

# ===== チャットUI =====
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
    st.session_state.topic_hint = None

# ===== 処理 =====
if ask_clicked and user_input.strip():
    # 直近のユーザー質問（現在は含めず）を新しい順で抽出
    past_user_qs = [m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"]
    res = retrieve_and_answer(user_input, past_user_qs, st.session_state.topic_hint)

    # ログ追加
    st.session_state.messages.append({"role": "user", "content": user_input})
    if res is None:
        st.session_state.messages.append({"role": "assistant", "content": "質問が空です。"})
    else:
        st.session_state.messages.append({"role": "assistant", "content": res["answer"], "meta": res})
        st.session_state.topic_hint = res.get("topic_hint", st.session_state.topic_hint)

# ===== 表示 =====
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            if SHOW_SOURCES:
                meta = msg.get("meta")
                if meta:
                    st.caption(f"score: {meta.get('score', 0):.3f}, topic: {st.session_state.topic_hint}")