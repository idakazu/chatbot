import streamlit as st
from openai import OpenAI

import constants as C
import initialize as I
import components as V
import utils as U


def main():
    # ---- ページ設定・クライアント ----
    I.set_page()
    client: OpenAI = I.create_client()

    # ---- データ読込・コーパス構築・埋め込み確保 ----
    df = I.load_csv(C.CSV_FILE, C.ENCODING)
    corpus = I.build_corpus(df)                 # utils.to_text / split_into_chunks を内部で使用
    EMB = I.ensure_embeddings(corpus)           # 必要なら埋め込みを生成＆st.session_stateに保持
    I.ensure_session_state()                    # 履歴などの初期化

    # ---- 画面描画（ヘッダー、履歴、入力欄）----
    V.render_header()
    V.render_chat_history(st.session_state.messages)

    user_input, ask_clicked = V.render_input(len(st.session_state.messages))

    # ---- 質問処理 ----
    if ask_clicked and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("💭 回答を考えています..."):
            answer = U.answer_pipeline(
                question=user_input,
                corpus=corpus,
                EMB=EMB,
                client=client
            )
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.pairs.append({"user": user_input, "assistant": answer})
        st.rerun()

    # ---- クリア（新規チャット） ----
    if V.render_clear_button():
        st.session_state.messages = []
        st.session_state.pairs = []
        st.rerun()


if __name__ == "__main__":
    main()