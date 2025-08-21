import streamlit as st


def render_header():
    st.title("💃 Multilingual Dance School Guide | 多言語ダンススクール案内")
    st.caption("Ask me anything about dance schools in any language! | どの言語でもダンススクールについて聞いてください！🌍✨")
    st.markdown("---")
    st.subheader("💬 Chat | チャット")

    # LINE風CSS
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


def render_chat_history(messages):
    if messages:
        for msg in messages:
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
        st.info("👋 Hello! I'd love to hear anything about dance schools. | こんにちは！ダンススクールについて何でもお聞きください。")


def render_input(message_count: int):
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    with col1:
        input_key = f"message_input_{message_count}"
        user_input = st.text_input(
            "💭 Enter your message | メッセージを入力",
            value="",
            placeholder="Please feel free to ask me anything about dance schools... | ダンススクールについて何でも聞いてください...",
            key=input_key
        )
    with col2:
        st.write(""); st.write("")
        ask_clicked = st.button("➤", help="送信")
    return user_input, ask_clicked


def render_clear_button():
    return st.button("New Project | 新しいチャットにする")