from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI

import constants as C


# ==== 基本テキスト処理 =============================================
def to_text(row: pd.Series) -> str:
    return " ".join(str(row[c]) for c in C.INDEX_COLS if c in row and str(row[c]).strip())


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    sents = [s.strip() for s in C.SENT_SPLIT_RE.split(text) if s and s.strip()]
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


# ==== 埋め込みクエリ＆検索 ==========================================
def embed_query(q: str, client: OpenAI) -> np.ndarray:
    resp = client.embeddings.create(model=C.EMBED_MODEL, input=[q])
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


def topk_search(q: str, EMB: np.ndarray, client: OpenAI, k: int) -> List[Tuple[int, float]]:
    qv = embed_query(q, client)
    sims = (EMB @ qv)
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]


def build_context(chosen: List[Tuple[int, float]], corpus: List[Dict], max_chars: int) -> str:
    parts, total = [], 0
    for idx, score in chosen:
        snippet = f"[{corpus[idx]['row_index']}-{corpus[idx]['chunk_index']} sim={score:.3f}]\n{corpus[idx]['text']}"
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n---\n\n".join(parts)


# ==== 言語判定＆プロンプト =========================================
def detect_language_and_get_system_prompt(question: str, client: OpenAI) -> tuple[str, str, str]:
    """質問の言語を判定し、(language_code, system_prompt, no_answer_msg) を返す"""
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

    # モデルで言語判定（例外時はja）
    try:
        detection_response = client.chat.completions.create(
            model=C.DETECT_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "以下のテキストの言語を判定してください。回答は言語コードのみ（例: ja, en, zh, ko, es, fr, de, etc.）で返してください。\n\n"
                    f"テキスト: {question}"
                )
            }],
            max_tokens=10,
            temperature=0.0
        )
        language_code = detection_response.choices[0].message.content.strip().lower()
    except Exception:
        language_code = "ja"

    selected = language_prompts.get(language_code, language_prompts["en"])
    return language_code, selected["system"], selected["no_answer"]


# ==== 回答生成（RAGパイプライン） ===================================
def answer_pipeline(
    question: str,
    corpus: List[Dict],
    EMB: np.ndarray,
    client: OpenAI
) -> str:
    # 言語判定
    language_code, system_prompt, no_answer_message = detect_language_and_get_system_prompt(question, client)

    # 近傍検索
    top = topk_search(question, EMB, client, C.TOP_K)
    if not top or top[0][1] < C.MIN_TOP_SIM:
        return no_answer_message

    # コンテキスト組み立て
    context = build_context(top, corpus, C.MAX_CONTEXT_CHARS)

    # LLMに回答生成
    try:
        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n参考資料:\n{context}"},
            {"role": "user", "content": question}
        ]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 元コード準拠（CHAT_MODELとは別で運用）
            messages=messages,
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        # 言語別エラー文
        error_messages = {
            "ja": f"回答生成中にエラーが発生しました。😅\n{str(e)}",
            "en": f"An error occurred while generating the answer. 😅\n{str(e)}",
            "zh": f"生成回答时发生错误。😅\n{str(e)}",
            "ko": f"답변 생성 중 오류가 발생했습니다。😅\n{str(e)}",
            "es": f"Ocurrió un error al generar la respuesta。😅\n{str(e)}",
            "fr": f"Une erreur s'est produite lors de la génération de la réponse。😅\n{str(e)}",
            "de": f"Ein Fehler ist bei der Antwortgenerierung aufgetreten。😅\n{str(e)}"
        }
        return error_messages.get(language_code, error_messages["en"])