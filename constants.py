import re

# ==== 固定設定 =====================================================
CSV_FILE = "dance_school.csv"          # 固定の参考資料CSV
ENCODING = "utf-8"
INDEX_COLS = ["name", "url", "about","price","reserve","youtube","instagram","schedule","instructor","access","tel","mail","open"]
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
DETECT_MODEL = "gpt-3.5-turbo"  # 言語判定用（元コード準拠）

TOP_K = 6
MIN_TOP_SIM = 0.20
USE_HISTORY_TURNS = 2
MAX_CONTEXT_CHARS = 4500
STREAM_ANSWER = True   # （現実装は非ストリーミングのanswer_pipelineを使用）

# 文分割用の正規表現（コンパイル済みを共有）
SENT_SPLIT_RE = re.compile(r"(?<=[。．.!?！？])\s+|\n+")