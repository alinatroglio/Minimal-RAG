CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
LLM_PATH = "model/openchat-3.5-1210.Q3_K_S.gguf"
PROMPT_TEMPLATE = """You are AI assistant. Use the following documents to answer the query.
    {% for doc in documents %}
        - {{ doc.content }}
    {% endfor %}
    Question: {{query}}
    Answer:
    """