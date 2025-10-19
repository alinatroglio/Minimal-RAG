from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from config import LLM_PATH, PROMPT_TEMPLATE, EMBEDDING_MODEL
from vector_database import get_vector_database
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
import os

# Init global vector database
vector_db = get_vector_database()

def get_text_embedder():
    """
    Create and return a text embedder for sentence-level embeddings.

    Returns
    ------
        SentenceTransformersTextEmbedder: Configured text embedder instance.
    Raises
    ------
        RuntimeError: If the embedding model fails to load.
    """
    try: 
        return SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL)
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model '{EMBEDDING_MODEL}': {e}")


def get_embedding_retriever():
    """
    Create and return a retriever that fetches top-k (here 5) semantically relevant documents.

    Returns
    ------
        ChromaEmbeddingRetriever: Retriever instance connected to the vector store.
    """
    return ChromaEmbeddingRetriever(document_store=vector_db, top_k=5)

def get_prompt_builder():
    """
    Create and return a prompt builder using the predefined template in config.py.

    Returns
    ------
        PromptBuilder: Configured prompt builder instance.
    """
    return PromptBuilder(template=PROMPT_TEMPLATE)

def get_llm():
    """
    Initialize and return a local Llama.cpp language model generator.

    Returns
    ------
        LlamaCppGenerator: Configured language model instance.
    Raises
    ------
        FileNotFoundError: If the model file at `LLM_PATH` does not exist.
        RuntimeError: If LLM model call fails.
    """
    if not os.path.exists(LLM_PATH):
        raise FileNotFoundError(f"LLM model not found at {LLM_PATH}.")
    try:
        return(LlamaCppGenerator(model=LLM_PATH, n_ctx=2048, n_batch=128))
    except Exception as e:
        raise RuntimeError(f"Failed to get LLM model {e}")

def get_answer_builder():
    """
    Create and return an answer builder component.

    Returns
    ------
        AnswerBuilder: Configured answer builder instance.
    """
    return AnswerBuilder()

def get_rag_pipeline():
    """
    Assemble and return the complete Retrieval-Augmented Generation (RAG) pipeline.

    The pipeline stages:
        1. Text embedding of the user query
        2. Semantic retrieval from the vector store
        3. Prompt construction for the language model
        4. Response generation via the LLM from Llama.cpp
        5. Answer building and formatting

    Returns
    ------
        Pipeline: Full RAG pipeline instance.
    """
    rag_pipeline = Pipeline()
    text_embedder = get_text_embedder()
    embedding_retriever = get_embedding_retriever()
    prompt_builder = get_prompt_builder()
    llm = get_llm()
    answer_builder = get_answer_builder()
    rag_pipeline.add_component("text_embedder",text_embedder)
    rag_pipeline.add_component("retriever", embedding_retriever)
    rag_pipeline.add_component("prompt_builder",prompt_builder)
    rag_pipeline.add_component("llm", llm)
    rag_pipeline.add_component("answer_builder", answer_builder)

    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("retriever.documents", "answer_builder.documents")
    return rag_pipeline
