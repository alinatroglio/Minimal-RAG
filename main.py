from fastapi import FastAPI, UploadFile
from haystack.dataclasses import Document
from vector_database import get_vector_database
from index_pipeline import get_index_pipeline
from rag_pipeline import get_rag_pipeline

app = FastAPI()
index_pipeline = get_index_pipeline()
rag_pipeline = get_rag_pipeline()

@app.post("/ingest")
async def ingest_document(file: UploadFile):
    """
    Ingest and store a document in the vector database.

    Parameters
    ------
        Uploaded text file to be indexed.

    Returns
    ------
        JSON response containing ingestion status and uploaded filename.
    """
    content = await file.read()
    document = Document(content=content.decode("utf-8"))
    index_pipeline.run({"splitter": {"documents": [document]}})
    vecot_db = get_vector_database()
    return {"status": "ingested", "filename": file}


@app.post("/query")
async def query(query: str):
    """
    Query the RAG pipeline to retrieve and generate an answer.

    Parameters
    ------
        Query containing user's question or search query
    Returns
    ------
        JSON response containing the original query and generated answer.
    """
    result = rag_pipeline.run({"text_embedder": {"text": query},
        "prompt_builder": {"query": query},
        "llm": {"generation_kwargs": {"max_tokens": 128, "temperature": 0.1}},
        "answer_builder": {"query": query},})
    return {"query": query, "answer": result["answer_builder"]["answers"][0]}



