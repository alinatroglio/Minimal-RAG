
from haystack import Pipeline
from config import EMBEDDING_MODEL
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from vector_database import get_vector_database
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter

# Init global vector_db
vector_db = get_vector_database()

def get_splitter():
    """
    Create and return a text splitter for chunking documents.

    Splitting parameters:
      - split_length: Maximum number of words per chunk
      - split_overlap: Number of overlapping words between chunks
      - split_by: Split unit by word

    Returns
    -------
        DocumentSplitter: Configured splitter instance for document segmentation.
    """
    return DocumentSplitter(split_length=200, split_overlap=50, split_by="word")

def get_embedder():
    """
    Create and return a sentence transformer embedder to transform chunks into dense vector representation.

    Returns
    -------
        SentenceTransformersDocumentEmbedder: Configured embedder instance
    """
    return SentenceTransformersDocumentEmbedder(model=EMBEDDING_MODEL)

def get_document_writer():
    """
    Create and return a document writer that persists documents to the vector store.

    Returns
    -------
        DocumentWriter: Configured writer instance for vector database storage.
    """
    return DocumentWriter(document_store=vector_db)

def get_index_pipeline():
    """
    Assemble and return the complete indexing pipeline.

    The pipeline processes raw text in three steps:
        1. Split text into chunks (splitter)
        2. Generate embeddings (embedder)
        3. Store embeddings in the vector database (writer)
    Returns
    -------
        Pipeline: A fully configured indexing pipeline ready for document ingestion.
    """
    index_pipeline = Pipeline()
    text_splitter = get_splitter()
    embedder = get_embedder()
    document_writer = get_document_writer()
    index_pipeline.add_component("splitter", text_splitter)
    index_pipeline.add_component("embedder",embedder)
    index_pipeline.add_component("writer", document_writer )
    index_pipeline.connect("splitter", "embedder")
    index_pipeline.connect("embedder", "writer")
    return index_pipeline