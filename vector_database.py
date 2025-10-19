from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from config import CHROMA_DIR


def get_vector_database():
    """ 
    Initialize and return a persistent vector database instance.

    Returns
    ---------
        ChromaDocumentStore: Persistent document store for vector embeddings.
    """
    return ChromaDocumentStore(persist_path=CHROMA_DIR)