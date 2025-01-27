"""
indexing.py

Builds a FAISS index from a list of documents.
Performs similarity search on that index.
"""

from typing import List
from langchain.docstore.document import Document as LangchainDocument
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy


def build_faiss_index(docs: List[LangchainDocument], embedding_model, distance_strategy=DistanceStrategy.COSINE):
    """
    Given a list of documents and an embedding model, build a FAISS index.
    """
    vector_db = FAISS.from_documents(
        docs,
        embedding_model,
        distance_strategy=distance_strategy
    )
    return vector_db


def similarity_search(query: str, vector_db: FAISS, k: int = 5) -> List[LangchainDocument]:
    """
    Use the FAISS index to perform a similarity search for the given query.
    Returns the top k documents.
    """
    return vector_db.similarity_search(query=query, k=k)
