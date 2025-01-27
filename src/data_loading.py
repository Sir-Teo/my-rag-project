"""
data_loading.py

Responsible for loading raw documents (from a local folder or from a Hugging Face dataset),
and splitting them into chunks.
"""

from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from typing import List, Optional
import datasets


# Example Hugging Face dataset: 'm-ric/huggingface_doc'
DEFAULT_DATASET_PATH = "m-ric/huggingface_doc"

# Markdown-based separators for chunking
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "\n\n",
    "\n",
    " ",
    ""
]


def load_documents_from_hf(dataset_path: str = DEFAULT_DATASET_PATH, split: str = "train"):
    """
    Loads documents from a Hugging Face dataset, returns them in LangChain's Document format.
    """
    ds = datasets.load_dataset(dataset_path, split=split)
    docs = [
        LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
        for doc in ds
    ]
    return docs


def split_documents(
    docs: List[LangchainDocument],
    chunk_size: int = 512,
    overlap_ratio: float = 0.1,
    tokenizer_name: str = "thenlper/gte-small"
) -> List[LangchainDocument]:
    """
    Split documents into smaller chunks, ensuring we don't exceed the embedding model's max seq length.

    :param docs: List of LangchainDocument objects
    :param chunk_size: Max tokens per chunk
    :param overlap_ratio: Overlap ratio between chunks
    :param tokenizer_name: Name of the HF tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * overlap_ratio),
        separators=MARKDOWN_SEPARATORS,
        add_start_index=True,
        strip_whitespace=True,
    )

    processed = []
    for doc in docs:
        processed += splitter.split_documents([doc])

    # Optionally remove duplicates
    unique_texts = {}
    final_docs = []
    for d in processed:
        if d.page_content not in unique_texts:
            unique_texts[d.page_content] = True
            final_docs.append(d)

    return final_docs
