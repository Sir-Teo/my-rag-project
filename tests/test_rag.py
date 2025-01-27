"""
test_rag.py

A minimal test to verify that your RAG pipeline doesn't crash.
"""

import unittest
import torch

from src.data_loading import load_documents_from_hf, split_documents
from src.embeddings import get_embedding_model
from src.indexing import build_faiss_index
from src.rag_pipeline import answer_with_rag
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class TestRAGPipeline(unittest.TestCase):
    def test_rag_end_to_end(self):
        # 1. Load a *small* subset of data (for testing)
        raw_docs = load_documents_from_hf(split="train")[:50]

        # 2. Split
        docs_processed = split_documents(raw_docs, chunk_size=256, overlap_ratio=0.1)

        # 3. Embedding model (CPU for testing)
        embedding_model = get_embedding_model(
            model_name="thenlper/gte-small",
            device="cpu",
            normalize_embeddings=True,
            multi_process=False
        )

        # 4. Index
        knowledge_db = build_faiss_index(docs_processed, embedding_model)

        # 5. LLM (You might want a small or CPU-friendly model for testing)
        bnb_config = BitsAndBytesConfig(load_in_4bit=False)  # disable for CPU test
        model_name = "HuggingFaceH4/zephyr-7b-beta"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        reader_llm = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            do_sample=False,  # disable sampling for test consistency
            max_new_tokens=50,
        )

        # 6. Pipeline test
        question = "How to create a pipeline object?"
        answer, used_docs = answer_with_rag(
            question=question,
            llm=reader_llm,
            knowledge_index=knowledge_db,
            use_reranker=False,
            num_retrieved_docs=5,
            num_docs_final=2
        )

        print("\nTest answer:", answer)
        self.assertTrue(len(answer.strip()) > 0, "Answer should not be empty")


if __name__ == "__main__":
    unittest.main()
