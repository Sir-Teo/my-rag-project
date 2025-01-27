"""
main.py

Command-line entry point to run the entire RAG pipeline end-to-end.
Example usage:
  python main.py \
      --question "How to create a pipeline object?" \
      --use_reranker
"""

import argparse
import torch
import sys

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Local imports from src/
from src.data_loading import load_documents_from_hf, split_documents
from src.embeddings import get_embedding_model
from src.indexing import build_faiss_index
from src.reranking import get_reranker
from src.rag_pipeline import answer_with_rag


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default="How to create a pipeline object?")
    parser.add_argument("--dataset_path", type=str, default="m-ric/huggingface_doc")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--embedding_model_name", type=str, default="thenlper/gte-small")
    parser.add_argument("--llm_model_name", type=str, default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--use_reranker", action="store_true")
    parser.add_argument("--reranker_model_name", type=str, default="colbert-ir/colbertv2.0")
    parser.add_argument("--num_retrieved_docs", type=int, default=10)
    parser.add_argument("--num_docs_final", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load data
    print("=> Loading dataset...")
    raw_docs = load_documents_from_hf(dataset_path=args.dataset_path, split=args.split)

    # 2. Split into chunks
    print("=> Splitting documents into chunks...")
    docs_processed = split_documents(raw_docs, chunk_size=512, overlap_ratio=0.1, tokenizer_name=args.embedding_model_name)

    # 3. Build embeddings
    print("=> Building embeddings...")
    embedding_model = get_embedding_model(
        model_name=args.embedding_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        normalize_embeddings=True
    )

    # 4. Build index
    print("=> Building FAISS index...")
    knowledge_db = build_faiss_index(docs_processed, embedding_model)

    # 5. Load optional reranker
    reranker = None
    if args.use_reranker:
        print("=> Loading reranker...")
        reranker = get_reranker(args.reranker_model_name)

    # 6. Load LLM
    print("=> Loading LLM reader...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(args.llm_model_name, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
    reader_llm = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )

    # 7. Run RAG pipeline
    print(f"=> Querying: {args.question}")
    answer, used_docs = answer_with_rag(
        question=args.question,
        llm=reader_llm,
        knowledge_index=knowledge_db,
        use_reranker=args.use_reranker,
        reranker=reranker,
        num_retrieved_docs=args.num_retrieved_docs,
        num_docs_final=args.num_docs_final
    )

    # 8. Print result
    print("\n====================== RAG Answer ======================")
    print(answer)
    print("========================================================")

    # If you want to see the final doc chunks used:
    # for i, doc in enumerate(used_docs):
    #     print(f"[Doc {i}]: {doc}")


if __name__ == "__main__":
    main()
