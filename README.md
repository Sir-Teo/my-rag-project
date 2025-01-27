# My RAG Project

This repository demonstrates an advanced Retrieval Augmented Generation (RAG) system using:

- [LangChain](https://github.com/hwchase17/langchain) for retrieval + LLM orchestration
- [FAISS](https://github.com/facebookresearch/faiss) for vector indexing
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the LLM & embeddings
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) for embedding models
- [ragatouille](https://github.com/huggingface-community/ragatouille) for optional reranking with cross-encoders
- [gradio]()

## Quickstart

1. **Clone** the repository:

```bash
   git clone https://github.com/<YOUR_USERNAME>/my-rag-project.git
   cd my-rag-project
```
2. Install the dependencies:

```bash 
pip install -r requirements.txt
```

3. Run the Main Script:

```bash
python main.py \
    --question "How to create a pipeline object?" \
    --use_reranker
```
The above command retrieves relevant documents, optionally reranks them, and uses an LLM to produce a final answer.