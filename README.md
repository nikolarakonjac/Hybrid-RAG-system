A local question-answering system over PDF documents, based on Retrieval-Augmented Generation (RAG).
Users upload their own files and ask questions in natural language.
The system does not answer from the model’s general knowledge, but from passages retrieved from those documents.

Large language models generate fluent text, yet they have no access to a private corpus and their knowledge is frozen at training time.
This project addresses that gap with a fully local pipeline: documents stay on the user’s machine, and the model receives only a short, relevant context.

Pipeline:
1) Each PDF is parsed, split into overlapping chunks, and indexed twice — semantically (dense vectors) and lexically (BM25).
2) For a question, both retrievers return candidates; the lists are merged with Reciprocal Rank Fusion and reranked by a cross-encoder.
3) If confidence scores are too low, the system refuses to answer rather than guess.
4) A local LLM (Ollama) then writes the answer using only the selected passages.


The stack is a FastAPI backend, an Angular UI, Chroma for the vector index, and Ollama for generation. The index is in-memory for the lifetime of the service. Documents are never sent to external APIs.
