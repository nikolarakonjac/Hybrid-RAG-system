from contextlib import asynccontextmanager
from typing import Annotated, Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from embedding_model import EmbeddingModel
from pdf_utils import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    PdfExtractError,
    extract_text_from_pdf,
    split_into_chunks,
)
from rag import ingest_pdf, query_rag
from reranker import CrossEncoderReranker
from vector_store import ChromaVectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = ChromaVectorStore()
    app.state.embedder = EmbeddingModel()
    app.state.reranker = CrossEncoderReranker()
    yield


app = FastAPI(title="Diplomski backend", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/pdf/chunks")
async def pdf_to_chunks(
    file: Annotated[UploadFile, File(description="PDF document")],
) -> dict[str, Any]:
    """Extract text and chunk (debug); no embedding or storage."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Expected a file with .pdf extension",
        )
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        text = extract_text_from_pdf(raw)
    except PdfExtractError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    chunks = split_into_chunks(
        text, size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP
    )
    return {
        "filename": file.filename,
        "total_chars": len(text),
        "chunk_count": len(chunks),
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "chunks": chunks,
    }


class RagAskBody(BaseModel):
    question: str = Field(..., min_length=1)


@app.post("/rag/ingest")
async def rag_ingest(
    file: Annotated[UploadFile, File(description="PDF bytes")],
    filename: Annotated[str, Form(description="Display name for this document (e.g. report.pdf)")],
) -> dict[str, Any]:
    """
    One PDF per request: extract text, chunk, embed, append to the in-memory vector DB.
    Call again with another PDF to grow the corpus (simple cumulative RAG).
    """
    name = filename.strip()
    if not name:
        raise HTTPException(status_code=400, detail="filename must be non-empty")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    store: ChromaVectorStore = app.state.store
    embedder: EmbeddingModel = app.state.embedder
    try:
        result = ingest_pdf(
            store,
            embedder,
            raw,
            filename=name,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        )
    except PdfExtractError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return {
        "filename": result.filename,
        "chunk_count": result.chunk_count,
        "total_chars": result.total_chars,
        "embedding_dim": result.embedding_dim,
    }


@app.post("/rag/ask")
async def rag_ask(body: RagAskBody) -> dict[str, Any]:
    """
    Answer using all ingested chunks: vector retrieve → cross-encoder rerank → Ollama.
    Body: ``question`` only.
    """
    store: ChromaVectorStore = app.state.store
    embedder: EmbeddingModel = app.state.embedder
    reranker: CrossEncoderReranker = app.state.reranker

    try:
        result = await query_rag(
            store,
            embedder,
            reranker,
            body.question,
            ollama_model=None,
        )
    except httpx.ReadTimeout as e:
        raise HTTPException(
            status_code=504,
            detail=(
                "Ollama did not finish within the HTTP read timeout. "
                "Set OLLAMA_TIMEOUT to a higher value (seconds), use a smaller/faster model, "
                "or ensure Ollama is not stuck loading the model."
            ),
        ) from e
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned HTTP {e.response.status_code}: {e.response.text[:500]}",
        ) from e
    except httpx.RequestError as e:
        msg = str(e)
        if getattr(e, "request", None) is not None:
            msg = f"{e.request.url}: {e}"
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach Ollama ({msg})",
        ) from e

    return {
        "answer": result.answer,
        "context_used": [
            {
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
                "source": c.source,
                "retrieval_score": c.retrieval_score,
                "rerank_score": c.rerank_score,
                "text": c.text,
            }
            for c in result.context_used
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
