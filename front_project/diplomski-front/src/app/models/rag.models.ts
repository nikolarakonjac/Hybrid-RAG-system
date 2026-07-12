export interface RagIngestResponse {
  filename: string;
  chunk_count: number;
  total_chars: number;
  embedding_dim: number;
}

export interface RagAskResponse {
  answer: string;
  context_used: RagContextChunk[];
}

export interface RagContextChunk {
  chunk_id: string;
  chunk_index: number;
  source: string;
  retrieval_score: number;
  rerank_score: number;
  text: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface UploadedDocument {
  filename: string;
  chunk_count: number;
  total_chars: number;
}

export interface HealthResponse {
  status: string;
}
