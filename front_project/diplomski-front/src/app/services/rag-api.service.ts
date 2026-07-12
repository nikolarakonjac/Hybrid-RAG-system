import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { RagAskResponse, RagIngestResponse, HealthResponse } from '../models/rag.models';

@Injectable({ providedIn: 'root' })
export class RagApiService {
  private readonly http = inject(HttpClient);
  private readonly apiBaseUrl = 'http://127.0.0.1:8000';

  ingestPdf(file: File): Observable<RagIngestResponse> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    formData.append('filename', file.name);

    return this.http.post<RagIngestResponse>(`${this.apiBaseUrl}/rag/ingest`, formData);
  }

  askQuestion(question: string): Observable<RagAskResponse> {
    return this.http.post<RagAskResponse>(`${this.apiBaseUrl}/rag/ask`, { question });
  }

  checkHealth(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.apiBaseUrl}/health`);
  }
}
