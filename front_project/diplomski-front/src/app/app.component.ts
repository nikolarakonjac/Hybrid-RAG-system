import { Component, ElementRef, ViewChild, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpErrorResponse } from '@angular/common/http';

import { ChatMessage, UploadedDocument } from './models/rag.models';
import { RagApiService } from './services/rag-api.service';

@Component({
  selector: 'app-root',
  imports: [FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent {
  private readonly ragApi = inject(RagApiService);

  @ViewChild('messagesContainer') messagesContainer?: ElementRef<HTMLElement>;
  @ViewChild('fileInput') fileInput?: ElementRef<HTMLInputElement>;

  messages: ChatMessage[] = [
    {
      role: 'assistant',
      content:
        'Upload a PDF, then ask questions about its content. Answers are generated from your uploaded documents.',
    },
  ];

  documents: UploadedDocument[] = [];
  question = '';
  uploading = false;
  asking = false;
  checkingHealth = false;
  errorMessage = '';
  healthStatus: 'unknown' | 'ok' | 'error' = 'unknown';
  healthMessage = '';

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) {
      return;
    }

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      this.errorMessage = 'Please select a PDF file.';
      input.value = '';
      return;
    }

    this.errorMessage = '';
    this.uploading = true;

    this.ragApi.ingestPdf(file).subscribe({
      next: (response) => {
        this.documents = [
          ...this.documents,
          {
            filename: response.filename,
            chunk_count: response.chunk_count,
            total_chars: response.total_chars,
          },
        ];
        this.messages = [
          ...this.messages,
          {
            role: 'system',
            content: `Uploaded "${response.filename}" (${response.chunk_count} chunks). You can ask questions now.`,
          },
        ];
        this.uploading = false;
        input.value = '';
        this.scrollToBottom();
      },
      error: (error: HttpErrorResponse) => {
        this.errorMessage = this.extractErrorMessage(error, 'Failed to upload PDF.');
        this.uploading = false;
        input.value = '';
      },
    });
  }

  sendQuestion(): void {
    const trimmed = this.question.trim();
    if (!trimmed || this.asking) {
      return;
    }

    if (this.documents.length === 0) {
      this.errorMessage = 'Upload at least one PDF before asking a question.';
      return;
    }

    this.errorMessage = '';
    this.asking = true;
    this.messages = [...this.messages, { role: 'user', content: trimmed }];
    this.question = '';
    this.scrollToBottom();

    this.ragApi.askQuestion(trimmed).subscribe({
      next: (response) => {
        this.messages = [...this.messages, { role: 'assistant', content: response.answer }];
        this.asking = false;
        this.scrollToBottom();
      },
      error: (error: HttpErrorResponse) => {
        this.messages = [
          ...this.messages,
          {
            role: 'assistant',
            content: this.extractErrorMessage(error, 'Something went wrong while generating an answer.'),
          },
        ];
        this.asking = false;
        this.scrollToBottom();
      },
    });
  }

  onKeyDown(event: KeyboardEvent): void {
    if (event.key !== 'Enter' || event.shiftKey) {
      return;
    }
    event.preventDefault();
    this.sendQuestion();
  }

  triggerFileUpload(): void {
    this.fileInput?.nativeElement.click();
  }

  checkBackendHealth(): void {
    this.checkingHealth = true;
    this.healthMessage = '';

    this.ragApi.checkHealth().subscribe({
      next: (response) => {
        if (response.status === 'ok') {
          this.healthStatus = 'ok';
          this.healthMessage = 'Backend is running';
        } else {
          this.healthStatus = 'error';
          this.healthMessage = `Unexpected status: ${response.status}`;
        }
        this.checkingHealth = false;
      },
      error: () => {
        this.healthStatus = 'error';
        this.healthMessage = 'Backend is not reachable';
        this.checkingHealth = false;
      },
    });
  }

  private scrollToBottom(): void {
    setTimeout(() => {
      const el = this.messagesContainer?.nativeElement;
      if (el) {
        el.scrollTop = el.scrollHeight;
      }
    });
  }

  private extractErrorMessage(error: HttpErrorResponse, fallback: string): string {
    const detail = error.error?.detail;
    if (typeof detail === 'string') {
      return detail;
    }
    if (Array.isArray(detail)) {
      return detail.map((item) => item.msg ?? String(item)).join(', ');
    }
    return fallback;
  }
}
