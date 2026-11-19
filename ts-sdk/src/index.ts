export interface GenerateRequest {
  prompt: string;
  max_new_tokens?: number;
  block_size?: number;
  num_steps?: number;
  sampler?: string;
  seed?: number;
}

export interface GenerateResponse {
  output: string;
  stats: Record<string, unknown>;
}

export class SigmaClient {
  constructor(private baseUrl: string = "http://127.0.0.1:8000") {}

  async generate(req: GenerateRequest): Promise<GenerateResponse> {
    const res = await fetch(`${this.baseUrl}/v1/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    return res.json();
  }

  async health(): Promise<{ status: string }> {
    const res = await fetch(`${this.baseUrl}/health`);
    return res.json();
  }
}
