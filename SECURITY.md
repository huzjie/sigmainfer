# Security Policy

## Reporting a Vulnerability

If you discover a security issue in SigmaInfer, please report it responsibly by
opening a private security advisory on GitHub. Do not open a public issue with
exploit details.

## Scope

SigmaInfer ships a reference implementation with randomly initialized weights
for demonstration and benchmarking. It is **not** a production language model.
When loading real Sigma checkpoints, always verify the provenance of the weight
files (SHA-256) before running them.

## Best practices

- Do not load untrusted checkpoints without verifying their hashes.
- Treat all generated text as untrusted output.
- Keep the inference service behind authentication if exposed to a network.
