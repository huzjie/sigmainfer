# Contributing to SigmaInfer

Thank you for your interest in contributing.

## How to contribute

1. **Report a bug** — open an issue with a minimal reproduction.
2. **Propose a feature** — describe the problem and your proposed approach.
3. **Send a pull request** — keep changes small and focused.

## Adding a new sampler

Samplers live in `sigmainfer/core/sampler.py`. Implement a subclass of
`BaseSampler`, override `sample(logits, mask, step) -> tuple[ids, confidence]`,
then register it in `sigmainfer/core/registry.py`. Add a unit test under
`tests/`.

## Adding a new mask schedule

Schedules live in `sigmainfer/core/schedule.py`. Implement `ratio(step, total)`
returning the fraction of positions to keep masked, then register it in the
`SCHEDULES` registry.

## Code style

Follow PEP 8. The core package is intentionally zero-dependency (Python
standard library only); optional features may declare extras in
`pyproject.toml`.
