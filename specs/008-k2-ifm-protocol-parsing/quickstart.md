# Quickstart: K2 IFM Protocol Parsing

This guide is for implementation validation. It uses local unit and API fixtures and does not load or download a model.

## Prerequisites

- Repository development environment and installed test dependencies.
- No network access, checkpoint, or Apple Silicon device is required for automated checks.

## Focused validation

Run the protocol parser and state-machine regressions:

```sh
uv run pytest tests/test_anthropic.py tests/test_tools.py -q
```

Run API integration regressions:

```sh
uv run pytest tests/test_server.py tests/test_responses.py -q
```

The focused assertions should establish:

- all three reasoning pairs work for self-opened and prefilled output, including truncation and arbitrary streaming splits;
- K2 single and multiple calls preserve order, schema types, and multiline strings;
- incomplete tool syntax is removed without raising or leaking into visible content;
- split K2 tool openers trip the existing streaming gate;
- OpenAI Chat, Anthropic Messages, and Responses retain their established wire shapes and reasoning policies;
- representative Qwen, Gemma, Muse, and existing tool dialect cases remain green.

## Optional manual validation

After implementation, an Apple Silicon operator may serve the cached `mlx-community/K2-Horizon-3.7B-4bit` checkpoint and confirm that a reasoning answer exposes reasoning through the existing reasoning channel and displays only the final answer as ordinary content. This smoke test supplements, but does not replace, the offline regressions.
