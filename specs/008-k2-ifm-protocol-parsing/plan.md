# Implementation Plan: K2 IFM Protocol Parsing

**Branch**: `008-k2-ifm-protocol-parsing` | **Date**: 2026-10-07 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/008-k2-ifm-protocol-parsing/spec.md`

## Summary

Extend the existing shared reasoning pairs and tool-dialect parsing to recognize K2's three IFM reasoning pairs and its default XML-like tool format. Extend the shared streaming tool gate to recognize the K2 call opener. Keep endpoint conversions and their existing policies; verify them with one narrow integration regression per API where required. Automated validation is local and model-free.

## Technical Context

**Language/Version**: Python 3.11+ (project runtime)
**Primary Dependencies**: Existing standard-library parser utilities and current mlx-dspark protocol modules; no dependency changes
**Storage**: N/A
**Testing**: Existing pytest unit and API test files; no model downloads required
**Target Platform**: mlx-dspark server protocol layer on supported project platforms
**Project Type**: Python library and HTTP server
**Performance Goals**: No performance target or tuning; preserve existing streaming behavior and bounded marker look-behind
**Constraints**: Preserve all supported dialects; syntax-based detection only; do not alter model loading, feature 007, decoding, lifecycle, or dependencies
**Scale/Scope**: Shared reasoning splitters, native tool parser, streaming tool gate, and minimal OpenAI Chat, Anthropic Messages, and Responses regressions

## Current Paths and Minimal Extension Points

### Reasoning

`server.py` uses `prompt_opens_thinking()` to detect a prompt-prefilled opener. Non-streaming chat and Responses paths pass that state to `split_thinking()`. OpenAI streaming uses `ThinkingStreamSplitter`; Anthropic Messages uses the pair-driven reasoning phase in `MessageStream`; Responses streaming uses the existing channel splitter and intentionally drops reasoning according to its current policy. All these paths share `_THINK_PAIRS` for self-open detection and matching closer selection. `prompt_opens_thinking()` already returns the matched closer for a prompt tail ending in a known opener. Therefore the minimal extension is adding exactly the three K2 pairs to the shared pair set and relying on those existing semantics; no K2-specific endpoint or effort mapping is needed.

Whole-text parsing handles self-opened pairs, prefilled output containing only a closer, and unterminated self-opened content. Prefilled truncation is only classifiable when the caller supplies the prompt hint (`in_thinking=True` / its matched closer), which existing non-streaming and streaming call paths already do. The streaming splitter buffers a possible opener until it is resolved and holds a closer-length tail while inside reasoning, so arbitrary token chunk splits are represented by the same state machine.

### Tools and Streaming Suppression

`tools.parse_tool_calls()` detects dialect markers, extracts each dialect's call tuples, removes recognized syntax from cleaned text, then serializes calls through `_as_openai()`. `schema_types()` builds a tool-name/argument-name to declared JSON-type mapping. `_coerce_typed()` already preserves declared strings verbatim and converts valid booleans, integers, numbers, arrays, and objects; the XML argument parser applies those rules. The K2 dialect should reuse that schema/coercion path, preserve argument text for string values (including multiline values), and append calls in source order to the same existing representation.

`anthropic_api._ToolGate` withholds a rolling tail of `_MAX_MARKER - 1` characters and trips on any marker in `_TOOL_MARKERS`, then buffers all later text for `parse_tool_calls()`. Add the K2 call opener to that marker set; this automatically provides split-marker recognition without a second stream parser. Existing OpenAI chat, Responses, and Anthropic streaming paths already use this gate before sending ordinary text.

For incomplete syntax, parse only complete K2 call elements and complete argument key/value pairs; do not synthesize partial calls. After extracting any complete calls, remove or truncate at the earliest remaining K2 control opener so dangling wrappers, calls, keys, or values are not returned as assistant prose. This should follow the parser's current best-effort behavior for truncated native syntax and must not raise on malformed/incomplete content. Completed calls can be retained even if an outer wrapper is truncated; no-argument calls are valid as an empty argument map.

### Formats

Only the default K2 XML-like tool format is in scope. No immediately present request/API selector for K2's optional `json` or `xml_typed` formats was identified in the inspected protocol paths; the feature input also makes them conditional on such a selector. Record both optional formats as out of scope and do not investigate further.

## Constitution Check

The project constitution governs the hybrid-target performance experiment and related work. This feature is a narrow protocol compatibility repair and does not change the experiment axis, target composition, performance claims, measurements, or benchmark controls. No constitution gate is implicated; scope is bounded to the existing protocol seam and preserves current behavior for other dialects. No complexity exception is required.

## API Integration Strategy

Keep implementation in shared parsing; do not duplicate K2 handling in endpoint code.

- **OpenAI Chat**: Add or adapt one server regression proving non-streaming output contains clean `content`, `reasoning_content`, and structured `tool_calls`. Existing OpenAI streaming tests exercise the shared channel splitter and tool gate; add one K2 streaming integration case if direct seam tests do not already prove endpoint composition.
- **Anthropic Messages**: Add one Messages endpoint regression proving enabled thinking is represented as a thinking block and a K2 call is a structured `tool_use`; use direct `MessageStream` tests for exhaustive chunk boundaries and gate behavior.
- **Responses**: Add or adapt one Responses endpoint regression proving K2 call syntax becomes a structured function call and no IFM markup appears in visible text. Reasoning remains intentionally omitted from visible output under Responses' existing policy; direct reasoning tests prove stripping.

These are needed because the APIs map shared parser results into distinct wire shapes. Exhaustive parser and state-machine boundary coverage stays in the existing unit/protocol test files. If a single existing HTTP fixture or test already proves the required output shape for a surface, extend it rather than add a duplicate test.

## Test-First Implementation Sequence

1. Add failing `tests/test_anthropic.py` coverage for all three reasoning pairs in whole-text self-open and prefilled-close forms, prompt opener matching, truncation with the prefilled hint, and closer-immediately-followed-by-answer behavior.
2. Add splitter tests that divide each opener and closer at every internal character boundary (or equivalently every possible two-chunk split), compare concatenated streaming partitions against whole-text partitions, and include one-character feeds. Retain existing Qwen, Gemma, and Muse reasoning tests as preservation controls.
3. Add failing `tests/test_tools.py` cases for K2 single/multiple calls, source order, multiple and empty arguments, all declared schema types, verbatim multiline declared strings, cleaned text, and incomplete/truncated syntax. Assert invalid or incomplete fragments do not crash, do not become prose, and do not create partially recovered calls.
4. Add `_ToolGate` tests in `tests/test_anthropic.py` with the K2 marker split at every internal boundary; prove no IFM markup is emitted and that ordinary pre-marker text is still released. Preserve marker coverage for existing dialects.
5. Add the smallest OpenAI Chat, Anthropic Messages, and Responses API integration regressions described above, extending their existing fixtures. Preserve representative existing tool dialect tests (Hermes, Gemma, ATEM, LFM, MiniCPM, and XML) as controls.
6. Implement the shared pair, K2 XML dialect and dangling-syntax handling, and tool-gate marker additions in the existing shared modules only. Re-run the focused protocol/API tests and the relevant existing dialect preservation tests; optional final validation is a real K2 smoke test on Apple Silicon.

## Expected File Changes

Production:

- `src/mlx_dspark/anthropic_api.py` — add K2 reasoning pairs and K2 tool-call opener marker; keep existing shared state machines.
- `src/mlx_dspark/tools.py` — parse default K2 XML call and paired argument tags, reuse schema coercion and OpenAI-shaped serialization, and strip dangling K2 control syntax safely.

Tests:

- `tests/test_anthropic.py` — shared reasoning splitter, prompt opener, `MessageStream`, `_ToolGate`, and Anthropic/OpenAI protocol regressions.
- `tests/test_tools.py` — default K2 XML parser and typed/incomplete input regressions.
- `tests/test_server.py` — only if needed for a single OpenAI Chat endpoint output-shape regression.
- `tests/test_responses.py` — only if needed for a single Responses endpoint output-shape regression.

No public API, endpoint, or dependency contract changes are expected, so no new `contracts/` artifact is needed. The OpenAI Chat HTTP path is covered by `tests/test_server.py`; Anthropic's existing endpoint fixture lives in `tests/test_anthropic.py`; Responses uses `tests/test_responses.py`.

## Project Structure

```text
specs/008-k2-ifm-protocol-parsing/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── checklists/requirements.md

src/mlx_dspark/
├── anthropic_api.py
└── tools.py

tests/
├── test_anthropic.py
├── test_tools.py
├── test_server.py       # only if API-shape integration needs this file
└── test_responses.py    # only if API-shape integration needs this file
```

**Structure Decision**: Extend the existing shared Python protocol modules and their current unit/API tests. No new production module, parser framework, endpoint, or dependency is warranted.

## Post-Design Constitution Check

No constitution conflict is introduced. The shared parser extension is minimal, keeps unrelated model/runtime behavior out of scope, makes no performance claims, and adds no extra architectural layer. No complexity tracking entry is needed.
