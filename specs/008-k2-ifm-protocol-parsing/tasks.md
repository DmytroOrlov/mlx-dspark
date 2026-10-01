---
description: "Implementation tasks for K2 IFM protocol parsing"
---

# Tasks: K2 IFM Protocol Parsing

**Input**: Design documents from `specs/008-k2-ifm-protocol-parsing/` (`spec.md`, `plan.md`)

**Scope**: Extend shared reasoning/tool parsing only. Production changes are limited to `src/mlx_dspark/anthropic_api.py` and `src/mlx_dspark/tools.py`; tests are limited to `tests/test_anthropic.py`, `tests/test_tools.py`, `tests/test_server.py`, and `tests/test_responses.py` as needed. Do not modify model routing/loading, feature 007, template/checkpoint files, optional K2 formats, decoding, runtime behavior, or dependencies.

## Phase 1: User Story 1 — Keep K2 reasoning out of answer content (Priority: P1)

**Goal**: Route all three K2 reasoning pairs to the existing reasoning channel for whole-text and streaming parsing, including prompt-prefilled and truncated outputs.

**Independent Test**: `tests/test_anthropic.py` verifies matching reasoning/answer partitions for parameterized pairs, prefilled hints, truncation, immediate answer suffixes, and chunk splits; existing Qwen, Gemma, and Muse cases remain preservation coverage.

- [X] T001 [US1] Add parameterized `tests/test_anthropic.py` regressions for self-opened and prompt-prefilled `<ifm|think>`, `<ifm|think_fast>`, and `<ifm|think_faster>` pairs, including matching closer selection, truncated prefilled reasoning, and closer immediately followed by answer; run these focused reasoning tests once and record the expected failure.
- [X] T002 [US1] Extend the existing `ThinkingStreamSplitter` cases in `tests/test_anthropic.py` to compare whole-text partitions against streaming with representative opener/closer splits, including split boundaries for each K2 pair and one-character feeds; retain existing dialect cases without duplicating them.
- [X] T003 [US1] Extend only the shared `_THINK_PAIRS` machinery in `src/mlx_dspark/anthropic_api.py` with the three K2 opener/closer pairs so existing prompt-prefilled, whole-text, `MessageStream`, and streaming splitter semantics classify K2 reasoning without changing effort mapping.

---

## Phase 2: User Story 2 — Return K2 default tool calls as structured calls (Priority: P1)

**Goal**: Parse default K2 XML tool syntax through the existing schema typing and OpenAI-shaped serialization path, and suppress incomplete/native syntax safely.

**Independent Test**: `tests/test_tools.py` covers K2 call parsing and cleaned text; `tests/test_anthropic.py` verifies `_ToolGate` suppresses a split K2 opener. Existing tool dialect tests remain preservation coverage.

- [X] T004 [US2] Add parameterized K2 XML regressions in `tests/test_tools.py` for a single call, multiple calls in source order, multiple arguments, valid no-argument calls, bool/int/number/array/object schema types, declared string and multiline-string preservation, and cleaned assistant text; include incomplete/truncated wrapper, call, key, and value fragments and run the focused new tool tests once to record the expected failure.
- [X] T005 [US2] Add the minimal K2 tool-opener regression to `_ToolGate` tests in `tests/test_anthropic.py`, including a marker split across chunks and ordinary text released before the marker; include this in one focused red run for the missing K2 tool dialect/gate.
- [X] T006 [US2] Implement default K2 XML parsing in `src/mlx_dspark/tools.py`: pair `arg_key`/`arg_value`, preserve source call order and declared strings, reuse `schema_types()`/`_coerce_typed()` and `_as_openai()`, accept empty arguments, retain only complete calls/pairs, and strip or truncate dangling K2 control syntax without raising or leaking it as prose.
- [X] T007 [US2] Add the K2 `<ifm|tool_calls>` opener to the shared marker set in `src/mlx_dspark/anthropic_api.py` so `_ToolGate` withholds split markers and routes buffered completed output through the existing parser; do not add partial-call recovery or a K2-specific stream parser.

---

## Phase 3: User Story 3 — Preserve shared behavior across APIs (Priority: P1)

**Goal**: Confirm shared parsing reaches each API's existing response shape without endpoint-specific K2 parsing.

**Independent Test**: Representative model-free regressions demonstrate clean OpenAI Chat content and reasoning plus structured tool calls, Anthropic thinking and `tool_use`, and Responses structured function calls with its existing reasoning policy.

- [X] T008 [US3] Extend existing API fixtures with the minimum representative K2 integration regressions: `tests/test_anthropic.py` for Anthropic Messages thinking plus structured `tool_use`, `tests/test_server.py` for OpenAI Chat clean `content`/`reasoning_content` plus structured `tool_calls`, and `tests/test_responses.py` for no visible IFM markup plus a structured function call while preserving Responses' current reasoning policy; reuse fixtures and do not repeat parser edge cases.
- [X] T009 [US3] Run the API test files changed by T008 and the focused shared parser/state-machine suite `uv run pytest -q tests/test_anthropic.py tests/test_tools.py`; fix only failures within the allowed production and test files.

---

## Phase 4: Polish & Cross-Cutting Validation

**Purpose**: Check the narrow change set and relevant combined protocol/API behavior.

- [X] T010 Run the narrow relevant combined protocol/API suite covering changed files `tests/test_anthropic.py`, `tests/test_tools.py`, and any of `tests/test_server.py` or `tests/test_responses.py` changed in T008; perform one final scope/test audit against `specs/008-k2-ifm-protocol-parsing/tasks.md` and the allowed paths, without adding another analysis phase.
- [ ] T011 [P] Optionally run a real K2 smoke test on Apple Silicon and record its non-blocking result in `specs/008-k2-ifm-protocol-parsing/tasks.md`; this task requires no model download for automated completion.

## Dependencies & Execution Order

### Phase Dependencies

- User Story 1 parser changes follow its focused regressions: T001–T002 before T003.
- User Story 2 parser and gate changes follow the focused regressions: T004–T005 before T006–T007. T006 and T007 touch separate production files and may proceed in parallel after both red runs.
- User Story 3 endpoint regressions depend on shared parsing implementation (T003, T006, T007); T009 follows T008 and runs after implementation is complete.
- T010 follows all required implementation and validation tasks. T011 is optional and non-blocking.

### User Story Dependencies

- **US1 (P1)**: Independent shared reasoning seam; first implementation increment.
- **US2 (P1)**: Independent shared tool parsing/gate seam; can follow or run alongside US1 after the scoped reasoning tests are in place.
- **US3 (P1)**: Depends on US1 and US2 because it validates their shared results through API-specific output shapes.

### Parallel Opportunities

- T001 and T004 edit separate test files and can be prepared in parallel.
- T006 and T007 edit separate production files and can be implemented in parallel after their red regressions.
- API integration cases in T008 touch distinct test files and may be split across contributors; each should reuse its existing fixture.

## Implementation Strategy

### MVP First

Complete US1 and US2 shared parser changes with their direct regressions, then US3's representative API checks. The core MVP is completed when T001–T009 pass; T010 records the final narrow audit and T011 remains optional.

### Validation Commands

- Focused protocol/state-machine suite: `uv run pytest -q tests/test_anthropic.py tests/test_tools.py`
- Run only API test files actually changed in T008, then the narrow relevant combined protocol/API suite in T010.

## Notes

- All tasks use the required checklist format and identify exact files.
- Existing Qwen, Gemma, Muse, Hermes, Gemma, ATEM, LFM, MiniCPM, and other supported dialect regressions are preservation coverage; do not duplicate them.
- No grammar research, reproduction re-proof, exhaustive API/tag/chunk Cartesian product, or second audit phase is included.
