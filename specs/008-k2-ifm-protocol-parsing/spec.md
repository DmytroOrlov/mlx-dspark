# Feature Specification: K2 IFM Protocol Parsing

**Feature Branch**: `008-k2-ifm-protocol-parsing`
**Created**: 2026-10-07
**Status**: Draft
**Input**: User description: Narrow compatibility repair for K2 Horizon reasoning and default XML tool-call parsing through mlx-dspark's shared protocol handling.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Keep K2 reasoning out of answer content (Priority: P1)

As a client consuming a K2 Horizon response, I receive reasoning in the existing reasoning channel and only the final answer in ordinary assistant content.

**Why this priority**: The accepted reproduction leaks reasoning into normal assistant content, changing what clients display as the answer.

**Independent Test**: Feed representative self-opened and prompt-prefilled K2 reasoning outputs through the shared non-streaming and streaming splitters, including truncation and arbitrary chunk boundaries.

**Acceptance Scenarios**:

1. **Given** a response containing any of the three K2 reasoning tag pairs, **When** it is split, **Then** text inside the pair is reasoning and text after its matching closer is answer content.
2. **Given** a generation prompt already ends with a K2 reasoning opener, **When** generation contains the matching closer and an answer, **Then** only the prefix is reasoning and only the suffix is answer content.
3. **Given** generation ends at its token limit while a prefetched K2 reasoning block is still open, **When** it is split, **Then** all generated text is reasoning and no text is classified as answer content.
4. **Given** an opener or closer is divided across arbitrary streaming chunks, **When** the stream is processed, **Then** the result matches whole-text parsing and no partial IFM tag appears in answer content.
5. **Given** high, medium, or low reasoning effort, **When** the existing template selects its corresponding prefilled opener, **Then** parsing recognizes the corresponding closer without changing effort mapping.

### User Story 2 - Return K2 default tool calls as structured calls (Priority: P1)

As a client supplying tools to a K2 Horizon request, I receive completed K2 tool calls in the existing structured tool-call format and do not see their native markup as assistant prose.

**Why this priority**: Tool invocation must remain actionable by API clients and native control syntax must not be shown as a user-facing answer.

**Independent Test**: Parse one-call and multiple-call default K2 XML outputs using representative schemas, then verify call names, arguments, JSON argument serialization, and cleaned assistant text.

**Acceptance Scenarios**:

1. **Given** a completed K2 `<ifm|tool_calls>` block with one call, **When** it is parsed, **Then** the function name and paired argument keys and values are returned as one structured call and the markup is removed from assistant text.
2. **Given** a completed K2 block with multiple calls, **When** it is parsed, **Then** each call is returned in generated order with its own arguments.
3. **Given** argument schemas declare booleans, integers, numbers, arrays, or objects, **When** raw K2 argument values are parsed, **Then** compatible values retain their declared JSON types; declared strings remain strings, including multiline values.
4. **Given** a K2 tool block is incomplete or truncated, **When** it is parsed, **Then** the request does not crash and incomplete native IFM markup is not exposed as assistant prose.
5. **Given** streaming output begins a K2 tool block, including an opener split across chunks, **When** the stream is emitted, **Then** native tool syntax is withheld until parsing can produce completed structured calls.

### User Story 3 - Preserve existing protocol behavior across APIs (Priority: P1)

As a client using OpenAI Chat Completions, Anthropic Messages, or Responses, I receive the existing API's reasoning and tool representation while K2 IFM syntax remains hidden from visible text.

**Why this priority**: Shared parsing must produce consistent behavior without creating protocol-specific K2 implementations or regressing other model dialects.

**Independent Test**: Exercise shared parser/state-machine seams directly and a minimal set of API-level integrations for the three existing API surfaces; retain representative Qwen, Gemma, Muse and existing tool-dialect regressions.

**Acceptance Scenarios**:

1. **Given** a K2 response through OpenAI chat non-streaming, **When** it includes reasoning and tools, **Then** content is clean, reasoning uses `reasoning_content`, and calls appear as structured `tool_calls`.
2. **Given** a K2 response through OpenAI chat streaming, **When** reasoning, answer, and tool markup cross chunk boundaries, **Then** reasoning and answer remain separated and IFM markup is not emitted as ordinary content.
3. **Given** K2 reasoning is enabled for an Anthropic Messages request, **When** a response contains reasoning and a tool call, **Then** reasoning is represented by thinking blocks and the call by a structured `tool_use` block.
4. **Given** a K2 response through the Responses API, **When** it contains reasoning or tool syntax, **Then** the API's existing reasoning policy is preserved, visible text is free of IFM markup, and tools follow the existing structured path.
5. **Given** existing Qwen, Gemma, Muse reasoning or supported tool-dialect outputs, **When** they are processed, **Then** their established channel splitting and tool parsing behavior remains unchanged.

## Edge Cases

- Any IFM opener or closer may be divided at any character boundary across streaming chunks.
- A generation may begin inside a prefilled K2 reasoning block and terminate without its closer.
- A completed reasoning block may be followed immediately by answer text with no whitespace.
- K2 calls may contain multiple argument key/value pairs and multiple calls in one wrapper.
- String argument values may contain whitespace and newlines; schema-declared string content must not be scalar-coerced or trimmed in a way that changes its value.
- An incomplete call may end within a wrapper, call name, argument key, or argument value.
- Existing parsers must continue to detect their own dialects without requiring a K2 model-name check.
- Optional `json` and `xml_typed` template formats are outside this feature unless repository inspection demonstrates a caller-selectable format that is already exposed.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The shared reasoning parser MUST recognize `<ifm|think>…</ifm|think>`, `<ifm|think_fast>…</ifm|think_fast>`, and `<ifm|think_faster>…</ifm|think_faster>` as reasoning blocks.
- **FR-002**: Reasoning parsing MUST split both self-opened blocks and blocks whose opener was prefixed to the prompt, including outputs containing only the corresponding closer followed by answer text.
- **FR-003**: When generation begins inside a prompt-prefilled K2 reasoning block and is truncated before its closer, all generated text MUST remain reasoning and MUST NOT become answer content.
- **FR-004**: Streaming reasoning parsing MUST correctly handle opener and closer tokens split across arbitrary chunks, without exposing partial or complete IFM reasoning markup in answer content.
- **FR-005**: The three existing reasoning-effort levels MUST continue to select their corresponding template-prefilled K2 opener; this feature MUST NOT redefine effort mapping.
- **FR-006**: Reasoning and tool parsing MUST remain model-agnostic and MUST NOT require a K2 model-name branch when syntax detection is sufficient.
- **FR-007**: Tool parsing MUST recognize K2's default XML structure using `<ifm|tool_calls>`, `<ifm|tool_call>`, `<ifm|arg_key>`, and `<ifm|arg_value>` tags, supporting one or multiple calls.
- **FR-008**: Each parsed K2 call MUST preserve the function name and pair argument names with their corresponding values in the existing OpenAI-shaped internal tool-call representation.
- **FR-009**: Parsed K2 tool markup MUST be removed from cleaned assistant text. Incomplete or truncated K2 tool syntax MUST NOT crash request processing or leak as assistant prose.
- **FR-010**: K2 argument parsing MUST use available tool schema types so valid boolean, integer, number, array, and object values retain their declared types; string values remain strings and multiline string values remain uncorrupted.
- **FR-011**: Streaming tool suppression MUST recognize the K2 tool-call opener early enough to withhold native markup, including when the opener is split across generation chunks. Completed calls continue to be emitted through the existing completed-call parsing path; incremental native XML/JSON emission is not required.
- **FR-012**: The shared repair MUST flow through existing OpenAI chat, Anthropic Messages, and Responses API behavior, preserving each API's established reasoning and structured-tool policies.
- **FR-013**: Existing Qwen, Gemma, Muse reasoning behavior and existing Hermes, Gemma, ATEM, LFM, MiniCPM, and other supported tool dialects MUST remain unchanged.
- **FR-014**: Regression coverage MUST directly exercise shared parser/state-machine behavior for all three reasoning pairs, self-opened and prefilled reasoning, truncation, arbitrary chunk boundaries, default K2 XML single and multiple calls, typed values, split tool markers, and incomplete calls, plus representative existing dialect cases and minimal API integration paths.
- **FR-015**: Automated regression coverage MUST NOT require network access or model downloads. A live K2 smoke test on Apple Silicon MAY be performed as optional manual validation after implementation.
- **FR-016**: This feature MUST NOT alter model loading, remote-code routing, model weights or tokenizer/checkpoint files, speculative decoding, lookup, performance tuning, battery pause behavior, caching, memory management, or server lifecycle.

### Key Entities *(include if feature involves data)*

- **Reasoning block**: Generated text bounded by a matching K2 IFM opener and closer, or by an unterminated prefilled block at generation end.
- **Native tool call**: A function name and paired argument names and values encoded in the default K2 XML-like syntax.
- **Structured tool call**: The existing API-neutral internal function-call representation, with arguments serialized for downstream API conversion.
- **Tool schema types**: Caller-supplied argument declarations used to interpret raw XML-like values as JSON types.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All three K2 reasoning pairs produce the expected reasoning and answer partitions in self-opened, prompt-prefilled, and truncated cases.
- **SC-002**: Streaming results for K2 reasoning match whole-text results across tested arbitrary chunk boundaries, with zero IFM reasoning markup in answer content.
- **SC-003**: A completed K2 tool block with one or multiple calls yields correctly ordered structured calls and zero native K2 tool markup in cleaned assistant text.
- **SC-004**: Schema-declared K2 argument values preserve their declared JSON types when valid, and string values including multiline values preserve their contents.
- **SC-005**: Split or incomplete K2 tool markup produces neither a request crash nor raw IFM markup in streamed ordinary text or cleaned assistant content.
- **SC-006**: OpenAI chat, Anthropic Messages, and Responses API regressions demonstrate their established response shapes for K2 reasoning and tools without visible IFM control markup.
- **SC-007**: Representative existing Qwen, Gemma, Muse reasoning and supported tool-dialect regressions remain green, with no requirement for model downloads.

## Assumptions

- The cached K2 checkpoint's `chat_template.jinja` grammar supplied in the feature description is authoritative for the three reasoning pairs and default XML tool format.
- The accepted observed reasoning leak is sufficient reproduction evidence; implementation planning does not need to re-prove it.
- The parser and streaming suppression seams described in the feature input are the existing shared behavioral boundaries; implementation should extend them without introducing separate protocol implementations.
- Only the default `xml` tool format is required. The optional `json` and `xml_typed` template formats remain out of scope unless repository inspection establishes an existing caller selection mechanism.
- Existing API policies determine whether reasoning is exposed, while shared parsing ensures IFM markup is not mistaken for visible answer text.
- No live model run is required for implementation acceptance; an Apple Silicon smoke test is optional manual evidence.
