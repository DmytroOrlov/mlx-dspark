# Research: K2 IFM Protocol Parsing

## Decision: Extend the shared pair-based reasoning machinery

**Rationale**: `split_thinking()`, `prompt_opens_thinking()`, `ThinkingStreamSplitter`, and `MessageStream` already consume `_THINK_PAIRS`. They support self-opened reasoning, a matching closer generated for a prefilled opener, and truncated prefilled reasoning when the caller provides the prompt-derived closer hint. Add the three specified K2 pairs to that shared set; do not create a K2-specific endpoint path.

**Alternatives considered**: A K2-specific parsing/server branch was rejected because the shared pair abstraction already represents the required grammar and is consumed by the relevant APIs.

## Decision: Parse K2's default XML format using the existing call and schema seams

**Rationale**: `parse_tool_calls()` already turns detected model dialects into source-ordered calls serialized by `_as_openai()`. `schema_types()` and `_coerce_typed()` provide the existing schema-aware conversion path. The K2 parser should extract paired argument tags, pass raw values to that path, and retain declared strings verbatim, including multiline data.

**Alternatives considered**: A separate K2 argument type system or API-specific conversion was rejected because the existing schema/coercion and call serialization seams satisfy the requirements.

## Decision: Drop unclosed K2 syntax without partial call recovery

**Rationale**: Existing dialect parsers best-effort parse complete syntax and trim a dangling native opener from cleaned prose. Apply the same safety rule to K2: only complete call and key/value structures create call data; after completed calls are extracted, dangling K2 control syntax is removed from assistant content. No recovery of partial arguments is needed.

**Alternatives considered**: Synthesizing partial calls was rejected because it would invent argument values and is not required by the spec. Returning malformed control syntax as prose is disallowed by the spec.

## Decision: Add the K2 opener to the existing rolling streaming gate

**Rationale**: `_ToolGate` already holds a marker-length tail and scans markers across chunk boundaries, then buffers the remainder until completed calls are parsed. Adding the K2 opener to its marker set reuses this behavior without incremental XML emission.

**Alternatives considered**: A separate K2 stream parser or incremental XML-to-event emission was rejected as redundant and contrary to the established buffered design.

## Decision: Keep optional K2 formats out of scope

**Rationale**: The requested repair requires the default XML tool syntax. No immediate request/API selector was found in the narrowly inspected integration paths, and the specification explicitly defers `json` and `xml_typed` unless such a selector exists.

**Alternatives considered**: Implementing formats solely because the template describes them would expand scope without a caller-facing selection mechanism.

## Decision: Validate API integration with one regression per distinct wire conversion

**Rationale**: Shared parsing reaches existing OpenAI Chat, Anthropic Messages, and Responses conversions, but each converts the shared result to a different wire representation or policy. One representative integration test per surface proves composition; exhaustive edge cases belong at direct parser/state-machine seams.

**Alternatives considered**: Duplicating all grammar and chunk-boundary cases across every API was rejected as redundant. Omitting all API-level checks was rejected because it would not verify endpoint conversion.
