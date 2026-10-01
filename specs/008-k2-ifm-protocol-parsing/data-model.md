# Data Model: K2 IFM Protocol Parsing

This feature introduces no persisted entities. These transient parsing values describe the boundary between generated text and existing API conversion.

## Reasoning Segment

- **Fields**: matching opener/closer pair; reasoning text; optional following answer text.
- **Source forms**: self-opened output, prompt-prefilled output where only the closer is generated, or prompt-prefilled output truncated before the closer.
- **Rules**: each opener is paired only with its own closer; a prefilled truncation is reasoning only when the caller supplies the prompt-derived open-state hint; control tags are not answer content.
- **Lifecycle**: raw generation → shared reasoning split/state machine → caller's established reasoning and answer policy.

## Native K2 Tool Call

- **Fields**: function name; ordered argument key/value pairs.
- **Source form**: one `<ifm|tool_call>` element inside the default `<ifm|tool_calls>` wrapper; zero or more paired argument key/value elements.
- **Rules**: a complete call with no argument pairs has an empty argument map; only complete key/value pairs are accepted; incomplete fragments do not create recovered calls.
- **Type interpretation**: schema-declared JSON types are applied by the existing schema/coercion behavior. Declared strings preserve raw content, including newlines.
- **Lifecycle**: buffered native text → shared tool parser → existing OpenAI-shaped function call → API-specific output conversion.

## Structured Tool Call

- **Fields**: generated call identifier, function name, and serialized JSON arguments using the existing internal representation.
- **Rules**: calls preserve source order; cleaned assistant content excludes completed and dangling K2 control syntax.
- **Lifecycle**: shared parser output → OpenAI Chat `tool_calls`, Anthropic `tool_use`, or Responses function call according to existing API policy.
