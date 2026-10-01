# Tasks: H3 Balanced Serve Wiring

**Input**: Revised design documents from `specs/006-h3-balanced-serve-wiring/`

**Prerequisites**: Revised `spec.md`, `plan.md`, `research.md`, `data-model.md`, and `contracts/serve-cli.md`

**Organization**: Tasks follow the three P1 user stories. No setup or foundational work is needed; the feature reuses the existing CLI and composition seam.

## Phase 1: User Story 3 - Validate Donor Selection Before Loading (Priority: P1)

**Goal**: Parse the small block-selection grammar and reject invalid or one-sided input before resolution or loading.

**Independent Test**: Model-free tests accept and normalize the specified grammar and prove malformed, out-of-range, descending, empty-element, and one-sided inputs stop before resolver, `Engine.load`, or model loading.

- [X] T001 [US3] Add focused model-free donor-block parser tests in `tests/test_cli_pause.py` covering `62` → `(62,)`, `56-63` → `(56,57,58,59,60,61,62,63)`, `12,28,41,55,62`, `12,20-23,62`, and `62,62,60-62` → `(60,61,62)`, plus integer-only syntax, bounds 0–63, ascending inclusive ranges, empty elements, malformed syntax, sorted deduplication, and rejection behavior.
- [X] T002 [US3] Implement the compact donor-block parser and normalized selection formatter in `src/mlx_dspark/cli.py`, enforcing the grammar and semantics covered by T001 without introducing a general composition language.
- [X] T003 [US3] Add model-free paired-option dispatch tests in `tests/test_cli_pause.py` proving exactly one of `--donor-model` and `--donor-blocks`, malformed selections, out-of-range indices, and descending ranges fail before resolver, `Engine.load`, or model loading.
- [X] T004 [US3] Add `--donor-model <repo-or-path>` and `--donor-blocks <block-selection>` to the serve parser in `src/mlx_dspark/cli.py`; enforce both-or-neither semantics and parse/validate selections before any model resolution.

---

## Phase 2: User Story 2 - Preserve Ordinary Serve (Priority: P1)

**Goal**: Keep the no-donor invocation on its existing loading path and preserve its kwargs exactly.

**Independent Test**: A model-free serve dispatch check confirms the ordinary target/mode/drafter values and existing load kwargs are preserved with no composition request.

- [X] T005 [US2] Add a model-free no-donor serve dispatch regression test in `tests/test_cli_pause.py` that asserts the current resolver/load path and kwargs remain unchanged and no target composition is requested.

---

## Phase 3: User Story 1 - Serve with Donor Blocks (Priority: P1) 🎯 MVP

**Goal**: Wire paired donor options through the existing immutable snapshot resolution and `TargetCompositionRequest` handoff; report the normalized selection and Qwen ownership.

**Independent Test**: Model-free dispatch checks prove H3 expands exactly to blocks 56–63, explicit drafter selection is independent, the existing request reaches `Engine.load(target_composition=...)`, and startup reports donor identity/revision when available, selection, and Qwen-owned components.

- [X] T006 [US1] Add model-free donor dispatch tests in `tests/test_cli_pause.py` proving `56-63` expands to `(56,57,58,59,60,61,62,63)`, explicit DFlash drafter selection is unchanged, and the existing composition request is passed to `Engine.load`.
- [X] T007 [US1] In `src/mlx_dspark/cli.py`, resolve target and donor using the established `_resolve`/snapshot path, obtain immutable identities only from resolved paths, construct the existing `TargetCompositionRequest`, and conditionally pass it through `Engine.load(target_composition=...)`; preserve existing immutable path/revision validation and fail closed when a local path lacks the required immutable revision.
- [X] T008 [US1] Add concise composition startup reporting in `src/mlx_dspark/cli.py` for donor identifier, resolved donor revision when available, normalized block selection, and Qwen ownership of embedding, every unselected block, final norm, and LM head; run the focused model-free tests in `tests/test_cli_pause.py` and `tests/test_hybrid_target.py`, then optionally perform only the single documented H3 startup/request smoke from `specs/006-h3-balanced-serve-wiring/quickstart.md` after those tests pass.

## Dependencies & Execution Order

### Phase Dependencies

- User Story 3 tasks T001–T004 establish parser semantics and early validation; write tests before their implementation.
- User Story 2 task T005 protects the existing omitted-option behavior and must pass alongside the donor dispatch cases.
- User Story 1 tasks T006–T008 depend on paired arguments and parser validation from T001–T004. Add dispatch assertions before wiring/reporting.
- T008 runs the focused model-free tests after implementation. Its hardware smoke is optional and only follows passing model-free tests.

### User Story Dependencies

- **User Story 3 (P1)**: First; supplies the validated paired interface required by the composition path.
- **User Story 2 (P1)**: Independent behavior guard; must remain valid throughout wiring.
- **User Story 1 (P1)**: Depends on the paired arguments and validated normalized selection from User Story 3.

### Parallel Opportunities

- No implementation tasks need parallel execution: parser, argument validation, and request wiring touch the same CLI file.
- Focused test cases may be designed independently, but consolidate edits to `tests/test_cli_pause.py` to avoid conflicts.

## Implementation Strategy

Deliver the MVP by completing the parser/early-validation path, protecting ordinary serve behavior, then wiring the donor request and startup report. The expected production diff is limited to `src/mlx_dspark/cli.py`; focused test changes belong in existing CLI/hybrid test files. Do not change `server.py`, `hybrid_target.py`, or `load.py` unless the existing request seam proves the revised plan factually wrong; stop and report that blocker rather than broadening scope. The final human command is the H3 serve invocation in `specs/006-h3-balanced-serve-wiring/quickstart.md` using paired `--donor-model` and `--donor-blocks 56-63`, with no preset selector.

## Notes

- No task introduces `--target-composition h3-balanced`, a general composition DSL, new resolver/loader/server branch, or changes outside the authorized scope.
- No tasks cover Features 002/003/005 re-validation, performance or quality evaluation, evidence frameworks, 64-layer sweeps, sparse-combination research, tokenizer/deprecation fixes, or speculative decoding/controller/cache changes.
- All tasks use checklist syntax, sequential IDs, story labels, and explicit file paths.
