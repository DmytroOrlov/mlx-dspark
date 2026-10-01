# Quickstart

1. Run `python specs/004-golden-prompt-panel/evidence/probes/prompt_panel.py --self-check`.
2. Run `python specs/004-golden-prompt-panel/evidence/probes/prompt_panel.py --preflight` and require PASS before any model inference.
3. Run the frozen screen (17 prompt observations plus midpoint/end P01 anchors) through the parent orchestrator; each observation starts a fresh child process and Engine.
4. Run only runner-selected certification/fallback repeats.
5. Run completeness validation and generate the per-prompt JSON/Markdown reports.

The physical screen is deliberately speculative-only. It never runs a serial target control.
