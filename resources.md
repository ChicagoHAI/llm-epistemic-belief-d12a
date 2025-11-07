# Resources and Initial Research Notes

## Workspace Check
- Existing directories (`artifacts/`, `logs/`, `notebooks/`, `results/`) are empty, so no user-provided datasets or notes were found.
- Fresh `pyproject.toml` created locally to keep this workspace isolated from `idea-explorer` as required.

## Literature & Conceptual References
1. **Belief Injection for Epistemic Control in Linguistic State Space** (Dumbrava, 2025, arXiv:2505.07693) – proposes proactive control mechanisms over agents' internal belief states via targeted linguistic fragments, highlighting practical strategies for steering epistemic content. Useful for framing epistemic belief manipulation in LLMs.
2. **The Realizability of Revision and Contraction Operators in Epistemic Spaces** (Sauerwald & Thimm, 2024, arXiv:2407.20918) – formal analysis of how belief revision dynamics operate in epistemic spaces; grounds our discussion of epistemic vs. non-epistemic belief representations and stability.
3. **TruthfulQA benchmark** (Lin et al., 2021; dataset card on HuggingFace `truthfulqa/truthful_qa`) – evaluates truthful knowledge vs. imitative falsehoods, giving precedents for measuring epistemic reliability in LLMs.
4. **Phi-3.5 Mini Instruct model card** (Microsoft, 2024) – documents a 3.8B-parameter instruction-tuned model with strong reasoning density that can run CPU-only, making it a practical open-weight LLM for local experiments when API keys are unavailable.

These references collectively motivate separating factual (epistemic) reasoning from preference/affective (non-epistemic) statements and offer candidate modeling baselines.

## Candidate Datasets / Benchmarks
| Dataset | Source | Size / Split | Relevance |
| --- | --- | --- | --- |
| `agentlans/fact-or-opinion` | HuggingFace (2025) | 10K–100K synthetic statements in 16 languages (`train.jsonl.zst`, `test.jsonl.zst`) labelled as `fact` vs `opinion`. | Directly maps to epistemic (fact) vs non-epistemic (opinion) beliefs; ideal for supervised evaluation of classification and calibration.
| `DKYoon/r1-triviaqa-epistemic` | HuggingFace (2025) | 1K validation examples derived from TriviaQA with `question`, `answers`, `model_think`, `prompt`. | Contains model reasoning traces tagged as "epistemic"; useful to probe whether LLMs shift behavior when questions demand factual commitments.
| `DKYoon/r1-nonambigqa-epistemic` | HuggingFace (2025) | Another 1K validation-only split built from non-ambiguous QA prompts. | Provides high-clarity factual questions for analyzing epistemic stance vs confidence.
| `truthfulqa/truthful_qa` | HuggingFace | 817 validation samples (generation + MC configs). | Established truthfulness benchmark; can serve as stretch goal to observe epistemic adherence.

If time limits prevent using every dataset, `agentlans/fact-or-opinion` will be prioritized for quantitative experiments, while the DKYoon splits support qualitative inspection of epistemic reasoning traces.

## Tooling & Models
- **Libraries**: `datasets`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `typer`/`click` for scripts, `tqdm` for progress, `requests` for API calls, `transformers` + `accelerate` for running open-weight LLMs (Phi-3.5 mini instruct) on CPU.
- **LLMs**: Preferring hosted APIs (OpenRouter GPT-4.1 / Claude 4.5 Sonnet) per instructions, but no API keys are currently exposed in the environment. Contingency is to run `microsoft/Phi-3.5-mini-instruct` locally to ensure experiments still use real LLM weights rather than simulations; document limitation if APIs remain inaccessible.

## Identified Gaps & Next Steps
- Need explicit operationalization of "non-epistemic belief" beyond simple opinions. Proposed approach: treat opinion/value/desire statements as non-epistemic using the fact-or-opinion dataset; complement with manually curated scenario set distinguishing desire/intention vs factual belief for qualitative probing.
- Lack of direct metric linking epistemic vs non-epistemic differentiation. Plan to measure: (a) classification accuracy (epistemic vs non) and macro-F1; (b) calibration (Brier score) based on model-reported confidence; (c) textual analysis of reasoning (lexical cues referencing evidence vs affect).
- Need to confirm feasibility of running Phi-3.5 locally (memory footprint) and ensure deterministic inference (set seeds, use greedy decoding at `temperature=0`).
- Download & preprocess datasets (decompress `.zst`, create balanced sample). Estimate sample size (~300 prompts) fitting API/time budget.

This research snapshot will guide the detailed planning phase next.
