# LLM Epistemic vs Non-Epistemic Belief Study

## Overview
This workspace investigates whether language models distinguish factual (epistemic) beliefs from opinions/preferences using the `agentlans/fact-or-opinion` dataset and qualitative DKYoon epistemic QA traces. Baseline models succeed, but the tested open-weight LLM does not.

## Key Findings
- TF-IDF + logistic regression achieves **94.1% accuracy / 0.94 macro-F1 / 0.081 Brier** on the English fact-vs-opinion test split.
- `Qwen/Qwen2.5-0.5B-Instruct` mislabeled most opinions as facts, yielding **50% accuracy** on 60 sampled statements and a poor **0.446 Brier**.
- The LLM’s reported confidence stayed ≥0.90 for both ground-truth labels (Welch’s t-test p=0.003), indicating weak differentiation between belief types.
- Rationale keyword analysis showed epistemic language even for human-labeled opinions (chi-square p=0.22), reinforcing the collapse toward factual framing.

## Reproduction Steps
```bash
# 1. Install dependencies
uv sync

# 2. Activate environment
source .venv/bin/activate

# 3. Prepare datasets
datasets cache will download automatically
python scripts/prep_fact_opinion.py

# 4. Run experiments (logistic baseline + LLM)
PYTHONPATH=src python scripts/run_experiments.py baseline
PYTHONPATH=src python scripts/run_experiments.py llm --limit 60 --max-new-tokens 60

# 5. Run statistical analysis and plots
PYTHONPATH=src python scripts/analyze_results.py
```
> Note: LLM inference on CPU for 60 samples, max 60 generated tokens, takes ≈9 minutes. Ensure enough RAM for `Qwen2.5-0.5B` (~2.7 GB).

## File Structure
- `resources.md`, `planning.md`: research context and plan.  
- `data/processed/`: prepared fact/opinion splits and DKYoon dumps.  
- `scripts/`: preprocessing, experiment, and analysis entry points.  
- `src/research_workspace/`: reusable modules (data utils, baselines, LLM runner, metrics).  
- `results/`: summaries, predictions, and plots (see `results/analysis_summary.json`).  
- `artifacts/dkyoon_examples.json`: sampled epistemic QA traces for qualitative reference.

## Full Report
See `REPORT.md` for methodology, results, statistical tests, visuals, and next-step recommendations.
