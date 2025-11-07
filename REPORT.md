# REPORT: Do LLMs Differentiate Epistemic vs Non-Epistemic Beliefs?

## 1. Executive Summary
- **Research question:** Can LLMs distinguish epistemic (truth-evaluable) beliefs from non-epistemic (value/preference) beliefs with distinct confidence and reasoning signatures?
- **Key finding:** A TF-IDF logistic baseline reached 94.1% accuracy and strong calibration on the English subset of `agentlans/fact-or-opinion`, while the small open-weight LLM (`Qwen/Qwen2.5-0.5B-Instruct`) collapsed to labeling nearly every statement as factual (50% accuracy on 60 sampled cases) and used epistemic rhetoric even on opinions.
- **Implication:** Off-the-shelf lightweight LLMs do not inherently separate belief types; explicit supervision or stronger models/APIs are needed to avoid over-confident factual framing of opinions.

## 2. Goal
- **Hypothesis:** LLMs express different behaviors (classification accuracy, confidence, rationales) for epistemic versus non-epistemic beliefs, mirroring human cognitive distinctions.  
- **Importance:** Safety, alignment, and user-modeling rely on models knowing when they are stating facts versus preferences.  
- **Problem solved:** Provide early-stage evidence about whether general-purpose LLMs encode separate belief categories.  
- **Expected impact:** Identify gaps where prompting or training must enforce epistemic awareness before deployment in safety-critical scenarios.

## 3. Data Construction
### Dataset Description
| Dataset | Source | Notes |
| --- | --- | --- |
| `agentlans/fact-or-opinion` (2025) | HuggingFace | Synthetic multilingual statements with labels `Fact`, `Opinion`, `Both`, `Neither`. Filtered to English + {Fact, Opinion}; total 1,342 items. Summary: mean length 47 chars / 7.9 tokens (see `results/fact_opinion_summary.json`). |
| `DKYoon/r1-triviaqa-epistemic` & `DKYoon/r1-nonambigqa-epistemic` | HuggingFace | 1k QA items each with `model_think` traces. Used qualitatively to inspect epistemic reasoning cues; sampled examples saved in `artifacts/dkyoon_examples.json`. |

### Example Samples
| Text | Label |
| --- | --- |
| “The currency of India is the rupee.” | Fact |
| “Online shopping is more convenient than going to stores.” | Opinion |
| “Watching sunsets is calming.” | Opinion |

### Data Quality
- No missing fields after filtering; duplicate IDs removed implicitly by sampling.  
- Length stats (English fact/opinion subset): char mean 47.25 (min 15, max 95); token mean 7.90 (min 3, max 18).  
- Class distribution: 736 opinions vs 606 facts (slightly imbalanced but both >40%).  
- Multilingual noise removed by restricting to `language == 'en'` and dropping `Both/Neither`.

### Preprocessing Steps
1. Download `.zst` archives, decompress to JSONL (`data/processed/`).
2. Filter to English statements and valid labels; compute char/token lengths.
3. Stratified splits via `sklearn.train_test_split`: 60/20/20 into train/val/test.
4. Save splits (`data/processed/fact_opinion_en_{train,val,test}.jsonl`) and dataset stats (`results/fact_opinion_summary.json`).
5. Download DKYoon QA splits using `datasets`, export to JSONL for inspection.

### Train/Val/Test Splits
| Split | Size |
| --- | --- |
| Train | 805 |
| Validation | 268 |
| Test | 269 |
| (LLM subset) | 60 sampled from test for expensive inference |

## 4. Experiment Description
### Methodology
#### High-Level Approach
- Establish a supervised baseline (TF-IDF + logistic regression) on the fact/opinion classification task.  
- Run an instruction-following LLM locally to produce label, confidence (probability statement is epistemic), and rationale.  
- Compare metrics, confidence distributions, and lexical cues; run statistical tests for accuracy and calibration differences.  
- Inspect DKYoon QA traces to contextualize epistemic reasoning language.

#### Why This Method?
- Logistic regression provides a transparent, competitive baseline and sanity check on dataset difficulty.  
- Running an open-weight LLM (due to missing API keys) satisfies the requirement to probe real model behavior while remaining CPU-compatible.  
- Confidence/rationale collection enables testing hypotheses beyond raw accuracy.

### Implementation Details
#### Tools and Libraries
- Python 3.12.2, `numpy` 2.3.4, `pandas` 2.3.3, `scikit-learn` 1.7.2, `scipy` 1.16.3.  
- `datasets` 4.4.1 for HuggingFace downloads, `transformers` 4.57.1 + `torch` 2.9.0 for LLM inference.  
- Visualization via `matplotlib` 3.10.7 and `seaborn` 0.13.2.  
- CLI orchestration with `typer` 0.20.0.

#### Algorithms/Models
- **Baseline:** TF-IDF (1-2 grams, 5k features) + logistic regression (`class_weight='balanced'`).  
- **LLM:** `Qwen/Qwen2.5-0.5B-Instruct` (chat template, greedy decoding, max 60 tokens) generating JSON label/confidence/rationale for each statement.  
- **Keyword analysis:** Regex-based counts of epistemic vs preference lexicon inside rationales.

#### Hyperparameters
| Component | Parameter | Value | Selection |
| --- | --- | --- | --- |
| TF-IDF | n-gram range | (1, 2) | Common for short-text classification |
| Logistic | max_iter | 2000 | Needed for convergence |
| Logistic | class_weight | balanced | Mitigates slight class skew |
| LLM | max_new_tokens | 60 | Trade-off between runtime and completeness |
| LLM | sample size | 60 statements | Ensures <10 min CPU inference |

#### Training / Analysis Pipeline
1. `scripts/prep_fact_opinion.py` → preprocess & split data.  
2. `scripts/run_experiments.py baseline` → train baseline, save predictions & metrics.  
3. `scripts/run_experiments.py llm --limit 60 --max-new-tokens 60` → run LLM on sampled test set.  
4. `scripts/analyze_results.py` → compute statistical tests, produce plots, and summarize findings.  
5. Manual inspection of DKYoon QA examples stored in `artifacts/dkyoon_examples.json` for qualitative cues.

### Experimental Protocol
#### Reproducibility Information
- Runs performed on CPU-only environment (`Qwen2.5-0.5B` inference ≈ 9 minutes for 60 prompts).  
- Random seeds fixed at 42 across sklearn splits and sampling.  
- Results artifacts: metrics JSONs in `results/`, predictions CSVs in `results/predictions/`, plots in `results/plots/`.  
- Code entry points documented in README (see Section 7).

#### Evaluation Metrics
- Accuracy, macro-F1 to capture balanced performance.  
- Precision/recall per class for asymmetry.  
- Brier score (probabilistic calibration for “Fact”).  
- Welch’s t-test + Cohen’s d on confidence per ground-truth label.  
- Two-proportion z-test comparing baseline vs LLM accuracy.  
- Chi-square tests on rationale keyword usage.

### Raw Results
#### Metrics Table (Test Set)
| Model | Accuracy | Macro-F1 | Brier ↓ | Notes |
| --- | --- | --- | --- | --- |
| TF-IDF Logistic | 0.941 | 0.940 | 0.081 | `n=269`, strong calibration (Fact conf = 0.77, Opinion conf = 0.27). |
| Qwen2.5-0.5B LLM | 0.500 | 0.384 | 0.446 | `n=60` sampled; predicted “Fact” almost everywhere; valid JSON ratio 0.967. |

#### Visualizations
- Confidence density plots stored at `results/plots/baseline_confidence_density.png` and `results/plots/llm_confidence_density.png`. Baseline shows clear bimodal separation; LLM curves collapse near 0.9 for both labels.  

#### Output Locations
- Metrics: `results/baseline_test_metrics.json`, `results/llm_test_metrics.json`, `results/analysis_summary.json`.  
- Predictions: `results/predictions/baseline_test_predictions.csv`, `results/predictions/llm_test_predictions.csv`.  
- DKYoon qualitative samples: `artifacts/dkyoon_examples.json`.

## 5. Result Analysis
### Key Findings
1. **Baseline success:** Logistic regression achieved 94.1% accuracy / 0.081 Brier, confirming the dataset cleanly separates fact vs opinion.  
2. **LLM collapse:** Qwen2.5-0.5B defaulted to labeling nearly everything as factual, yielding 50% accuracy and poor calibration (Brier 0.446).  
3. **Confidence misuse:** Despite mislabeling, the LLM kept average confidence ≥0.90 for both ground-truth labels (Welch’s t-test p=0.003, but means only differ by 0.07), showing little modulation by belief type.  
4. **Rationale language:** Epistemic keywords appeared often even for human-labeled opinions (11/32 cases), while preference keywords were virtually absent (chi-square p=1.0), indicating rationales stayed factual regardless of ground truth.

### Hypothesis Testing Results
- **H1:** Baseline accuracy significantly above chance (binomial p ≪ 1e-10). LLM accuracy difference vs baseline: z=8.90, p<1e-18 → LLM fails H1 on sampled set.  
- **H2:** Baseline confidence gap (Fact mean 0.77 vs Opinion 0.27, t=31.5, Cohen’s d=3.89) supports differentiated treatment. LLM gap (0.97 vs 0.91, t=3.18) exists but both are near certainty, so practical differentiation absent.  
- **H3:** Chi-square tests on rationale keywords not significant; LLM explanations reuse epistemic framing irrespective of label, refuting H3 for this small model.

### Comparison to Baselines
- Logistic baseline dramatically outperforms the LLM; random guess (50% accuracy) matches LLM’s performance. This indicates dataset difficulty does not explain the LLM’s failure—it's a modeling issue.

### Visualizations
- **Figure:** `results/plots/baseline_confidence_density.png` shows distinct peaks around 0.2 (opinions) vs 0.8 (facts).  
- **Figure:** `results/plots/llm_confidence_density.png` shows both labels clustered near 1.0, underscoring overconfidence.

### Surprises and Insights
- Even without advanced prompts, simple TF-IDF features capture belief type cues effectively.  
- The lightweight LLM’s JSON compliance was high (96.7%), but label diversity was minimal—structured outputs alone do not ensure conceptual differentiation.  
- DKYoon QA traces revealed frequent use of epistemic verbs (“I remember”, “sources indicate”) regardless of uncertainty, mirroring the factual-overgeneralization seen in the classification task.

### Error Analysis
- Manual inspection of LLM mistakes: statements like “Watching sunsets is calming” were justified as “observable phenomena,” suggesting the model conflates subjective experiences with factual claims.  
- Baseline errors often involved borderline statements (“Recycling is essential for communities”) where lexical cues mix normative and factual terms.

### Limitations
- LLM evaluation limited to 60 samples due to CPU-only inference constraints; results may differ for larger runs.  
- Unable to call premium APIs (GPT-4.1/Claude 4.5) because no keys were present; conclusions pertain to an open-weight 0.5B model.  
- Dataset synthetic nature may not capture the full nuance of human belief statements.  
- DKYoon QA analysis remained qualitative; no automatic scoring performed.

## 6. Conclusions
### Summary
The small open-weight LLM tested here does **not** reliably differentiate epistemic from non-epistemic beliefs: it classifies most statements as factual and articulates rationales with epistemic language regardless of ground truth. In contrast, a classical supervised model easily captures the distinction, highlighting that LLM prompting alone is insufficient.

### Implications
- Deployment of lightweight LLMs in moderation or advisory roles risks over-stating opinions as facts.  
- Calibration routines or fine-tuning specifically on epistemic/non-epistemic labels are necessary to avoid misleading confidence signals.  
- For safety research, benchmarking against simple baselines remains critical to detect such regressions.

### Confidence in Findings
Moderate: baseline results are robust, but LLM analysis is constrained by sample size and model choice. Running stronger API models and larger samples would improve confidence.

## 7. Next Steps
### Immediate Follow-ups
1. Run the same pipeline on a higher-capacity API model (e.g., GPT-4.1) to check if larger models exhibit the desired differentiation.  
2. Fine-tune or prompt-tune the Qwen model with explicit counterexamples to encourage opinion recognition, then reassess calibration.

### Alternative Approaches
1. Augment dataset with real-world corpora (news vs editorials) to reduce synthetic bias.  
2. Replace JSON confidence reporting with logit extraction (via log-probs) to avoid self-reported probabilities.

### Broader Extensions
- Combine belief-type classification with DKYoon epistemic QA traces to see whether reasoning tokens correlate with correctness.  
- Explore multi-class belief taxonomies (epistemic, deontic, bouletic) to better match cognitive science literature.

### Open Questions
- Which prompting strategies (chain-of-thought, contrastive instructions) best elicit belief-type differentiation?  
- Can confidence metrics derived from logits (instead of self-report) better align with belief categories?  
- How do larger instruction-tuned models behave on multilingual belief statements?
