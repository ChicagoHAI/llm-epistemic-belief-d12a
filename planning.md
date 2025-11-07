# Planning Document

## Research Question
Do contemporary LLMs differentiate epistemic beliefs (about facts/knowledge) from non-epistemic beliefs (values, preferences, intentions) in a measurable way when prompted with natural-language statements?

## Background and Motivation
Human cognition distinguishes epistemic beliefs (truth-evaluable propositions) from conative or evaluative beliefs (preferences, desires). Vesga et al. argue these belief types follow different cognitive dynamics. Recent AI work on epistemic control (e.g., Dumbrava, 2025) and belief revision (Sauerwald & Thimm, 2024) focuses on factual beliefs but rarely probes whether LLMs internally separate epistemic vs non-epistemic content. Understanding this distinction matters for safety/alignment (ensuring LLM factual claims remain traceable) and for user modeling (responding differently to preferences vs facts). Our study aims to operationalize this distinction via existing classification datasets and qualitative probes.

## Hypothesis Decomposition
1. **H1 (Classification Competence):** Given a statement, an LLM can classify whether it encodes an epistemic belief (fact) or a non-epistemic belief (opinion/value) with accuracy significantly above chance.
2. **H2 (Calibration Shift):** The LLM expresses different confidence distributions for epistemic vs non-epistemic classifications (e.g., higher certainty on factual statements), indicating differentiated internal treatment.
3. **H3 (Reasoning Signatures):** The textual rationales LLMs provide for epistemic labels mention evidence/knowledge terms more often than rationales for non-epistemic labels, reflecting qualitative separation.

Independent variables: statement type (fact vs opinion), prompt framing (zero-shot vs with instructions). Dependent variables: classification accuracy, macro-F1, Brier score, confidence distribution statistics, lexical counts in rationales.

Success criteria:
- Accuracy > 0.7 and macro-F1 > 0.7 on fact-or-opinion benchmark.
- Statistically significant difference (p < 0.05) between mean reported confidences for epistemic vs non-epistemic predictions.
- Evidence of lexical divergence (e.g., >20% difference in frequency of evidence-oriented keywords) in rationale analysis.

## Proposed Methodology

### Approach
Combine quantitative benchmarking on `agentlans/fact-or-opinion` with qualitative analysis on DKYoon epistemic QA traces. Use both an open-weight LLM (Phi-3.5 Mini Instruct, CPU-friendly) and a lightweight logistic-regression baseline to contextualize LLM performance. Collect structured outputs (JSON: label, probability, rationale) to compute metrics and analyze textual cues.

### Experimental Steps
1. **Dataset acquisition & preprocessing** – download/decompress fact-or-opinion dataset, select English samples, stratify into train/dev/test (e.g., 5k/1k/1k). Inspect DKYoon QA splits for qualitative study.
2. **Baseline training** – use TF-IDF + logistic regression trained on train split to create a reference classifier.
3. **LLM prompt harness** – design deterministic prompt for classification, requesting JSON with `label`, `confidence`, `rationale`. Implement runner supporting open-weight inference (transformers pipeline) and, if API keys become available, OpenRouter calls (GPT-4.1/Claude Sonnet) for comparison.
4. **Evaluation** – run models on shared test set (~300 samples for cost control) with temperature=0. Parse outputs, compute accuracy, macro-F1, precision/recall, Brier score. For confidence shift, perform Welch's t-test between epistemic vs non-epistemic predictions.
5. **Rationale analysis** – tokenise rationales, count frequency of epistemic keywords ("evidence", "data", "fact", etc.) vs affective keywords ("feel", "prefer", "value"). Conduct chi-square or difference-in-proportions test.
6. **Qualitative DKYoon review** – prompt LLM on select QA entries to see if it references belief type indicators when reasoning about epistemic questions; document examples.

### Baselines
- **TF-IDF Logistic Regression** trained on fact-or-opinion train split (scikit-learn). Acts as a non-LLM baseline.
- **Random guess** (50/50) to contextualize performance floor.
- Optional: majority class baseline if class imbalance observed.

### Evaluation Metrics
- Classification accuracy and macro-F1 on test set (captures performance per class).
- Precision/recall per class to detect asymmetry.
- Brier score using reported `confidence` for calibrated probability evaluation.
- Welch's t-test on confidence per class; effect size (Cohen's d).
- Keyword frequency ratios in rationales; chi-square test for independence.

### Statistical Analysis Plan
- For H1: compare accuracy to 0.5 using one-sample binomial test; compare LLM vs logistic via McNemar (paired predictions) if feasible.
- For H2: compute mean/variance of confidence scores per class; apply Welch's t-test (unequal variances) and report Cohen's d. Verify assumptions (approx. normality via Shapiro on samples >=30; fall back to Mann-Whitney if violated).
- For H3: Build contingency table of keyword counts vs label; apply chi-square test of independence; compute Cramér's V for effect size.
- Adjust for multiple comparisons using Holm-Bonferroni when testing multiple hypotheses simultaneously.

## Expected Outcomes
- Evidence that LLM distinguishes facts vs opinions (accuracy >0.7) and expresses higher confidence plus evidence-oriented language on epistemic cases. Negative result would indicate undifferentiated treatment.

## Timeline and Milestones (target total ~6h, with 20% buffer)
1. **Phase 2 – Setup & EDA (0.8h):** install deps, download datasets, inspect distributions.
2. **Phase 3 – Implementation (1.5h):** baseline training, LLM harness coding, config files.
3. **Phase 4 – Experimentation (1.5h):** run models, collect outputs, control randomness.
4. **Phase 5 – Analysis (1.0h):** stats, plots, rationale keyword mining.
5. **Phase 6 – Documentation (0.8h):** REPORT.md, README, finalize artifacts.
Buffer (~0.4h) for debugging/API access issues.

## Potential Challenges & Mitigations
- **API key unavailability:** fallback to open-weight LLM (Phi-3.5) and document limitation; if API later available, rerun harness by swapping backend.
- **Dataset imbalance or multilingual content:** filter to English, use stratified sampling.
- **Parsing LLM output errors:** enforce JSON schema via regex validation; retry on failures.
- **Compute limits (CPU-only):** restrict sample sizes, use batching, quantized model if needed.
- **Ambiguous statements:** incorporate human-readable rationale review; treat ambiguous cases separately during error analysis.

## Success Criteria
- Completed experiments with reproducible code and saved outputs in `results/`.
- At least one LLM baseline evaluated with quantitative metrics plus statistical tests supporting or refuting hypotheses.
- REPORT.md summarizing actual findings, limitations, and future work.
