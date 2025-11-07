import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style='whitegrid')

BASELINE_PATH = Path('results/predictions/baseline_test_predictions.csv')
LLM_PATH = Path('results/predictions/llm_test_predictions.csv')
SUMMARY_PATH = Path('results/analysis_summary.json')
PLOTS_DIR = Path('results/plots')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

EPIS_KEYWORDS = {
    'evidence', 'data', 'verified', 'proven', 'research', 'record', 'measurement', 'reported', 'source', 'fact'
}
OPINION_KEYWORDS = {
    'feel', 'believe', 'prefer', 'opinion', 'should', 'value', 'think', 'want', 'like', 'desire'
}


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return math.nan
    varx, vary = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * varx + (ny - 1) * vary) / (nx + ny - 2)
    if pooled <= 0:
        return math.nan
    return (np.mean(x) - np.mean(y)) / math.sqrt(pooled)


def keyword_counts(series: pd.Series, keywords: set[str]) -> np.ndarray:
    counts = []
    pattern = re.compile(r"[a-z']+")
    for text in series.fillna('').astype(str):
        tokens = pattern.findall(text.lower())
        counts.append(sum(token in keywords for token in tokens))
    return np.array(counts)


def make_histogram(df: pd.DataFrame, title: str, filename: str):
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df, x='prob_fact', hue='true_label', fill=True, common_norm=False, alpha=0.4)
    plt.title(title)
    plt.xlabel('Predicted probability of Fact (epistemic)')
    plt.ylabel('Density')
    plt.xlim(0, 1)
    plt.tight_layout()
    path = PLOTS_DIR / filename
    plt.savefig(path)
    plt.close()
    return str(path)


def main():
    baseline_df = pd.read_csv(BASELINE_PATH)
    llm_df = pd.read_csv(LLM_PATH)

    baseline_correct = (baseline_df['true_label'] == baseline_df['pred_label']).sum()
    llm_correct = (llm_df['true_label'] == llm_df['pred_label']).sum()
    n1, n2 = len(baseline_df), len(llm_df)
    p1, p2 = baseline_correct / n1, llm_correct / n2
    pooled = (baseline_correct + llm_correct) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se else math.inf
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    # Confidence distributions
    base_fact = baseline_df.loc[baseline_df['true_label'] == 'Fact', 'prob_fact'].values
    base_op = baseline_df.loc[baseline_df['true_label'] == 'Opinion', 'prob_fact'].values
    llm_fact = llm_df.loc[llm_df['true_label'] == 'Fact', 'prob_fact'].values
    llm_op = llm_df.loc[llm_df['true_label'] == 'Opinion', 'prob_fact'].values

    base_t = stats.ttest_ind(base_fact, base_op, equal_var=False)
    llm_t = stats.ttest_ind(llm_fact, llm_op, equal_var=False)

    base_d = cohens_d(base_fact, base_op)
    llm_d = cohens_d(llm_fact, llm_op)

    base_hist = make_histogram(baseline_df, 'Baseline confidence by true label', 'baseline_confidence_density.png')
    llm_hist = make_histogram(llm_df, 'LLM confidence by true label', 'llm_confidence_density.png')

    # Keyword analysis on LLM rationales grouped by true label
    epi_counts = keyword_counts(llm_df['rationale'], EPIS_KEYWORDS)
    op_counts = keyword_counts(llm_df['rationale'], OPINION_KEYWORDS)

    def contingency(counts, mask):
        subset = counts[mask]
        hits = int((subset > 0).sum())
        misses = int((subset == 0).sum())
        return [hits, misses]

    mask_fact = llm_df['true_label'] == 'Fact'
    mask_op = llm_df['true_label'] == 'Opinion'

    epi_table = [
        contingency(epi_counts, mask_op),
        contingency(epi_counts, mask_fact),
    ]
    opin_table = [
        contingency(op_counts, mask_op),
        contingency(op_counts, mask_fact),
    ]

    epi_chi2, epi_p, _, _ = stats.chi2_contingency(epi_table)
    opin_chi2, opin_p, _, _ = stats.chi2_contingency(opin_table)

    summary = {
        'accuracy_comparison': {
            'baseline_accuracy': p1,
            'llm_accuracy': p2,
            'z_stat': z,
            'p_value': p_val,
            'baseline_n': n1,
            'llm_n': n2,
        },
        'confidence_tests': {
            'baseline': {
                'fact_mean': float(np.mean(base_fact)),
                'opinion_mean': float(np.mean(base_op)),
                't_stat': float(base_t.statistic),
                'p_value': float(base_t.pvalue),
                'cohens_d': base_d,
            },
            'llm': {
                'fact_mean': float(np.mean(llm_fact)),
                'opinion_mean': float(np.mean(llm_op)),
                't_stat': float(llm_t.statistic),
                'p_value': float(llm_t.pvalue),
                'cohens_d': llm_d,
            },
        },
        'keyword_analysis': {
            'epistemic_table': epi_table,
            'epistemic_chi2_p': float(epi_p),
            'opinion_table': opin_table,
            'opinion_chi2_p': float(opin_p),
        },
        'plots': {
            'baseline_confidence_density': base_hist,
            'llm_confidence_density': llm_hist,
        },
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
