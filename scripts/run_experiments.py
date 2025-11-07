import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from research_workspace.baselines import LogisticFactOpinion
from research_workspace.data_utils import (
    load_fact_opinion_split,
    save_metrics,
    save_predictions,
)
from research_workspace.llm_runner import LocalLLMClassifier

app = typer.Typer()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


@app.command()
def baseline(seed: int = 42):
    """Train TF-IDF logistic baseline and evaluate on val/test."""
    set_seed(seed)
    train_df = load_fact_opinion_split('train')
    val_df = load_fact_opinion_split('val')
    test_df = load_fact_opinion_split('test')

    model = LogisticFactOpinion(random_state=seed)
    model.fit(train_df)
    val_results = model.evaluate(val_df)
    test_results = model.evaluate(test_df)

    save_predictions(val_results.predictions, 'baseline_val_predictions')
    save_predictions(test_results.predictions, 'baseline_test_predictions')
    save_metrics(val_results.metrics, 'baseline_val_metrics')
    save_metrics(test_results.metrics, 'baseline_test_metrics')

    typer.echo('Baseline val metrics:')
    typer.echo(json.dumps(val_results.metrics, indent=2))
    typer.echo('Baseline test metrics:')
    typer.echo(json.dumps(test_results.metrics, indent=2))


@app.command()
def llm(
    model_id: str = typer.Option('Qwen/Qwen2.5-0.5B-Instruct', help='HF model id'),
    limit: Optional[int] = typer.Option(200, help='Number of samples to evaluate (None = all).'),
    max_new_tokens: int = typer.Option(180, help='Generation length for responses.'),
    seed: int = 42,
):
    """Run local LLM classifier on the test split."""
    set_seed(seed)
    test_df = load_fact_opinion_split('test')
    classifier = LocalLLMClassifier(model_id=model_id, max_new_tokens=max_new_tokens)
    result = classifier.evaluate(test_df, limit=limit)

    save_predictions(result.predictions, 'llm_test_predictions')
    save_metrics(result.metrics, 'llm_test_metrics')

    typer.echo('LLM metrics:')
    typer.echo(json.dumps(result.metrics, indent=2))


@app.command()
def full(
    limit: Optional[int] = 200,
    model_id: str = 'Qwen/Qwen2.5-0.5B-Instruct',
    max_new_tokens: int = 180,
    seed: int = 42,
):
    """Run both baseline and LLM experiments."""
    baseline(seed)
    llm(model_id=model_id, limit=limit, max_new_tokens=max_new_tokens, seed=seed)
    overview = {
        'baseline_test': json.loads(Path('results/baseline_test_metrics.json').read_text()),
        'llm_test': json.loads(Path('results/llm_test_metrics.json').read_text()),
    }
    save_metrics(overview, 'metrics_overview')
    typer.echo('Overview saved to results/metrics_overview.json')


if __name__ == '__main__':
    app()
