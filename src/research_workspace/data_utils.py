from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

DATA_DIR = Path('data/processed')
RESULTS_DIR = Path('results')
FACT_LABEL = 'Fact'
OPINION_LABEL = 'Opinion'
TARGET_LABELS = [FACT_LABEL, OPINION_LABEL]


def load_fact_opinion_split(split: str) -> pd.DataFrame:
    path = DATA_DIR / f'fact_opinion_en_{split}.jsonl'
    if not path.exists():
        raise FileNotFoundError(f'Missing split file: {path}')
    return pd.read_json(path, lines=True)


def save_predictions(df: pd.DataFrame, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_dir = RESULTS_DIR / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f'{name}.csv'
    df.to_csv(path, index=False)
    return path


def save_metrics(metrics: Dict, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f'{name}.json'
    path.write_text(json.dumps(metrics, indent=2))
    return path
