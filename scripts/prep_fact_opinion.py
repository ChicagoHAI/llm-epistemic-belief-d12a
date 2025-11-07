import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_FILES = [
    Path('data/processed/fact_or_opinion_train.jsonl'),
    Path('data/processed/fact_or_opinion_test.jsonl'),
]
OUTPUT_DIR = Path('data/processed')
RESULTS_DIR = Path('results')
VALID_LABELS = {'Fact', 'Opinion'}
TARGET_LANG = 'en'


def load_samples(paths: List[Path]) -> pd.DataFrame:
    records: List[Dict] = []
    for path in paths:
        with path.open() as f:
            for line in f:
                obj = json.loads(line)
                if obj.get('language') != TARGET_LANG:
                    continue
                label = obj.get('label')
                if label not in VALID_LABELS:
                    continue
                text = obj['text'].strip()
                if not text:
                    continue
                records.append(
                    {
                        'id': obj['id'],
                        'text': text,
                        'label': label,
                        'source': obj.get('source', 'unknown'),
                        'language': obj['language'],
                        'char_len': len(text),
                        'token_len': len(text.split()),
                    }
                )
    return pd.DataFrame(records)


def stratified_split(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.4,
        stratify=df['label'],
        random_state=42,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['label'],
        random_state=42,
    )
    return train_df, val_df, test_df


def save_jsonl(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for record in df[['id', 'text', 'label', 'source']].to_dict(orient='records'):
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    df = load_samples(RAW_FILES)
    if df.empty:
        raise RuntimeError('No samples found after filtering.')

    train_df, val_df, test_df = stratified_split(df)

    save_jsonl(train_df, OUTPUT_DIR / 'fact_opinion_en_train.jsonl')
    save_jsonl(val_df, OUTPUT_DIR / 'fact_opinion_en_val.jsonl')
    save_jsonl(test_df, OUTPUT_DIR / 'fact_opinion_en_test.jsonl')

    stats = {
        'total_samples': int(len(df)),
        'label_distribution': df['label'].value_counts().to_dict(),
        'train_split': len(train_df),
        'val_split': len(val_df),
        'test_split': len(test_df),
        'char_length': {
            'mean': float(df['char_len'].mean()),
            'std': float(df['char_len'].std(ddof=0)),
            'min': int(df['char_len'].min()),
            'max': int(df['char_len'].max()),
        },
        'token_length': {
            'mean': float(df['token_len'].mean()),
            'std': float(df['token_len'].std(ddof=0)),
            'min': int(df['token_len'].min()),
            'max': int(df['token_len'].max()),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / 'fact_opinion_summary.json'
    summary_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
