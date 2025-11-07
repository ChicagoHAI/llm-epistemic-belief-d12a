from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state

from .metrics import classification_metrics


@dataclass
class BaselineResults:
    metrics: Dict
    predictions: pd.DataFrame


class LogisticFactOpinion:
    def __init__(self, random_state: int = 42):
        self.random_state = check_random_state(random_state)
        self.pipeline: Pipeline | None = None
        self.fact_index: int | None = None

    def fit(self, df: pd.DataFrame):
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=self.random_state)
        self.pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('clf', clf),
        ])
        self.pipeline.fit(df['text'], df['label'])
        self.fact_index = int(np.where(self.pipeline.named_steps['clf'].classes_ == 'Fact')[0][0])

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None or self.fact_index is None:
            raise RuntimeError('Model not fitted.')
        proba = self.pipeline.predict_proba(df['text'])
        p_fact = proba[:, self.fact_index]
        preds = self.pipeline.predict(df['text'])
        return pd.DataFrame({
            'id': df['id'].values,
            'text': df['text'].values,
            'true_label': df['label'].values,
            'pred_label': preds,
            'prob_fact': p_fact,
        })

    def evaluate(self, df: pd.DataFrame) -> BaselineResults:
        preds_df = self.predict(df)
        metrics_bundle = classification_metrics(
            y_true=preds_df['true_label'].tolist(),
            y_pred=preds_df['pred_label'].tolist(),
            prob_fact=preds_df['prob_fact'].tolist(),
        )
        metrics = {
            **metrics_bundle.overall,
            'per_class': metrics_bundle.per_class,
            'confusion': metrics_bundle.confusion,
            'confidence_by_true': metrics_bundle.confidence_by_true,
        }
        return BaselineResults(metrics=metrics, predictions=preds_df)
