from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from .metrics import classification_metrics

JSON_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)


@dataclass
class LLMRunResult:
    metrics: Dict
    predictions: pd.DataFrame


class LocalLLMClassifier:
    def __init__(
        self,
        model_id: str = 'Qwen/Qwen2.5-0.5B-Instruct',
        max_new_tokens: int = 180,
        device: str = 'cpu',
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
        )
        self.model.eval()

    @staticmethod
    def _normalize_label(label: str) -> Optional[str]:
        clean = label.strip().lower()
        if 'fact' in clean:
            return 'Fact'
        if 'opinion' in clean or 'value' in clean:
            return 'Opinion'
        return None

    @staticmethod
    def _clamp_conf(value) -> float:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return 0.5
        if math.isnan(num) or math.isinf(num):
            return 0.5
        return float(min(max(num, 0.0), 1.0))

    def _build_messages(self, text: str) -> List[Dict[str, str]]:
        system_prompt = (
            "You are an analyst distinguishing epistemic beliefs (facts) from non-epistemic"
            " beliefs (opinions/preferences). Return JSON only."
        )
        user_prompt = (
            "Classify whether the following statement expresses an epistemic belief (Fact)"
            " or a non-epistemic belief (Opinion).\n"
            "State clear reasoning referencing either evidence or preferences.\n"
            "Respond strictly as JSON with keys label (Fact/Opinion), confidence (0-1 probability"
            " that it is a Fact), and rationale.\n\n"
            f"Statement: {text}"
        )
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]

    def _generate(self, text: str) -> str:
        messages = self._build_messages(text)
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors='pt',
            add_generation_prompt=True,
        ).to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[:, input_ids.shape[-1]:]
        decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return decoded.strip()

    def _parse(self, text: str) -> Dict:
        match = JSON_PATTERN.search(text)
        if not match:
            raise ValueError('No JSON object found')
        data = json.loads(match.group())
        label = self._normalize_label(data.get('label', ''))
        if not label:
            raise ValueError('Invalid label')
        confidence = self._clamp_conf(data.get('confidence', 0.5))
        rationale = data.get('rationale', '').strip()
        return {
            'pred_label': label,
            'prob_fact': confidence,
            'rationale': rationale,
            'raw_response': text,
        }

    def classify(self, text: str) -> Dict:
        raw = self._generate(text)
        try:
            parsed = self._parse(raw)
            parsed['valid'] = True
        except Exception:
            parsed = {
                'pred_label': 'Opinion',
                'prob_fact': 0.5,
                'rationale': '',
                'raw_response': raw,
                'valid': False,
            }
        return parsed

    def evaluate(self, df: pd.DataFrame, limit: Optional[int] = None) -> LLMRunResult:
        subset = df.copy()
        if limit is not None:
            subset = subset.sample(n=min(limit, len(df)), random_state=42).reset_index(drop=True)
        records = []
        for row in tqdm(subset.itertuples(index=False), total=len(subset), desc='LLM eval'):
            result = self.classify(row.text)
            records.append({
                'id': row.id,
                'text': row.text,
                'true_label': row.label,
                **result,
            })
        pred_df = pd.DataFrame(records)
        metrics_bundle = classification_metrics(
            y_true=pred_df['true_label'].tolist(),
            y_pred=pred_df['pred_label'].tolist(),
            prob_fact=pred_df['prob_fact'].tolist(),
        )
        metrics = {
            **metrics_bundle.overall,
            'per_class': metrics_bundle.per_class,
            'confusion': metrics_bundle.confusion,
            'confidence_by_true': metrics_bundle.confidence_by_true,
            'valid_ratio': float(pred_df['valid'].mean()),
            'samples': int(len(pred_df)),
            'model_id': self.model_id,
        }
        return LLMRunResult(metrics=metrics, predictions=pred_df)
