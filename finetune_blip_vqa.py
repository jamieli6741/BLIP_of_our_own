# finetune_blip_vqa.py
import os
import re
import string
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering,
    TrainingArguments,
    Trainer,
)
from custom_dataset import CustomVQADataset


# --------- Text normalization & metrics (EM + token-level F1) ---------
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(s: str) -> str:
    """Lowercase, remove punctuation, remove articles, squeeze spaces."""
    s = s.lower().strip()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _f1_score(pred: str, gold: str) -> float:
    """Token-level F1 between normalized strings (common for VQA/QA)."""
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()
    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0
    common = {}
    for t in pred_toks:
        common[t] = min(pred_toks.count(t), gold_toks.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def _exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


class BLIPVQAFineTuner:
    def __init__(self, model_name="Salesforce/blip-vqa-base", output_dir="./finetuned_blip"):
        self.model_name = model_name
        self.output_dir = output_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)

    def _postprocess_text(self, text: str) -> str:
        """Light cleanup of decoded strings."""
        return text.strip()

    def build_datasets(
        self,
        train_json: str,
        val_json: str,
        image_root: str = "",
        question_max_length: int = 40,
        answer_max_length: int = 20,
    ):
        train_ds = CustomVQADataset(
            annotations_path=train_json,
            processor=self.processor,
            image_root=image_root,
            question_max_length=question_max_length,
            answer_max_length=answer_max_length,
        )
        val_ds = CustomVQADataset(
            annotations_path=val_json,
            processor=self.processor,
            image_root=image_root,
            question_max_length=question_max_length,
            answer_max_length=answer_max_length,
        )
        return train_ds, val_ds

    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        """
        With predict_with_generate=True, `eval_preds.predictions` are generated sequences (np.ndarray of token IDs).
        `eval_preds.label_ids` are labels with -100 in pad positions.
        """
        pred_ids, label_ids = eval_preds
        # Convert -100 back to pad_token_id so we can decode cleanly
        pad_id = self.processor.tokenizer.pad_token_id
        label_ids = np.where(label_ids == -100, pad_id, label_ids)

        decoded_preds: List[str] = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        decoded_labels: List[str] = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        decoded_preds = [self._postprocess_text(p) for p in decoded_preds]
        decoded_labels = [self._postprocess_text(l) for l in decoded_labels]

        ems = [_exact_match(p, g) for p, g in zip(decoded_preds, decoded_labels)]
        f1s = [_f1_score(p, g) for p, g in zip(decoded_preds, decoded_labels)]

        return {
            "exact_match": float(np.mean(ems)),
            "f1": float(np.mean(f1s)),
        }

    def train(
        self,
        train_dataset,
        val_dataset,
        training_config: Dict = None,
    ):
        # Default training hyperparameters
        default_cfg = dict(
            num_train_epochs=5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=3e-5,
            weight_decay=0.01,
            warmup_ratio=0.06,
            logging_steps=50,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=1,
            predict_with_generate=True,        # important for generative VQA
            generation_max_length=20,          # answer length cap
            generation_num_beams=1,            # greedy by default; increase if needed
        )
        if training_config:
            default_cfg.update(training_config)

        args = TrainingArguments(
            output_dir=self.output_dir,
            **default_cfg,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.processor.tokenizer,  # for generation & logging
            compute_metrics=self.compute_metrics,
        )

        print("Starting training...")
        trainer.train()
        print("Saving model and processor...")
        trainer.save_model(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        return trainer


def main():
    # Minimal config you can tweak
    config = {
        "model_name": "Salesforce/blip-vqa-base",
        "output_dir": "./my_finetuned_blip",
        "train_json": "/home/jamie/Downloads/archive/processed/train.json",
        "val_json": "/home/jamie/Downloads/archive/processed/val.json",
        # If JSON "image" paths are relative, set the common image root here; else keep "".
        "image_root": "",
        "num_epochs": 10,
        "batch_size": 4,
        "learning_rate": 3e-5,
    }

    finetuner = BLIPVQAFineTuner(
        model_name=config["model_name"],
        output_dir=config["output_dir"],
    )

    train_ds, val_ds = finetuner.build_datasets(
        train_json=config["train_json"],
        val_json=config["val_json"],
        image_root=config["image_root"],
        question_max_length=40,
        answer_max_length=20,
    )

    train_cfg = {
        "num_train_epochs": config["num_epochs"],
        "per_device_train_batch_size": config["batch_size"],
        "per_device_eval_batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
    }

    finetuner.train(train_ds, val_ds, train_cfg)
    print("Training completed!")


if __name__ == "__main__":
    main()
