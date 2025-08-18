# medical_vqa_model.py
import os
import re
import string
from typing import Dict, List, Tuple, Optional

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


class MedicalVQASystem:
    """
    A comprehensive Medical Visual Question Answering system based on BLIP.
    Handles medical images with questions and generates appropriate answers.
    """

    def __init__(self, model_name="Salesforce/blip-vqa-base", output_dir="./medical_vqa_model"):
        self.model_name = model_name
        self.output_dir = output_dir

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load processor and model
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)

        # Medical-specific configurations
        self.medical_keywords = {
            'anatomy': ['bone', 'muscle', 'organ', 'tissue', 'cell', 'artery', 'vein'],
            'pathology': ['tumor', 'lesion', 'fracture', 'inflammation', 'infection', 'cancer'],
            'imaging': ['x-ray', 'ct', 'mri', 'ultrasound', 'scan', 'image', 'radiological'],
            'symptoms': ['pain', 'swelling', 'abnormal', 'normal', 'healthy', 'disease']
        }

    def _postprocess_text(self, text: str) -> str:
        """Medical-specific text cleanup."""
        text = text.strip()
        # Remove common artifacts from medical text generation
        text = re.sub(r'\b(the patient|patient)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _is_medical_question(self, question: str) -> bool:
        """Check if question is medical-related."""
        question_lower = question.lower()
        for category, keywords in self.medical_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return True
        return False

    def build_datasets(
            self,
            train_json: str,
            val_json: str,
            test_json: Optional[str] = None,
            image_root: str = "",
            question_max_length: int = 50,  # Longer for medical questions
            answer_max_length: int = 30,  # Longer for medical answers
    ):
        """Build training, validation, and optional test datasets."""

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

        test_ds = None
        if test_json:
            test_ds = CustomVQADataset(
                annotations_path=test_json,
                processor=self.processor,
                image_root=image_root,
                question_max_length=question_max_length,
                answer_max_length=answer_max_length,
            )

        return train_ds, val_ds, test_ds

    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        """
        Compute VQA metrics including exact match and F1 score.
        Enhanced with medical-specific evaluation criteria.
        """
        pred_ids, label_ids = eval_preds

        # Handle -100 padding tokens
        pad_id = self.processor.tokenizer.pad_token_id
        label_ids = np.where(label_ids == -100, pad_id, label_ids)

        # Decode predictions and labels
        decoded_preds: List[str] = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        decoded_labels: List[str] = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        # Postprocess
        decoded_preds = [self._postprocess_text(p) for p in decoded_preds]
        decoded_labels = [self._postprocess_text(l) for l in decoded_labels]

        # Compute standard metrics
        ems = [_exact_match(p, g) for p, g in zip(decoded_preds, decoded_labels)]
        f1s = [_f1_score(p, g) for p, g in zip(decoded_preds, decoded_labels)]

        # Medical-specific metrics
        medical_accuracy = self._compute_medical_accuracy(decoded_preds, decoded_labels)

        return {
            "exact_match": float(np.mean(ems)),
            "f1": float(np.mean(f1s)),
            "medical_accuracy": medical_accuracy,
        }

    def _compute_medical_accuracy(self, predictions: List[str], labels: List[str]) -> float:
        """Compute medical domain-specific accuracy."""
        correct = 0
        total = len(predictions)

        for pred, label in zip(predictions, labels):
            # Check for medical terminology matches
            pred_medical_terms = set()
            label_medical_terms = set()

            for category, keywords in self.medical_keywords.items():
                pred_medical_terms.update([kw for kw in keywords if kw in pred.lower()])
                label_medical_terms.update([kw for kw in keywords if kw in label.lower()])

            # Consider correct if significant medical term overlap
            if pred_medical_terms and label_medical_terms:
                overlap = len(pred_medical_terms.intersection(label_medical_terms))
                if overlap / len(label_medical_terms) >= 0.5:
                    correct += 1
            elif _exact_match(pred, label) > 0.8:  # Fallback to high threshold EM
                correct += 1

        return correct / total if total > 0 else 0.0

    def train(
            self,
            train_dataset,
            val_dataset,
            training_config: Dict = None,
    ):
        """Train the medical VQA model."""

        # Default training configuration optimized for medical VQA
        default_cfg = {
            "num_train_epochs": 2,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "learning_rate": 2e-5,  # Lower LR for medical domain
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "logging_steps": 50,
            "save_strategy": "epoch",
            "evaluation_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "greater_is_better": True,
            "fp16": torch.cuda.is_available(),
            "gradient_accumulation_steps": 2,
            "save_total_limit": 3,
            "report_to": None,  # Disable wandb logging by default
        }

        if training_config:
            default_cfg.update(training_config)

        # Use regular TrainingArguments for BLIP-VQA
        args = TrainingArguments(
            output_dir=self.output_dir,
            **default_cfg,
        )

        # Use regular Trainer for BLIP-VQA
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.processor.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        print("Starting medical VQA training...")
        trainer.train()

        print("Saving model and processor...")
        trainer.save_model(self.output_dir)
        self.processor.save_pretrained(self.output_dir)

        return trainer

    def predict(self, image, question: str, max_length: int = 30) -> str:
        """
        Generate answer for a given image and question.

        Args:
            image: PIL Image or path to image
            question: Question string
            max_length: Maximum length of generated answer

        Returns:
            Generated answer string
        """
        self.model.eval()

        # Process inputs
        inputs = self.processor(image, question, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate answer
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=3,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        # Decode answer
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        return self._postprocess_text(answer)

    def evaluate_model(self, test_dataset):
        """Evaluate the model on test dataset."""

        args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_eval_batch_size=4,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            tokenizer=self.processor.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        print("Evaluating model...")
        results = trainer.evaluate(test_dataset)

        print("Evaluation Results:")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")

        return results

    @classmethod
    def load_trained_model(cls, model_path: str):
        """Load a previously trained model."""
        system = cls.__new__(cls)
        system.output_dir = model_path
        system.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        system.processor = BlipProcessor.from_pretrained(model_path)
        system.model = BlipForQuestionAnswering.from_pretrained(model_path)
        system.model.to(system.device)

        return system


def main():
    """Main training script for medical VQA system."""

    # Configuration
    config = {
        "model_name": "Salesforce/blip-vqa-base",
        "output_dir": "./medical_vqa_output",
        "train_json": "/home/jamie/Downloads/archive/processed/train.json",
        "val_json": "/home/jamie/Downloads/archive/processed/val.json",
        "test_json": None,  # Optional test set
        "image_root": "",
        "num_epochs": 8,
        "batch_size": 4,
        "learning_rate": 2e-5,
    }

    print("Initializing Medical VQA System...")
    vqa_system = MedicalVQASystem(
        model_name=config["model_name"],
        output_dir=config["output_dir"],
    )

    print("Loading datasets...")
    train_ds, val_ds, test_ds = vqa_system.build_datasets(
        train_json=config["train_json"],
        val_json=config["val_json"],
        test_json=config["test_json"],
        image_root=config["image_root"],
        question_max_length=50,
        answer_max_length=30,
    )

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")

    # Training configuration
    train_cfg = {
        "num_train_epochs": config["num_epochs"],
        "per_device_train_batch_size": config["batch_size"],
        "per_device_eval_batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
    }

    print("Starting training...")
    trainer = vqa_system.train(train_ds, val_ds, train_cfg)

    print("Training completed!")

    # Optional: Evaluate on test set if available
    if test_ds:
        print("Evaluating on test set...")
        vqa_system.evaluate_model(test_ds)


if __name__ == "__main__":
    main()
