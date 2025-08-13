# custom_dataset.py
import json
from pathlib import Path
from typing import Dict, Any, List

from PIL import Image
import torch
from torch.utils.data import Dataset


class CustomVQADataset(Dataset):
    """
    A minimal VQA dataset for BLIP-style VQA fine-tuning.
    Expects a JSON list with items containing:
      {
        "image": "/abs/or/relative/path/to/img.jpg",
        "question": "What ...?",
        "answer": "red",
        "question_id": 123
      }
    """
    def __init__(
        self,
        annotations_path: str,
        processor,
        image_root: str = "",
        question_max_length: int = 40,
        answer_max_length: int = 20,
        padding: str = "max_length"
    ):
        """
        Args:
            annotations_path: Path to train/val JSON.
            processor: Hugging Face BlipProcessor (tokenizer + image processor).
            image_root: Optional root prefix if "image" paths are relative.
            question_max_length: Max tokens for question text.
            answer_max_length: Max tokens for answer text (labels).
            padding: "max_length" (recommended for stable batching) or "longest".
        """
        self.items: List[Dict[str, Any]] = json.loads(Path(annotations_path).read_text())
        self.processor = processor
        self.image_root = Path(image_root) if image_root else None
        self.question_max_length = question_max_length
        self.answer_max_length = answer_max_length
        self.padding = padding

        # Basic validation
        required_keys = {"image", "question", "answer"}
        for i, ex in enumerate(self.items):
            missing = required_keys - set(ex)
            if missing:
                raise KeyError(f"Example {i} is missing keys: {missing}. Found: {list(ex.keys())}")

    def __len__(self):
        return len(self.items)

    def _resolve_image_path(self, img_field: str) -> Path:
        p = Path(img_field)
        if self.image_root is not None and not p.is_absolute():
            p = self.image_root / p
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        return p

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.items[idx]
        img_path = self._resolve_image_path(ex["image"])
        question = ex["question"]
        answer = ex["answer"]

        # Load image (RGB)
        image = Image.open(img_path).convert("RGB")

        # Encode image + question (inputs)
        enc = self.processor(
            images=image,
            text=question,
            padding=self.padding,
            max_length=self.question_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Encode answer (labels). We set pad positions to -100 so they are ignored by loss.
        labels = self.processor.tokenizer(
            answer,
            padding=self.padding,
            max_length=self.answer_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        pad_token_id = self.processor.tokenizer.pad_token_id
        labels = labels.clone()
        labels[labels == pad_token_id] = -100  # ignore padding in loss

        sample = {
            "pixel_values": enc["pixel_values"][0],      # (3, H, W)
            "input_ids": enc["input_ids"][0],            # (Q_len,)
            "attention_mask": enc["attention_mask"][0],  # (Q_len,)
            "labels": labels,                             # (A_len,)
        }
        # Optional helpful fields
        if "question_id" in ex:
            sample["question_id"] = torch.tensor(ex["question_id"], dtype=torch.long)
        return sample
