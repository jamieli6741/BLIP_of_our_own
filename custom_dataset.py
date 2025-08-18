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
        padding: str = "max_length",
        is_medical: bool = False,
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

        # method 1: callback processor
        try:
            inputs = self.processor(
                images=image,
                text=question,
                text_target=answer,
                padding=self.padding,
                max_length=self.question_max_length,
                truncation=True,
                return_tensors="pt",
            )

            sample = {
                "pixel_values": inputs["pixel_values"][0],
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "labels": inputs["labels"][0],
            }

        except Exception as e:
            print(f"Method 1 failed: {e}, trying method 2...")

            # 方法2：分别处理图像+问题和答案
            # metho 2: deal with image+question and answer separately
            enc = self.processor(
                images=image,
                text=question,
                padding=self.padding,
                max_length=self.question_max_length,
                truncation=True,
                return_tensors="pt",
            )

            # directly use tokenizer to handle answers (not text_target）
            labels = self.processor.tokenizer(
                answer,  # remove text_target
                padding=self.padding,
                max_length=self.answer_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]

            pad_token_id = self.processor.tokenizer.pad_token_id
            labels = labels.clone()
            labels[labels == pad_token_id] = -100

            sample = {
                "pixel_values": enc["pixel_values"][0],
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels": labels,
            }

        # Debug info
        # print(f"Answer: {answer}")
        # print(f"Labels shape: {sample['labels'].shape}")
        # print(f"Labels min/max: {sample['labels'].min()}, {sample['labels'].max()}")
        # print(f"Vocab size: {self.processor.tokenizer.vocab_size}")

        return sample
