#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_custom_dataset.py

Build BLIP-style VQA JSONs (train/val/test) from a CSV or JSON annotations file.

Features:
- Accepts image column as one of: `image_path`, `image`, or `image_id`.
- If using `image_id`, will search under <data_dir>/images for {id}.png/.jpg/.jpeg.
- If using `image_path` or `image`, supports both absolute and relative paths (relative to data_dir).
- Accepts CSV or JSON (array) as annotations.
- Reuses `question_id` when present; otherwise auto-generates.
- Saves train/val/test JSON files under output_dir.

Typical usage:
    python prepare_custom_dataset.py \
        --data-dir /path/to/dataset \
        --annotations /path/to/annotations.csv \
        --output-dir /path/to/processed
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class CustomDatasetPreparer:
    def __init__(self, data_dir: str, output_dir: str, id_prefix: str = "q"):
        """
        Args:
            data_dir:   Root directory where images live. If annotation uses relative paths,
                        they are joined to this directory. If using image_id, images are
                        looked up in <data_dir>/images.
            output_dir: Directory to write train/val/test JSONs.
            id_prefix:  Optional prefix for auto-generated question_id to avoid collisions.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.id_prefix = id_prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Column & file loading ----------

    def _load_annotations(self, annotations_file: str) -> pd.DataFrame:
        """Load CSV or JSON (array) annotations."""
        ext = Path(annotations_file).suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(annotations_file)
        elif ext == ".json":
            df = pd.read_json(annotations_file)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Use .csv or .json")
        if df.empty:
            raise RuntimeError(f"Annotation file has no rows: {annotations_file}")
        return df

    def _resolve_columns(
        self, df: pd.DataFrame
    ) -> Tuple[str, str, str, Optional[str]]:
        """
        Normalize column names to canonical ones:
        Returns (image_col, question_col, answer_col, qid_col_or_None)
        """
        lower = {c.lower(): c for c in df.columns}

        # Image column: prefer explicit paths; otherwise allow IDs.
        image_col = lower.get("image_path") or lower.get("image") or lower.get("image_id")
        if image_col is None:
            raise KeyError(
                f"Need one of ['image_path','image','image_id'] in annotations. "
                f"Available: {list(df.columns)}"
            )

        question_col = lower.get("question")
        answer_col = lower.get("answer")
        if question_col is None or answer_col is None:
            raise KeyError(
                f"Need columns 'question' and 'answer'. Available: {list(df.columns)}"
            )

        qid_col = lower.get("question_id")  # optional
        return image_col, question_col, answer_col, qid_col

    # ---------- Image path resolving ----------

    def _resolve_path_from_id(self, img_id: str) -> Optional[str]:
        """
        Resolve an image path when the annotation provides `image_id` without extension.
        Tries <data_dir>/images/{id}.png/.jpg/.jpeg in that order.
        """
        if not isinstance(img_id, str) or not img_id.strip():
            return None
        img_id = img_id.strip()
        candidates = [
            self.data_dir / "images" / f"{img_id}.png",
            self.data_dir / "images" / f"{img_id}.jpg",
            self.data_dir / "images" / f"{img_id}.jpeg",
        ]
        for p in candidates:
            if p.is_file():
                return p.as_posix()
        return None

    def _resolve_path_from_str(self, raw_path: str) -> Optional[str]:
        """
        Resolve an image path when the annotation provides a path (`image_path` or `image`).
        - If absolute and exists: return as-is.
        - Else join with data_dir and check existence.
        """
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        raw_path = raw_path.strip().replace("\\", "/")
        p = Path(raw_path)
        if p.is_file():
            return p.as_posix()
        p2 = self.data_dir / p
        return p2.as_posix() if p2.is_file() else None

    # ---------- Core building ----------

    def _build_records(
        self,
        df: pd.DataFrame,
        image_col: str,
        question_col: str,
        answer_col: str,
        qid_col: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Iterate rows and build BLIP-style records:
        {image, question, answer, question_id}
        Skips rows where image cannot be resolved or mandatory fields are empty.
        """
        use_image_id = image_col.lower() == "image_id"
        out: List[Dict[str, Any]] = []

        for idx, row in df.iterrows():
            # Resolve image path
            if use_image_id:
                img = self._resolve_path_from_id(str(row[image_col]))
            else:
                img = self._resolve_path_from_str(str(row[image_col]))
            if img is None:
                continue  # skip unresolved images

            # Question / Answer
            q = str(row[question_col]).strip() if pd.notna(row[question_col]) else ""
            a = row[answer_col]
            # If answer is a list, take the first item or join (customize if needed)
            if isinstance(a, list):
                a = a[0] if a else ""
            a = str(a).strip() if pd.notna(a) else ""

            if not q or not a:
                continue

            # Question ID: reuse if present, else generate
            if qid_col is not None and pd.notna(row[qid_col]):
                try:
                    qid = int(row[qid_col])
                except Exception:
                    qid = f"{self.id_prefix}{idx}"
            else:
                qid = f"{self.id_prefix}{idx}"

            out.append(
                {
                    "image": img,          # absolute path preferred for downstream
                    "question": q,
                    "answer": a,
                    "question_id": qid,
                }
            )

        if not out:
            raise RuntimeError(
                "No valid samples were built. "
                "Check that images exist and required columns are correct."
            )
        return out

    # ---------- Public API ----------

    def prepare_vqa_dataset(
        self,
        annotations_file: str,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Build BLIP-style VQA JSONs (train/val/test) from annotations.

        Args:
            annotations_file: CSV or JSON with columns:
                             - image_path / image / image_id
                             - question
                             - answer
                             - (optional) question_id
            test_size:        Fraction of total samples to allocate to test split.
            val_size:         Fraction of total samples to allocate to validation split.
            random_state:     Random seed for deterministic splits.

        Returns:
            (train_list, val_list, test_list)
        """
        df = self._load_annotations(annotations_file)
        image_col, question_col, answer_col, qid_col = self._resolve_columns(df)
        records = self._build_records(df, image_col, question_col, answer_col, qid_col)

        # Split: first split off test, then split train/val (val proportion relative to the remaining)
        train_val, test = train_test_split(records, test_size=test_size, random_state=random_state)
        val_rel = val_size / (1.0 - test_size)
        train, val = train_test_split(train_val, test_size=val_rel, random_state=random_state)

        # Save files
        def _dump(name: str, data: List[Dict[str, Any]]):
            out_path = self.output_dir / f"{name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        _dump("train", train)
        _dump("val", val)
        _dump("test", test)

        print("Dataset prepared:")
        print(f"  Train: {len(train)}")
        print(f"  Val:   {len(val)}")
        print(f"  Test:  {len(test)}")
        print(f"JSONs saved to: {self.output_dir.resolve()}")

        return train, val, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BLIP-style VQA dataset splits.")
    parser.add_argument(
        "--data-dir", required=True, type=str,
        help="Root directory of the dataset. If using image_id, images are searched in <data-dir>/images."
    )
    parser.add_argument(
        "--annotations", required=True, type=str,
        help="CSV or JSON annotations file (must include image_path/image/image_id, question, answer)."
    )
    parser.add_argument(
        "--output-dir", required=True, type=str,
        help="Directory to write train/val/test JSONs."
    )
    parser.add_argument(
        "--id-prefix", default="q", type=str,
        help="Prefix for auto-generated question_id (used when question_id column is absent)."
    )
    parser.add_argument("--test-size", default=0.15, type=float, help="Test fraction (default 0.15)")
    parser.add_argument("--val-size", default=0.15, type=float, help="Validation fraction (default 0.15)")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for splits")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preparer = CustomDatasetPreparer(args.data_dir, args.output_dir, id_prefix=args.id_prefix)
    preparer.prepare_vqa_dataset(
        annotations_file=args.annotations,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
    )
  
# usage example: 
# python prepare_custom_dataset.py \
#  --data-dir /home/jamie/Downloads/archive/dataset \
#  --annotations /home/jamie/Downloads/archive/dataset/data_train.csv \
#  --output-dir /home/jamie/Downloads/archive/processed
