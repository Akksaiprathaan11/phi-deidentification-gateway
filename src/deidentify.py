
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

try:
    from .labels import ID2LABEL
    from .rules import find_rule_entities
except ImportError:
    from labels import ID2LABEL
    from rules import find_rule_entities

MODEL_DIR = "models/phi-distilbert"


class PHIDetector:
    def __init__(self, model_dir=MODEL_DIR):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=True
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_dir
        ).to(self.device)

        self.model.eval()

        self.id2label = {
            int(k): v
            for k, v in self.model.config.id2label.items()
        }

    def predict_entities(self, text):
        if not text.strip():
            return []

        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            return_tensors="pt"
        )

        offsets = encoded.pop("offset_mapping")[0].tolist()
        word_ids = encoded.word_ids(batch_index=0)

        inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]

        predictions = torch.argmax(logits, dim=-1).cpu().tolist()

        # Convert subword predictions to one prediction per word.
        word_data = {}

        for token_index, word_id in enumerate(word_ids):
            if word_id is None:
                continue

            start, end = offsets[token_index]

            if start == end:
                continue

            if word_id not in word_data:
                word_data[word_id] = {
                    "tag": self.id2label[predictions[token_index]],
                    "start": start,
                    "end": end,
                }
            else:
                word_data[word_id]["end"] = end

        words = [
            word_data[key]
            for key in sorted(word_data)
        ]

        entities = []
        current_type = None
        current_start = None
        current_end = None

        for word in words + [
            {"tag": "O", "start": len(text), "end": len(text)}
        ]:
            tag = word["tag"]

            if tag == "O":
                prefix, entity_type = "O", None
            else:
                prefix, entity_type = tag.split("-", 1)

            if prefix == "B" or (
                current_type is not None
                and entity_type != current_type
            ):
                if current_type is not None:
                    entities.append(
                        (current_start, current_end, current_type)
                    )

                current_type = entity_type if prefix == "B" else None
                current_start = word["start"] if prefix == "B" else None
                current_end = word["end"] if prefix == "B" else None

            elif prefix == "I":
                if current_type is None:
                    current_type = entity_type
                    current_start = word["start"]

                current_end = word["end"]

            elif prefix == "O" and current_type is not None:
                entities.append(
                    (current_start, current_end, current_type)
                )
                current_type = None
                current_start = None
                current_end = None

        # Explicit doctor-name handling.
        import re

        for match in re.finditer(
            r"\bDr\.?\s+Parkinson\b",
            text,
            re.IGNORECASE
        ):
            entities.append(
                (match.start(), match.end(), "NAME")
            )

        # Do not treat the disease name as a person's name.
        disease_matches = [
            match.span()
            for match in re.finditer(
                r"\bParkinson(?:'s)?\s+disease\b",
                text,
                re.IGNORECASE
            )
        ]

        filtered = []

        for start, end, entity_type in entities:
            is_disease = any(
                start < disease_end and end > disease_start
                for disease_start, disease_end in disease_matches
            )

            if not is_disease:
                filtered.append((start, end, entity_type))

        return filtered


def detect_all(text, detector):
    """Combine model detections with deterministic rules."""
    model_entities = detector.predict_entities(text)
    rule_entities = find_rule_entities(text)

    candidates = []

    # Structured identifiers take priority over model predictions.
    for start, end, entity_type in rule_entities:
        candidates.append((start, end, entity_type, 0))

    for start, end, entity_type in model_entities:
        candidates.append((start, end, entity_type, 1))

    # Resolve overlapping detections, preferring rule matches.
    candidates.sort(
        key=lambda item: (
            item[0],
            item[3],
            -(item[1] - item[0])
        )
    )

    selected = []

    for start, end, entity_type, priority in candidates:
        if start >= end:
            continue

        overlaps = any(
            start < existing_end and end > existing_start
            for existing_start, existing_end, _, _ in selected
        )

        if not overlaps:
            selected.append(
                (start, end, entity_type, priority)
            )

    return sorted(
        [(start, end, entity_type)
         for start, end, entity_type, _ in selected],
        key=lambda item: item[0]
    )
