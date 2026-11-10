
import json
import csv
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models" / "phi-distilbert"
TEST_FILE = ROOT / "data" / "test.jsonl"
OUTPUT_DIR = ROOT / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENTITY_TYPES = [
    "NAME",
    "LOCATION",
    "DATE",
    "PHONE",
    "EMAIL",
    "SSN",
    "MRN",
    "HEALTH_PLAN",
    "ACCOUNT",
    "LICENSE",
    "URL",
    "IP",
    "AGE_OVER_89",
]


# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------

def load_test_data(path):
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):

            if not line.strip():
                continue

            row = json.loads(line)

            if len(row["tokens"]) != len(row["ner_tags"]):
                raise ValueError(
                    f"Token/tag mismatch on line {line_number}"
                )

            rows.append(row)

    if not rows:
        raise ValueError("Test dataset is empty.")

    return rows


# --------------------------------------------------
# EXTRACT ENTITY SPANS
# --------------------------------------------------

def extract_entities(tags):
    """
    Extract BIO entity spans.

    Returns:
        (entity_type, start_token, end_token)
    """

    entities = []

    current_type = None
    start = None

    for index, tag in enumerate(tags + ["O"]):

        if tag == "O":
            prefix = "O"
            entity_type = None

        else:
            prefix, entity_type = tag.split("-", 1)

        if prefix == "B" or (
            current_type is not None
            and entity_type != current_type
        ):

            if current_type is not None:
                entities.append(
                    (current_type, start, index)
                )

            current_type = (
                entity_type if prefix == "B" else None
            )

            start = (
                index if prefix == "B" else None
            )

        elif prefix == "I":

            if current_type is None:
                # Treat an I-tag without a preceding entity
                # as a new entity span.
                current_type = entity_type
                start = index

        elif prefix == "O" and current_type is not None:

            entities.append(
                (current_type, start, index)
            )

            current_type = None
            start = None

    return entities


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(gold, predicted):

    gold_set = set(gold)
    predicted_set = set(predicted)

    tp = len(gold_set & predicted_set)
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)

    precision = (
        tp / (tp + fp)
        if tp + fp else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        use_fast=True
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required."
        )

    model = AutoModelForTokenClassification.from_pretrained(
        str(MODEL_DIR)
    )

    model.to(device)
    model.eval()

    id2label = {
        int(key): value
        for key, value in model.config.id2label.items()
    }

    rows = load_test_data(TEST_FILE)

    gold_all = []
    predicted_all = []

    gold_by_type = defaultdict(list)
    predicted_by_type = defaultdict(list)

    prediction_rows = []
    mismatch_lines = []

    truncated_notes = []

    print(f"Test notes: {len(rows)}")

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Model parameters: {parameter_count:,}")

    # --------------------------------------------------
    # PROCESS EACH NOTE
    # --------------------------------------------------

    for note_number, row in enumerate(rows, 1):

        tokens = row["tokens"]
        gold_tags = row["ner_tags"]

        encoded = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            return_tensors="pt"
        )

        word_ids = encoded.word_ids(
            batch_index=0
        )

        model_inputs = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            logits = model(**model_inputs).logits[0]

        predicted_ids = torch.argmax(
            logits,
            dim=-1
        ).cpu().tolist()

        predicted_tags = [None] * len(tokens)

        for token_position, word_id in enumerate(word_ids):

            if (
                word_id is not None
                and predicted_tags[word_id] is None
            ):
                predicted_tags[word_id] = id2label[
                    predicted_ids[token_position]
                ]

        # Identify evaluated tokens.
        valid_indices = [
            index
            for index, prediction in enumerate(predicted_tags)
            if prediction is not None
        ]

        if len(valid_indices) != len(tokens):

            truncated_notes.append(note_number)

            print(
                f"WARNING: Note {note_number} was truncated. "
                f"{len(tokens) - len(valid_indices)} tokens "
                "were not evaluated."
            )

        gold_valid = [
            gold_tags[index]
            for index in valid_indices
        ]

        pred_valid = [
            predicted_tags[index]
            for index in valid_indices
        ]

        gold_entities = extract_entities(gold_valid)
        predicted_entities = extract_entities(pred_valid)

        # Add note number to each entity.
        # This prevents duplicate offsets in different notes
        # from being treated as the same entity.
        gold_tagged = [
            (note_number, entity_type, start, end)
            for entity_type, start, end in gold_entities
        ]

        predicted_tagged = [
            (note_number, entity_type, start, end)
            for entity_type, start, end in predicted_entities
        ]

        gold_all.extend(gold_tagged)
        predicted_all.extend(predicted_tagged)

        # Per-entity collection.
        for entity in gold_tagged:
            gold_by_type[entity[1]].append(entity)

        for entity in predicted_tagged:
            predicted_by_type[entity[1]].append(entity)

        # Exact mismatch analysis.
        gold_set = set(gold_tagged)
        predicted_set = set(predicted_tagged)

        false_positive = predicted_set - gold_set
        false_negative = gold_set - predicted_set

        if false_positive or false_negative:

            mismatch_lines.append(
                f"\nNOTE {note_number}"
            )

            mismatch_lines.append(
                "Text: " + " ".join(tokens)
            )

            for _, entity_type, start, end in sorted(
                false_positive
            ):

                entity_text = " ".join(
                    tokens[valid_indices[index]]
                    for index in range(start, end)
                )

                mismatch_lines.append(
                    f"FALSE POSITIVE [{entity_type}]: "
                    f"{entity_text}"
                )

            for _, entity_type, start, end in sorted(
                false_negative
            ):

                entity_text = " ".join(
                    tokens[valid_indices[index]]
                    for index in range(start, end)
                )

                mismatch_lines.append(
                    f"FALSE NEGATIVE [{entity_type}]: "
                    f"{entity_text}"
                )

        prediction_rows.append({

            "note_number": note_number,

            "text": " ".join(tokens),

            "gold_tags": " ".join(gold_tags),

            "predicted_tags": " ".join(
                tag if tag is not None
                else "NOT_EVALUATED"
                for tag in predicted_tags
            )

        })

    # --------------------------------------------------
    # OVERALL AND PER-ENTITY METRICS
    # --------------------------------------------------

    overall = calculate_metrics(
        gold_all,
        predicted_all
    )

    per_entity = {}

    for entity_type in ENTITY_TYPES:

        gold = gold_by_type[entity_type]
        predicted = predicted_by_type[entity_type]

        if not gold:

            per_entity[entity_type] = {
                "status": "N/A - no gold examples in test set",
                "gold_count": 0,
                "predicted_count": len(predicted),
                "false_positives": len(predicted),
            }

        else:

            metrics = calculate_metrics(
                gold,
                predicted
            )

            metrics["gold_count"] = len(gold)
            metrics["predicted_count"] = len(predicted)
            metrics["status"] = "Evaluated"

            per_entity[entity_type] = metrics

    # --------------------------------------------------
    # JSON REPORT
    # --------------------------------------------------

    report = {

        "system": "DistilBERT PHI De-identification Model",

        "test_notes": len(rows),

        "model_parameters": parameter_count,

        "matching_method": (
            "Exact token-span, entity-type and note-ID match"
        ),

        "truncated_notes": truncated_notes,

        "overall": overall,

        "per_entity": per_entity

    }

    json_path = OUTPUT_DIR / "distilbert_comparison.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # --------------------------------------------------
    # TEXT REPORT
    # --------------------------------------------------

    report_lines = []

    report_lines.append(
        "PHI DE-IDENTIFICATION MODEL EVALUATION"
    )

    report_lines.append("=" * 75)

    report_lines.append(
        f"Test notes: {len(rows)}"
    )

    report_lines.append(
        f"Model parameters: {parameter_count:,}"
    )

    report_lines.append(
        "Matching: Exact token-span, entity-type and note-ID"
    )

    report_lines.append(
        f"Truncated notes: {len(truncated_notes)}"
    )

    report_lines.append("")

    report_lines.append(
        f"{'ENTITY':<18} {'PRECISION':>10} "
        f"{'RECALL':>10} {'F1':>10} "
        f"{'TP':>6} {'FP':>6} {'FN':>6}"
    )

    report_lines.append("-" * 75)

    for entity_type in ENTITY_TYPES:

        values = per_entity[entity_type]

        if values["status"] != "Evaluated":

            report_lines.append(
                f"{entity_type:<18} {'N/A':>10} "
                f"{'N/A':>10} {'N/A':>10} "
                f"{'-':>6} "
                f"{values['false_positives']:>6} "
                f"{'-':>6}"
            )

            continue

        report_lines.append(

            f"{entity_type:<18} "
            f"{values['precision']:>10.4f} "
            f"{values['recall']:>10.4f} "
            f"{values['f1']:>10.4f} "
            f"{values['TP']:>6} "
            f"{values['FP']:>6} "
            f"{values['FN']:>6}"

        )

    report_lines.append("-" * 75)

    report_lines.append(

        f"{'OVERALL':<18} "
        f"{overall['precision']:>10.4f} "
        f"{overall['recall']:>10.4f} "
        f"{overall['f1']:>10.4f} "
        f"{overall['TP']:>6} "
        f"{overall['FP']:>6} "
        f"{overall['FN']:>6}"

    )

    report_lines.append("")
    report_lines.append("ENTITY COUNTS")

    report_lines.append(
        f"True positives: {overall['TP']}"
    )

    report_lines.append(
        f"False positives: {overall['FP']}"
    )

    report_lines.append(
        f"False negatives: {overall['FN']}"
    )

    report_lines.append(
        f"Gold entities: {overall['TP'] + overall['FN']}"
    )

    report_lines.append(
        f"Predicted entities: {overall['TP'] + overall['FP']}"
    )

    report_path = OUTPUT_DIR / "evaluation_report.txt"

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    # --------------------------------------------------
    # PREDICTIONS CSV
    # --------------------------------------------------

    csv_path = OUTPUT_DIR / "predictions.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=[
                "note_number",
                "text",
                "gold_tags",
                "predicted_tags"
            ]

        )

        writer.writeheader()
        writer.writerows(prediction_rows)

    # --------------------------------------------------
    # MISMATCH REPORT
    # --------------------------------------------------

    mismatch_path = OUTPUT_DIR / "mismatches.txt"

    mismatch_path.write_text(

        "\n".join(mismatch_lines)
        if mismatch_lines
        else "No mismatches found.",

        encoding="utf-8"

    )

    # --------------------------------------------------
    # TERMINAL OUTPUT
    # --------------------------------------------------

    print("\n".join(report_lines))

    print("\nEvaluation files saved:")

    print(f"- {json_path}")
    print(f"- {report_path}")
    print(f"- {csv_path}")
    print(f"- {mismatch_path}")


if __name__ == "__main__":
    main()
