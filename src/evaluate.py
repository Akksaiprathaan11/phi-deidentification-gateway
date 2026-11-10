import json
import csv
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


MODEL_DIR = "models/phi-distilbert"
TEST_FILE = "data/test.jsonl"
OUTPUT_DIR = Path("evaluation")
OUTPUT_DIR.mkdir(exist_ok=True)


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


def extract_entities(tags):
    """Return entity spans as (entity_type, start, end)."""
    entities = []
    current_type = None
    start = None

    for index, tag in enumerate(tags + ["O"]):
        if tag == "O":
            prefix, entity_type = "O", None
        else:
            prefix, entity_type = tag.split("-", 1)

        if prefix == "B" or (
            current_type is not None and entity_type != current_type
        ):
            if current_type is not None:
                entities.append((current_type, start, index))
            current_type = entity_type if prefix == "B" else None
            start = index if prefix == "B" else None

        elif prefix == "I":
            if current_type is None:
                # Treat an I-tag without a preceding entity as a new span.
                current_type = entity_type
                start = index

        elif prefix == "O" and current_type is not None:
            entities.append((current_type, start, index))
            current_type = None
            start = None

    return set(entities)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_fast=True
    )

    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required.")

    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    id2label = {
        int(key): value
        for key, value in model.config.id2label.items()
    }

    rows = load_test_data(TEST_FILE)

    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    total_tp = total_fp = total_fn = 0
    prediction_rows = []
    mismatch_lines = []

    print(f"Test notes: {len(rows)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for note_number, row in enumerate(rows, 1):
        tokens = row["tokens"]
        gold_tags = row["ner_tags"]

        encoded = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            return_tensors="pt"
        )

        word_ids = encoded.word_ids(batch_index=0)
        model_inputs = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            logits = model(**model_inputs).logits[0]

        predicted_ids = torch.argmax(logits, dim=-1).cpu().tolist()

        predicted_tags = [None] * len(tokens)

        for token_position, word_id in enumerate(word_ids):
            if word_id is not None and predicted_tags[word_id] is None:
                predicted_tags[word_id] = id2label[predicted_ids[token_position]]

        # Exclude tokens lost to truncation from scoring.
        valid_indices = [
            i for i, prediction in enumerate(predicted_tags)
            if prediction is not None
        ]

        if len(valid_indices) != len(tokens):
            print(
                f"WARNING: Note {note_number} was truncated. "
                f"{len(tokens) - len(valid_indices)} tokens were not evaluated."
            )

        gold_valid = [gold_tags[i] for i in valid_indices]
        pred_valid = [predicted_tags[i] for i in valid_indices]

        gold_entities = extract_entities(gold_valid)
        pred_entities = extract_entities(pred_valid)

        true_positive = gold_entities & pred_entities
        false_positive = pred_entities - gold_entities
        false_negative = gold_entities - pred_entities

        total_tp += len(true_positive)
        total_fp += len(false_positive)
        total_fn += len(false_negative)

        for entity_type, _, _ in true_positive:
            counts[entity_type]["tp"] += 1

        for entity_type, _, _ in false_positive:
            counts[entity_type]["fp"] += 1

        for entity_type, _, _ in false_negative:
            counts[entity_type]["fn"] += 1

        if false_positive or false_negative:
            mismatch_lines.append(f"\nNOTE {note_number}")
            mismatch_lines.append("Text: " + " ".join(tokens))

            for entity_type, start, end in sorted(false_positive):
                entity_text = " ".join(
                    [tokens[valid_indices[i]] for i in range(start, end)]
                )
                mismatch_lines.append(
                    f"FALSE POSITIVE [{entity_type}]: {entity_text}"
                )

            for entity_type, start, end in sorted(false_negative):
                entity_text = " ".join(
                    [tokens[valid_indices[i]] for i in range(start, end)]
                )
                mismatch_lines.append(
                    f"FALSE NEGATIVE [{entity_type}]: {entity_text}"
                )

        prediction_rows.append({
            "note_number": note_number,
            "text": " ".join(tokens),
            "gold_tags": " ".join(gold_tags),
            "predicted_tags": " ".join(
                tag if tag is not None else "NOT_EVALUATED"
                for tag in predicted_tags
            )
        })

    def calculate(tp, fp, fn):
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        return precision, recall, f1

    report_lines = []
    report_lines.append("PHI DE-IDENTIFICATION MODEL EVALUATION")
    report_lines.append("=" * 55)
    report_lines.append(f"Test notes: {len(rows)}")
    report_lines.append(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    report_lines.append("")

    report_lines.append(
        f"{'ENTITY':<18} {'PRECISION':>10} {'RECALL':>10} {'F1':>10} "
        f"{'TP':>6} {'FP':>6} {'FN':>6}"
    )
    report_lines.append("-" * 75)

    for entity_type in sorted(counts):
        values = counts[entity_type]
        p, r, f1 = calculate(
            values["tp"], values["fp"], values["fn"]
        )
        report_lines.append(
            f"{entity_type:<18} {p:>10.4f} {r:>10.4f} {f1:>10.4f} "
            f"{values['tp']:>6} {values['fp']:>6} {values['fn']:>6}"
        )

    p, r, f1 = calculate(total_tp, total_fp, total_fn)

    report_lines.append("-" * 75)
    report_lines.append(
        f"{'OVERALL':<18} {p:>10.4f} {r:>10.4f} {f1:>10.4f} "
        f"{total_tp:>6} {total_fp:>6} {total_fn:>6}"
    )

    report_lines.append("")
    report_lines.append("ENTITY COUNTS")
    report_lines.append(f"True positives: {total_tp}")
    report_lines.append(f"False positives: {total_fp}")
    report_lines.append(f"False negatives: {total_fn}")

    report_path = OUTPUT_DIR / "evaluation_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    csv_path = OUTPUT_DIR / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
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

    mismatch_path = OUTPUT_DIR / "mismatches.txt"
    mismatch_path.write_text(
        "\n".join(mismatch_lines) if mismatch_lines else "No mismatches found.",
        encoding="utf-8"
    )

    print("\n".join(report_lines))
    print("\nFiles saved:")
    print(f"- {report_path}")
    print(f"- {csv_path}")
    print(f"- {mismatch_path}")


if __name__ == "__main__":
    main()