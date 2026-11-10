
import json
from pathlib import Path
from collections import defaultdict

from presidio_analyzer import AnalyzerEngine


ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "data" / "test.jsonl"
OUTPUT_FILE = ROOT / "evaluation" / "presidio_comparison.json"

analyzer = AnalyzerEngine()

# Map Presidio entity types to project entity types.
ENTITY_MAP = {
    "PERSON": "NAME",
    "LOCATION": "LOCATION",
    "GPE": "LOCATION",
    "DATE_TIME": "DATE",
    "PHONE_NUMBER": "PHONE",
    "EMAIL_ADDRESS": "EMAIL",
    "US_SSN": "SSN",
    "URL": "URL",
    "IP_ADDRESS": "IP",
    "AGE": "AGE_OVER_89",
}

SUPPORTED_TYPES = sorted(set(ENTITY_MAP.values()))


def extract_gold(tokens, tags):
    """Convert BIO tags into exact character-offset spans."""

    text = " ".join(tokens)
    entities = []
    offsets = []
    position = 0

    for token in tokens:
        start = position
        end = start + len(token)
        offsets.append((start, end))
        position = end + 1

    current_type = None
    current_start = None
    current_end = None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if current_type:
                entities.append(
                    (current_start, current_end, current_type)
                )

            current_type = tag[2:]
            current_start = offsets[i][0]
            current_end = offsets[i][1]

        elif tag.startswith("I-") and current_type == tag[2:]:
            current_end = offsets[i][1]

        else:
            if current_type:
                entities.append(
                    (current_start, current_end, current_type)
                )

            current_type = None
            current_start = None
            current_end = None

    if current_type:
        entities.append(
            (current_start, current_end, current_type)
        )

    return text, entities


def extract_presidio(text):
    """Run Presidio and return mapped character-offset spans."""

    results = analyzer.analyze(
        text=text,
        language="en"
    )

    entities = []

    for result in results:
        mapped_type = ENTITY_MAP.get(result.entity_type)

        if mapped_type:
            entities.append(
                (result.start, result.end, mapped_type)
            )

    return entities


def calculate_metrics(gold, predicted):
    """Calculate exact-span precision, recall and F1."""

    gold_set = set(gold)
    predicted_set = set(predicted)

    tp = len(gold_set & predicted_set)
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

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



def main():
    gold_all = []
    predicted_all = []

    gold_by_type = defaultdict(list)
    predicted_by_type = defaultdict(list)

    note_count = 0

    with open(TEST_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            row = json.loads(line)

            text, gold = extract_gold(
                row["tokens"],
                row["ner_tags"]
            )

            predicted = extract_presidio(text)

            # Include note ID so identical offsets in different
            # notes are treated as different entities.
            gold_tagged = [
                (note_count, start, end, entity_type)
                for start, end, entity_type in gold
            ]

            predicted_tagged = [
                (note_count, start, end, entity_type)
                for start, end, entity_type in predicted
            ]

            gold_all.extend(gold_tagged)
            predicted_all.extend(predicted_tagged)

            for entity in gold_tagged:
                gold_by_type[entity[3]].append(entity)

            for entity in predicted_tagged:
                predicted_by_type[entity[3]].append(entity)

            note_count += 1

    report = {
        "system": "Microsoft Presidio",
        "test_notes": note_count,
        "matching_method": "Exact character-span, entity-type and note-ID match",
        "overall": calculate_metrics(gold_all, predicted_all),
        "per_entity": {}
    }

    for entity_type in SUPPORTED_TYPES:
        gold = gold_by_type[entity_type]
        predicted = predicted_by_type[entity_type]

        if not gold:
            report["per_entity"][entity_type] = {
                "status": "N/A - no gold examples in test set",
                "gold_count": 0,
                "predicted_count": len(predicted),
                "false_positives": len(predicted),
            }
        else:
            metrics = calculate_metrics(gold, predicted)
            metrics["gold_count"] = len(gold)
            metrics["predicted_count"] = len(predicted)
            metrics["status"] = "Evaluated"

            report["per_entity"][entity_type] = metrics

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n===== PRESIDIO EVALUATION =====")
    print("Test notes:", note_count)
    print("Overall:", report["overall"])

    print("\nPer-entity results:")

    for entity_type, metrics in report["per_entity"].items():
        print(f"\n{entity_type}:")
        print(metrics)

    print("\nReport saved to:", OUTPUT_FILE)

if __name__ == "__main__":
    main()