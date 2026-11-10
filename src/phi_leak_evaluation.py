
import json
import csv
import re
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gateway import deidentify

TEST_FILE = ROOT / "data" / "test.jsonl"
OUTPUT_DIR = ROOT / "evaluation"

JSON_OUTPUT = OUTPUT_DIR / "phi_leak_report.json"
CSV_OUTPUT = OUTPUT_DIR / "phi_leak_details.csv"


def extract_gold_entities(tokens, ner_tags):
    """
    Convert BIO tags into character-offset gold entities.
    Text reconstruction uses the same space-joining method
    as the existing evaluation dataset loader.
    """

    text = " ".join(tokens)
    entities = []

    token_spans = []
    cursor = 0

    for token in tokens:
        start = cursor
        end = start + len(token)

        token_spans.append((start, end))
        cursor = end + 1

    current_type = None
    current_start = None
    current_end = None

    def save_entity():
        if current_type is not None:
            entities.append({
                "type": current_type,
                "start": current_start,
                "end": current_end,
                "text": text[current_start:current_end]
            })

    for tag, (start, end) in zip(ner_tags, token_spans):

        if tag == "O":
            save_entity()
            current_type = None
            current_start = None
            current_end = None
            continue

        prefix, entity_type = tag.split("-", 1)

        if prefix == "B":
            save_entity()

            current_type = entity_type
            current_start = start
            current_end = end

        elif prefix == "I":

            if current_type == entity_type:
                current_end = end

            else:
                # Recover safely if an I-tag appears without
                # a matching preceding B-tag.
                save_entity()

                current_type = entity_type
                current_start = start
                current_end = end

    save_entity()

    return text, entities


def classify_entity_leak(gold_entity, mapping):
    """
    Determine how much of a gold PHI entity was masked.

    Only mapping entries of the matching entity type count.
    Matching is case-insensitive.
    """

    gold_text = gold_entity["text"]
    entity_type = gold_entity["type"]

    covered = [False] * len(gold_text)

    # Only alphanumeric characters are used for coverage.
    identifying_positions = [
        index
        for index, character in enumerate(gold_text)
        if character.isalnum()
    ]

    for placeholder, original_value in mapping.items():

        match = re.fullmatch(
            r"\[([A-Z][A-Z0-9_]*)_\d+\]",
            placeholder
        )

        if not match:
            continue

        mapped_type = match.group(1)

        if mapped_type != entity_type:
            continue

        mapped_value = str(original_value)

        if not mapped_value:
            continue

        # If the mapped value contains the entire gold entity,
        # the entire entity is covered.
        if gold_text.casefold() in mapped_value.casefold():
            for position in identifying_positions:
                covered[position] = True
            continue

        # Find every occurrence of a mapped substring in the
        # gold entity and mark those characters as covered.
        for occurrence in re.finditer(
            re.escape(mapped_value),
            gold_text,
            flags=re.IGNORECASE
        ):

            for position in range(
                occurrence.start(),
                occurrence.end()
            ):
                covered[position] = True

    total = len(identifying_positions)

    if total == 0:
        return "FULLY_MASKED", ""

    covered_count = sum(
        covered[position]
        for position in identifying_positions
    )

    if covered_count == total:
        return "FULLY_MASKED", ""

    remaining = "".join(
        gold_text[position]
        for position in identifying_positions
        if not covered[position]
    )

    if covered_count == 0:
        return "FULLY_LEAKED", remaining

    return "PARTIALLY_LEAKED", remaining


def load_test_data():

    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    notes = []

    with open(TEST_FILE, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, 1):

            if not line.strip():
                continue

            row = json.loads(line)

            if len(row["tokens"]) != len(row["ner_tags"]):
                raise ValueError(
                    f"Token/tag length mismatch on line {line_number}"
                )

            text, entities = extract_gold_entities(
                row["tokens"],
                row["ner_tags"]
            )

            notes.append({
                "note_number": len(notes) + 1,
                "text": text,
                "entities": entities
            })

    return notes


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("PHI LEAK-RATE EVALUATION")
    print("=" * 65)

    notes = load_test_data()

    print("Test notes:", len(notes))

    detail_rows = []

    type_stats = defaultdict(
        lambda: {
            "gold": 0,
            "fully_masked": 0,
            "partially_leaked": 0,
            "fully_leaked": 0
        }
    )

    total_gold = 0
    total_fully_masked = 0
    total_partial = 0
    total_full_leaks = 0
    notes_with_leaks = 0

    for note in notes:

        print(
            f"\nProcessing note "
            f"{note['note_number']}/{len(notes)}"
        )

        masked_text, mapping = deidentify(note["text"])

        note_has_leak = False

        for entity in note["entities"]:

            classification, remaining = classify_entity_leak(
                entity,
                mapping
            )

            entity_type = entity["type"]

            stats = type_stats[entity_type]

            stats["gold"] += 1
            total_gold += 1

            if classification == "FULLY_MASKED":
                stats["fully_masked"] += 1
                total_fully_masked += 1

            elif classification == "PARTIALLY_LEAKED":
                stats["partially_leaked"] += 1
                total_partial += 1
                note_has_leak = True

            else:
                stats["fully_leaked"] += 1
                total_full_leaks += 1
                note_has_leak = True

            detail_rows.append({
                "note_number": note["note_number"],
                "entity_type": entity_type,
                "original_entity": entity["text"],
                "classification": classification,
                "unmasked_fragment": remaining
            })

        if note_has_leak:
            notes_with_leaks += 1

        print("Gold entities:", len(note["entities"]))
        print("Identifiers replaced:", len(mapping))
        print("Note contains leak:", note_has_leak)

    total_leaks = total_partial + total_full_leaks

    entity_leak_rate = (
        total_leaks / total_gold * 100
        if total_gold else 0
    )

    note_leak_rate = (
        notes_with_leaks / len(notes) * 100
        if notes else 0
    )

    report = {
        "experiment": "PHI Leak-Rate Evaluation",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "test_notes": len(notes),
        "total_gold_entities": total_gold,
        "fully_masked_entities": total_fully_masked,
        "partially_leaked_entities": total_partial,
        "fully_leaked_entities": total_full_leaks,
        "total_leaked_entities": total_leaks,
        "entity_leak_rate_percent": round(
            entity_leak_rate, 2
        ),
        "notes_with_at_least_one_leak": notes_with_leaks,
        "note_leak_rate_percent": round(
            note_leak_rate, 2
        ),
        "per_entity_type": {}
    }

    for entity_type, stats in sorted(type_stats.items()):

        gold = stats["gold"]
        leaks = (
            stats["partially_leaked"]
            + stats["fully_leaked"]
        )

        report["per_entity_type"][entity_type] = {
            **stats,
            "leaked_entities": leaks,
            "leak_rate_percent": round(
                leaks / gold * 100, 2
            ) if gold else 0
        }

    with open(
        JSON_OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    with open(
        CSV_OUTPUT,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "note_number",
            "entity_type",
            "original_entity",
            "classification",
            "unmasked_fragment"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(detail_rows)

    print("\n" + "=" * 65)
    print("PHI LEAK-RATE RESULTS")
    print("=" * 65)

    print("Test notes:", len(notes))
    print("Gold PHI entities:", total_gold)
    print("Fully masked:", total_fully_masked)
    print("Partially leaked:", total_partial)
    print("Fully leaked:", total_full_leaks)
    print("Total leaked:", total_leaks)

    print(
        "Entity leak rate:",
        f"{entity_leak_rate:.2f}%"
    )

    print(
        "Notes containing leaks:",
        notes_with_leaks
    )

    print(
        "Note leak rate:",
        f"{note_leak_rate:.2f}%"
    )

    print("\nPer-entity results:")

    for entity_type, stats in report[
        "per_entity_type"
    ].items():

        print(
            entity_type,
            "Gold:", stats["gold"],
            "Leaked:", stats["leaked_entities"],
            "Leak rate:", str(
                stats["leak_rate_percent"]
            ) + "%"
        )

    print("\nJSON report:", JSON_OUTPUT)
    print("CSV details:", CSV_OUTPUT)


if __name__ == "__main__":
    main()
