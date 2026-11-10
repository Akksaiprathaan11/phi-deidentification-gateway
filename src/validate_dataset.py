
import json
from collections import Counter
from pathlib import Path

ENTITY_TYPES = [
    "NAME", "LOCATION", "DATE", "PHONE", "EMAIL",
    "SSN", "MRN", "HEALTH_PLAN", "ACCOUNT",
    "LICENSE", "URL", "IP", "AGE_OVER_89"
]

def validate_file(filename):
    path = Path(filename)
    counts = Counter()
    errors = []
    total = 0

    if not path.exists():
        print(f"\n{filename}: FILE MISSING")
        return

    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
                tokens = row["tokens"]
                tags = row["ner_tags"]
            except (json.JSONDecodeError, KeyError) as exc:
                errors.append(f"Line {line_no}: {exc}")
                continue

            total += 1

            if len(tokens) != len(tags):
                errors.append(f"Line {line_no}: Token/tag length mismatch")
                continue

            previous = None

            for i, tag in enumerate(tags):
                if tag == "O":
                    previous = None
                    continue

                if not isinstance(tag, str) or "-" not in tag:
                    errors.append(f"Line {line_no}: Invalid tag {tag}")
                    previous = None
                    continue

                prefix, entity = tag.split("-", 1)

                if prefix not in ("B", "I") or entity not in ENTITY_TYPES:
                    errors.append(f"Line {line_no}: Unknown tag {tag}")
                    previous = None
                    continue

                if prefix == "I" and previous != entity:
                    errors.append(
                        f"Line {line_no}, token {i}: Invalid BIO sequence"
                    )

                if prefix == "B":
                    counts[entity] += 1

                previous = entity

    print(f"\nDataset: {filename}")
    print("Total notes:", total)
    print("Entity counts:", dict(counts))
    print("Missing categories:",
          sorted(set(ENTITY_TYPES) - set(counts)))
    print("Annotation errors:", len(errors))

    for error in errors[:10]:
        print(error)


if __name__ == "__main__":
    for filename in [
        "data/train.jsonl",
        "data/validation.jsonl",
        "data/test.jsonl"
    ]:
        validate_file(filename)
