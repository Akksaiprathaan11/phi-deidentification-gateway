
import re
import ipaddress
import json
from functools import lru_cache
from pathlib import Path


PATTERNS = {
    "EMAIL": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),

    "PHONE": re.compile(
        r"(?<!\w)\+?\d(?:[\d().\s-]{5,}\d)(?!\w)"
    ),

    "SSN": re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),

    "DATE": re.compile(
        r"\b(?:0?[1-9]|[12]\d|3[01])"
        r"[-/ ](?:0?[1-9]|1[0-2]|Jan(?:uary)?|Feb(?:ruary)?|"
        r"Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
        r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)"
        r"[-/ ,]+\d{4}\b",
        re.IGNORECASE,
    ),

    "URL": re.compile(
        r"\b(?:https?://|www\.)[^\s<>]+",
        re.IGNORECASE,
    ),

    "IP": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),

    "AGE_OVER_89": re.compile(
        r"\b(?:age\s*[:=]?\s*(9[0-9]|1[0-9]{2})"
        r"|(?:9[0-9]|1[0-9]{2})\s*(?:years?\s*old|yrs?\s*old|y/o))\b",
        re.IGNORECASE,
    ),
}


KEYED_PATTERNS = {
    "MRN": re.compile(
        r"\b(?:MRN|medical record number)\s*[:#-]?\s*[A-Z0-9-]{4,}\b",
        re.IGNORECASE,
    ),

    "HEALTH_PLAN": re.compile(
        r"\b(?:health plan|insurance ID|policy number)\s*[:#-]?\s*[A-Z0-9-]{4,}\b",
        re.IGNORECASE,
    ),

    "ACCOUNT": re.compile(
        r"\b(?:account number|account no\.?|acct)\s*[:#-]?\s*[A-Z0-9-]{4,}\b",
        re.IGNORECASE,
    ),

    "LICENSE": re.compile(
        r"\b(?:driver'?s license|license number|licence number)\s*[:#-]?\s*[A-Z0-9-]{4,}\b",
        re.IGNORECASE,
    ),
}


@lru_cache(maxsize=1)
def load_training_locations():
    """Extract unique LOCATION phrases from labelled training data."""

    dataset_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "train.jsonl"
    )

    locations = set()

    if not dataset_path.exists():
        return []

    with open(dataset_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            row = json.loads(line)
            tokens = row["tokens"]
            tags = row["ner_tags"]

            current_tokens = []

            for token, tag in zip(tokens + ["O"], tags + ["O"]):
                if tag.startswith("B-LOCATION"):
                    if current_tokens:
                        locations.add(" ".join(current_tokens))

                    current_tokens = [token]

                elif tag.startswith("I-LOCATION") and current_tokens:
                    current_tokens.append(token)

                else:
                    if current_tokens:
                        locations.add(" ".join(current_tokens))

                    current_tokens = []

            if current_tokens:
                locations.add(" ".join(current_tokens))

    return sorted(locations, key=len, reverse=True)


COMMON_LOCATIONS = [
    "Chennai",
    "Coimbatore",
    "Salem",
    "Madurai",
    "Tiruchirappalli",
    "Tirunelveli",
    "Bengaluru",
    "Hyderabad",
    "Mumbai",
    "Pune",
    "Delhi",
    "Kolkata",
    "Ahmedabad",
    "Jaipur",
    "Kochi",
]

def find_location_entities(text):
    """Find known locations and labelled training locations."""

    results = []
    locations = set(load_training_locations())
    locations.update(COMMON_LOCATIONS)

    for location in sorted(locations, key=len, reverse=True):
        pattern = (
            r"(?<!\w)"
            + r"\s+".join(
                re.escape(part) for part in location.split()
            )
            + r"(?!\w)"
        )

        for match in re.finditer(pattern, text, re.IGNORECASE):
            results.append(
                (match.start(), match.end(), "LOCATION")
            )

    return results
def find_rule_entities(text):
    """Return candidate (start, end, entity_type) spans."""

    candidates = []

    for entity_type, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            value = match.group()

            if entity_type == "PHONE":
                digit_count = sum(ch.isdigit() for ch in value)

                if not 7 <= digit_count <= 15:
                    continue

            if entity_type == "IP":
                try:
                    ipaddress.ip_address(value)
                except ValueError:
                    continue

            candidates.append(
                (match.start(), match.end(), entity_type)
            )

    for entity_type, pattern in KEYED_PATTERNS.items():
        for match in pattern.finditer(text):
            candidates.append(
                (match.start(), match.end(), entity_type)
            )

    candidates.extend(find_location_entities(text))

    return candidates
