
import re
from collections import defaultdict

try:
    from .deidentify import PHIDetector, detect_all
except ImportError:
    from deidentify import PHIDetector, detect_all

_detector = None


def get_detector():
    """Load the model once and reuse it."""
    global _detector

    if _detector is None:
        _detector = PHIDetector()

    return _detector


def deidentify(text):
    """
    Replace detected PHI with placeholders.

    Returns:
        masked_text: Text with PHI replaced.
        mapping: Private placeholder-to-original-value dictionary.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    detector = get_detector()
    entities = detect_all(text, detector)

    counters = defaultdict(int)
    mapping = {}
    replacements = []
    repeated_values = {}

    for start, end, entity_type in entities:
        original_value = text[start:end]
        identity = (entity_type, original_value.casefold())

        if identity in repeated_values:
            placeholder = repeated_values[identity]
        else:
            counters[entity_type] += 1
            placeholder = f"[{entity_type}_{counters[entity_type]}]"
            repeated_values[identity] = placeholder
            mapping[placeholder] = original_value

        replacements.append((start, end, placeholder))

    masked_text = text

    # Replace from right to left to preserve character positions.
    for start, end, placeholder in sorted(
        replacements,
        key=lambda item: item[0],
        reverse=True
    ):
        masked_text = (
            masked_text[:start]
            + placeholder
            + masked_text[end:]
        )

    return masked_text, mapping


def rehydrate(response, mapping):
    """
    Restore only placeholders present in the private mapping.

    Unexpected placeholders remain unchanged.
    """
    if not isinstance(response, str):
        raise TypeError("response must be a string")

    if not isinstance(mapping, dict):
        raise TypeError("mapping must be a dictionary")

    restored = response

    # Replace longer placeholders first.
    for placeholder in sorted(
        mapping.keys(),
        key=len,
        reverse=True
    ):
        if not re.fullmatch(
            r"\[[A-Z][A-Z0-9_]*_\d+\]",
            placeholder
        ):
            raise ValueError("Invalid placeholder key")

        restored = restored.replace(
            placeholder,
            str(mapping[placeholder])
        )

    return restored
