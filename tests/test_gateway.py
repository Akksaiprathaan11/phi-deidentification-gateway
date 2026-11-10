
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

import gateway
from rules import find_rule_entities


class FakeDetector:
    """Mock detector so gateway behaviour can be tested independently."""

    def predict_entities(self, text):
        return []


def test_email_detection():
    text = "Contact patient@example.com for details."
    results = find_rule_entities(text)

    assert any(entity[2] == "EMAIL" for entity in results)


def test_phone_detection():
    text = "Call +1 212 555 0198."
    results = find_rule_entities(text)

    assert any(entity[2] == "PHONE" for entity in results)


def test_ssn_detection():
    text = "SSN: 123-45-6789."
    results = find_rule_entities(text)

    assert any(entity[2] == "SSN" for entity in results)


def test_full_date_detection():
    text = "Appointment date is 12 May 2026."
    results = find_rule_entities(text)

    assert any(entity[2] == "DATE" for entity in results)


def test_year_only_is_not_masked_by_date_rule():
    text = "The patient was treated in 2026."
    results = find_rule_entities(text)

    assert not any(entity[2] == "DATE" for entity in results)


def test_age_above_89():
    text = "The patient is 92 years old."
    results = find_rule_entities(text)

    assert any(entity[2] == "AGE_OVER_89" for entity in results)


def test_gateway_round_trip(monkeypatch):
    monkeypatch.setattr(
        gateway,
        "get_detector",
        lambda: FakeDetector()
    )

    original = (
        "Contact patient@example.com about the appointment "
        "on 12 May 2026."
    )

    masked, mapping = gateway.deidentify(original)

    assert "patient@example.com" not in masked
    assert "12 May 2026" not in masked
    assert mapping

    restored = gateway.rehydrate(masked, mapping)

    assert restored == original


def test_unknown_placeholder_is_not_replaced():
    response = "Patient [NAME_99] needs review."
    restored = gateway.rehydrate(response, {})

    assert restored == response


def test_repeated_email_uses_same_placeholder(monkeypatch):
    monkeypatch.setattr(
        gateway,
        "get_detector",
        lambda: FakeDetector()
    )

    original = (
        "Email patient@example.com or "
        "patient@example.com again."
    )

    masked, mapping = gateway.deidentify(original)

    assert masked.count("[EMAIL_1]") == 2
    assert len(mapping) == 1

    assert gateway.rehydrate(masked, mapping) == original


def test_parkinsons_disease_is_not_a_rule_match():
    text = "The patient was diagnosed with Parkinson's disease."
    results = find_rule_entities(text)

    assert not any(
        entity[2] == "NAME"
        for entity in results
    )
