
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from gateway import deidentify, rehydrate

original = """
Patient John Smith, age 92, visited Chennai on 12 May 2026.
Contact: john.smith@example.com.
The patient was diagnosed with Parkinson's disease.
"""

print("\n--- ORIGINAL NOTE ---")
print(original)

masked_text, mapping = deidentify(original)

print("\n--- MASKED NOTE ---")
print(masked_text)

print("\n--- PRIVATE MAPPING (LOCAL DEMO ONLY) ---")
print(mapping)

# Simulated LLM response: no actual LLM call is made.
simulated_response = (
    "Patient [NAME_1] attended an appointment. "
    "The appointment date was [DATE_1]. "
)

restored_response = rehydrate(simulated_response, mapping)

print("\n--- REHYDRATED RESPONSE ---")
print(restored_response)
