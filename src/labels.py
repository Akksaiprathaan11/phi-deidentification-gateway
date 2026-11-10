LABELS = [
    "O",
    "B-NAME", "I-NAME",
    "B-LOCATION", "I-LOCATION",
    "B-DATE", "I-DATE",
    "B-PHONE", "I-PHONE",
    "B-EMAIL", "I-EMAIL",
    "B-SSN", "I-SSN",
    "B-MRN", "I-MRN",
    "B-HEALTH_PLAN", "I-HEALTH_PLAN",
    "B-ACCOUNT", "I-ACCOUNT",
    "B-LICENSE", "I-LICENSE",
    "B-URL", "I-URL",
    "B-IP", "I-IP",
    "B-AGE_OVER_89", "I-AGE_OVER_89",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
