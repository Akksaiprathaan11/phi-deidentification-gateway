import argparse, json, random, re
from pathlib import Path
from faker import Faker

fake = Faker()
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def plain(text):
    return [(w, "O") for w in TOKEN_RE.findall(str(text))]

def entity(text, label):
    return [(w, ("B-" if i == 0 else "I-") + label)
            for i, w in enumerate(TOKEN_RE.findall(str(text)))]

def combine(*parts):
    return [item for part in parts for item in part]

def make_note(rng):
    patient, doctor = fake.name(), "Dr. " + fake.last_name()
    city, street = fake.city(), fake.street_address()
    date = fake.date_between(start_date="-8y", end_date="today").strftime("%m/%d/%Y")
    phone, email, ssn = fake.phone_number(), fake.email(), fake.ssn()
    mrn = "MRN" + str(rng.randint(100000, 999999))
    plan = "HP" + str(rng.randint(1000000, 9999999))
    account = str(rng.randint(1000000000, 9999999999))
    license_no = "LIC-" + str(rng.randint(100000, 999999))
    url, ip = "https://" + fake.domain_name() + "/patient", fake.ipv4()
    age = rng.randint(90, 104)

    templates = [
      lambda: combine(plain("Patient "), entity(patient,"NAME"), plain(" was seen by "),
        entity(doctor,"NAME"), plain(" on "), entity(date,"DATE"), plain(".")),
      lambda: combine(plain("The patient resides at "), entity(street,"LOCATION"),
        plain(" in "), entity(city,"LOCATION"), plain(". Email: "), entity(email,"EMAIL"), plain(".")),
      lambda: combine(plain("Medical record "), entity(mrn,"MRN"), plain(" belongs to "),
        entity(patient,"NAME"), plain(". Health plan: "), entity(plan,"HEALTH_PLAN"), plain(".")),
      lambda: combine(plain("Billing account "), entity(account,"ACCOUNT"),
        plain(" was reviewed. Phone: "), entity(phone,"PHONE"), plain(".")),
      lambda: combine(plain("Clinician "), entity(doctor,"NAME"), plain(" documented the visit. Licence: "),
        entity(license_no,"LICENSE"), plain(".")),
      lambda: combine(plain("Patient portal: "), entity(url,"URL"), plain(". Server address: "),
        entity(ip,"IP"), plain(".")),
      lambda: combine(plain("The patient is "), entity(age,"AGE_OVER_89"),
        plain(" years old and returned 3 days post-op. Year of review: "),
        plain(rng.randint(2018,2026)), plain(".")),
      lambda: combine(plain("Dr. "), entity("Parkinson","NAME"),
        plain(" diagnosed Parkinson's disease. Follow-up in 2 weeks.")),
      lambda: combine(plain("Social security number: "), entity(ssn,"SSN"),
        plain(". Contact "), entity(email,"EMAIL"), plain(".")),
    ]
    parts = [templates[i]() for i in rng.sample(range(len(templates)), k=rng.randint(2,4))]
    tokens = [item for part in parts for item in part]
    return {"tokens":[x[0] for x in tokens], "ner_tags":[x[1] for x in tokens]}

def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=500)
    ap.add_argument("--validation", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    Faker.seed(args.seed)
    write_jsonl("data/train.jsonl", [make_note(rng) for _ in range(args.train)])
    write_jsonl("data/validation.jsonl", [make_note(rng) for _ in range(args.validation)])
    Path("data/test.jsonl").touch(exist_ok=True)
    print(f"Created {args.train} training notes and {args.validation} validation notes.")
    print("Populate data/test.jsonl separately with 20+ manually labelled notes.")

if __name__ == "__main__":
    main()
