# PHI Gateway dataset generator addition

Copy `src/generate_dataset.py` into the existing project's `src` folder and create `data/test.jsonl`.

Install Faker:
```powershell
python -m pip install Faker
```

From the project root:
```powershell
python src/generate_dataset.py --train 500 --validation 100 --seed 42
```

Review generated examples and label coverage before training. The test file is intentionally empty: populate it separately with at least 20 manually labelled notes. These synthetic templates are starter data, not a substitute for broad annotation or independent quality review.
