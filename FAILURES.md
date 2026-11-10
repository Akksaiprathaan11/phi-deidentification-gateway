# Failure Analysis – PHI De-identification Gateway

## 1. Purpose

This document records the issues, limitations, evaluation failures, and improvements identified during the development of the PHI De-identification Gateway.

The purpose is to maintain transparency about the system's current performance and identify areas requiring further development.

All experiments described here use synthetic healthcare-style data.

## 2. Model Generalisation Issues

### Observation

During training, the DistilBERT model achieved a validation F1-score of 1.0 after five epochs.

However, the independent evaluation on 20 manually labelled test notes produced significantly lower results.

### Independent Test Results

| Metric | Result |
|---|---:|
| True positives | 19 |
| False positives | 48 |
| False negatives | 42 |
| Precision | 28.36% |
| Recall | 31.15% |
| F1-score | 29.69% |

### Per-Entity Results

| Entity | Precision | Recall | F1-score |
|---|---:|---:|---:|
| NAME | 12.82% | 25.00% | 16.95% |
| LOCATION | 100.00% | 33.33% | 50.00% |
| DATE | 35.00% | 35.00% | 35.00% |

### Analysis

The difference between validation and test performance indicates that the model did not generalise well to the independent test examples.

Name recognition produced particularly weak results, while location recall and date recognition also require improvement.

### Planned Improvements

- Increase the diversity of training examples.
- Review annotation consistency.
- Add more variations of names, locations, and date formats.
- Evaluate alternative model configurations.
- Test on a larger independent dataset.

## 3. Initial Location Detection Failures

### Observation

The initial PHI leak-rate evaluation identified seven leaked entities among 61 gold entities.

Most failures involved locations that were not recognised by the original location-detection implementation.

### Initial Failure Examples

| Location | Initial result |
|---|---|
| New Delhi | Partially leaked |
| Chandigarh | Fully leaked |
| Lucknow | Fully leaked |
| Bhopal | Fully leaked |
| Surat | Fully leaked |
| Nagpur | Fully leaked |
| Bhubaneswar | Fully leaked |

### Initial Leak-Rate Results

| Metric | Result |
|---|---:|
| Test notes | 20 |
| Gold entities | 61 |
| Fully masked | 54 |
| Partially leaked | 1 |
| Fully leaked | 6 |
| Total leaked | 7 |
| Entity leak rate | 11.48% |
| Notes containing leaks | 7 |
| Note leak rate | 35.00% |

### Root Cause

The initial implementation relied heavily on the location information available in the synthetic training data.

Unseen locations were not always recognised, resulting in incomplete masking.

### Improvement Implemented

The location-detection component was enhanced using spaCy named-entity recognition alongside the training-data location gazetteer.

The updated implementation recognises location-related entity labels such as GPE, LOC, and FAC.

Diagnostic checks were performed using the previously missed location examples.

## 4. Post-Fix Leak-Rate Evaluation

The evaluation was repeated after the location-detection improvement.

### Results

| Metric | Initial baseline | Post-fix |
|---|---:|---:|
| Gold entities | 61 | 61 |
| Fully masked | 54 | 61 |
| Partially leaked | 1 | 0 |
| Fully leaked | 6 | 0 |
| Total leaked | 7 | 0 |
| Entity leak rate | 11.48% | 0.00% |
| Notes containing leaks | 7 | 0 |
| Note leak rate | 35.00% | 0.00% |

### Analysis

The updated run reported that all 61 labelled entities were fully masked.

No partially leaked or fully leaked entities were reported.

This indicates an improvement in the location-detection behaviour on the evaluated examples.

### Important Evaluation Limitation

The post-fix evaluation was performed on the same test set that had been inspected to identify the initial location failures.

Therefore, this result is an exploratory post-fix evaluation and not an independent measurement of generalisation.

Additionally, the current leak-evaluation implementation relies on the private replacement mapping to determine masking status. It does not independently verify every original entity position against the actual masked output.

A stronger evaluator should inspect the actual masked text for residual sensitive characters and substrings.

A new independently annotated test set is required before making a stronger claim about leak prevention.

## 5. Presidio Comparison

### Results

| Metric | DistilBERT | Presidio |
|---|---:|---:|
| Precision | 28.36% | 98.31% |
| Recall | 31.15% | 95.08% |
| F1-score | 29.69% | 96.67% |

### Analysis

Presidio achieved substantially higher performance on the current synthetic test set.

This suggests that the current custom DistilBERT model requires further improvement before it can serve as a reliable standalone PHI detector.

The comparison is limited by the small and repetitive test dataset, which contains only NAME, LOCATION, and DATE as gold categories.

The two implementations also use different span representations in their evaluation pipelines. Therefore, the comparison should be treated as a useful prototype-level comparison rather than a definitive benchmark.

## 6. Language Model Utility Issues

### Observation

The utility experiment compared Gemini responses generated from five original synthetic notes and their corresponding masked versions.

The experiment used `gemini-3.5-flash-lite` after repeated service-overload errors with the initially selected model.

### Issues Observed

**1. Placeholder interpretation**

Some responses changed the wording or relationships between placeholders.

**2. Partial location masking**

One masked example represented New Delhi as:

```text
New [LOCATION_1]
```

This showed that part of the original location remained in the text.

**3. Response variation**

The model sometimes changed sentence structure while processing masked text.

**4. Manual scoring**

The scoring CSV was created, but the final manual utility scores require confirmation.

### Planned Improvements

- Improve complete entity-boundary recognition.
- Validate the masked text before external submission.
- Check that placeholders remain intact.
- Complete manual scoring of all five note pairs.
- Compare factual preservation and hallucination behaviour systematically.

## 7. Placeholder and Rehydration Limitations

### Observation

The gateway uses placeholders such as:

```text
[NAME_1]
[LOCATION_1]
[DATE_1]
```

An external LLM may modify or remove placeholders while generating a response.

### Potential Risks

- A placeholder may be removed.
- A placeholder may be duplicated.
- An unexpected placeholder may appear.
- The model may change placeholder spelling.
- The generated response may change the relationship between two entities.

### Planned Improvements

- Validate placeholders before rehydration.
- Detect missing and unexpected placeholders.
- Add more tests for malformed responses.
- Prevent unsafe or unintended substitutions.
- Review the final response before returning it.

## 8. Dataset Limitations

The current dataset contains:

| Dataset | Notes |
|---|---:|
| Training | 500 |
| Validation | 100 |
| Test | 20 |
| Total | 620 |

The test dataset contains 61 manually labelled entities:

- NAME: 20
- LOCATION: 21
- DATE: 20

The remaining supported entity categories do not have gold test examples.

Therefore, the current evaluation does not establish the detection performance of all 13 supported entity categories.

### Planned Improvements

- Generate additional synthetic examples.
- Include more diverse sentence structures.
- Add examples for every supported entity type.
- Create an independently annotated test set.
- Include more complex healthcare-style text.

## 9. Automated Testing

The current automated test suite contains ten tests.

The latest recorded test execution reported:

```text
10 passed in 13.18s
```

The tests cover:

- Email detection.
- Phone number detection.
- SSN detection.
- Date detection.
- Year-only handling.
- Age above 89.
- Round-trip restoration.
- Unknown placeholder handling.
- Repeated identifier consistency.
- Parkinson disease and doctor-name ambiguity.

### Remaining Testing Gaps

The current test suite does not provide exhaustive coverage of all supported PHI categories, all possible entity boundaries, or all external LLM response behaviours.

Additional integration, edge-case, and security tests are required.

## 10. Security and Privacy Limitations

Although the prototype maintains the replacement mapping locally, additional safeguards are necessary for production use.

Current limitations include:

- No complete production-grade encrypted mapping storage.
- No comprehensive access-control implementation.
- No formal secure deletion process.
- No production security audit.
- No independent residual-PHI verification.
- No certification for healthcare privacy compliance.

The system should only be used with synthetic data in its current assessment configuration.

## 11. Summary of Known Issues

| Area | Observed issue | Current status |
|---|---|---|
| DistilBERT generalisation | Low independent test F1 | Requires improvement |
| Name detection | Low precision and recall | Requires improvement |
| Initial location detection | Seven leaked entities | Improved in post-fix run |
| Leak-rate validation | Same test set used for post-fix | Requires independent evaluation |
| Leak evaluator | Relies on private mapping | Requires actual-output verification |
| Presidio comparison | Different span representations | Comparison caveat documented |
| LLM utility | Placeholder and wording variation | Requires further evaluation |
| Utility scoring | Manual scores not finalised | Pending review |
| Dataset coverage | Only three test entity categories | Requires expansion |
| Production security | Additional safeguards needed | Not production-ready |

## 12. Conclusion

The failure analysis helped identify weaknesses that were not visible during the initial model validation.

The most important issues were poor DistilBERT generalisation, incomplete location recognition, and the need for stronger output-level privacy verification.

The location-detection improvement produced a zero-leak result on the same 20-note test set, which is an encouraging exploratory result. However, further independent evaluation is necessary.

This document records both the improvements made and the limitations that remain. The objective is to maintain an honest record of the system's current capabilities and provide a clear direction for future development.
