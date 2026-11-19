# Recommender Systems for Course Scheduling

Evaluation code for a bachelor's thesis on elective-course recommendation for university students: content-based, collaborative, hybrid, chart-prior and popularity methods, evaluated with leave-one-student-out on a small survey.

> **Main finding.** On 23 survey responses (80 tests from 12 students) no personalised method beat the plain popularity baseline, and no pairwise difference is significant after Holm correction. Mean Recall@3 across students: random 0.21, content-based 0.23, collaborative 0.40, hybrid (α=0.3) 0.41, popularity 0.51. The code is shared so the negative result can be checked and extended.

The real survey is not published (privacy). `sample_data/` holds **synthetic** random data in the same format so every script can be run. Course names are Persian because they are matched against the Persian degree chart (`code/chart_computer_science.xlsx`).

## Layout

```
code/         recommender, survey parser, degree-chart loader, evaluation scripts
sample_data/  synthetic survey responses and professor scores
results/      aggregate results (extra_results.json)
```

## Setup

```bash
pip install -r requirements.txt     # numpy, pandas, openpyxl
cd code
```

## Run

```bash
# main leave-one-student-out evaluation (permutation test, bootstrap, Holm correction)
python evaluate_rq1.py ../sample_data/form_responses_synthetic.csv --draws 30 --seed 0 --out rq1_out

# graded preference definition (satisfaction-weighted) instead of the binary one
python evaluate_rq1.py ../sample_data/form_responses_synthetic.csv --relevance graded --out rq1_graded

# parameter sensitivity, ties and collaborative-fallback diagnostics
python sweep_rq1.py ../sample_data/form_responses_synthetic.csv
python diagnostics_rq1.py ../sample_data/form_responses_synthetic.csv

# only the ~10 courses actually offered as targets, Given-n protocol
python evaluate_given_n.py ../sample_data/form_responses_synthetic.csv --n 3 --draws 50 --out given_out
python evaluate_given_n.py ../sample_data/form_responses_synthetic.csv --n 3 --context-all      # all ratings as context
python evaluate_given_n.py ../sample_data/form_responses_synthetic.csv --n 3 --without-offered # targets exclude the offered ten

# extra baselines (e.g. "count of students who passed the course")
python evaluate_extra.py ../sample_data/form_responses_synthetic.csv
```

## Protocol

- Leave-one-student-out; leak-free (the student's term is derived from the entry year, not from the held-out course).
- Metrics: Recall@3, NDCG@5, MRR; bootstrap confidence intervals; Holm-corrected permutation tests.
- Prerequisite filter from the degree chart removes courses the student is not eligible for.
- Methods (`recommender.py`): content-based (cosine on ten topic areas), user-based collaborative filtering (mean-centred), hybrid (α=0.3), hybrid + chart prior, popularity, and a count baseline.

## Limitations

- Very small sample (23 responses); results are preliminary.
- "Not taken" is treated as "not wanted", which mixes taste with availability: only about ten of the 28 surveyed electives are offered in a typical term.
- One department and one chart.

## License

MIT License – see [LICENSE](LICENSE).
