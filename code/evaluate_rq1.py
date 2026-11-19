# -*- coding: utf-8 -*-
"""
evaluate_rq1.py  -  RQ1 evaluation (quality of course recommendations), leave-one-student-out (LOSO)
------------------------------------------------------------------------------------
Question: does the hybrid model (CB+CF), or the model with the degree-chart prior, rank a course the
held-out student liked higher than the baselines do?

Protocol (leak-free):
  For every target student u:
    1) The "training set" is all OTHER students (u is not in it). Course popularity and the CF neighbours
       are built from these students only.
    2) From u's own ratings, one liked course (satisfaction >= 4) is held out (one per test); the rest of
       u's ratings and u's declared interests are the model's input.
    3) Candidates = courses u has not rated + the held-out course (after the prerequisite filter).
       The model ranks the candidates.
  Each student averages their own tests; the final result is the mean +/- SD "across students".
  (With exactly one relevant course per test: Recall@K = HitRate@K, Precision@K = Hit/K, F1 from these two.)

Learning curve: the training-set size is set to 5/10/15/20/... students (random subset of the other
students, several draws) and quality is measured, to see whether more data makes the model better.

Run:
    python evaluate_rq1.py form_responses.csv
    python evaluate_rq1.py form_responses.csv --no-straightliners --draws 30 --seed 0 --out rq1_out
Output: rq1_out/rq1_results.json, rq1_main.csv, rq1_learning_curve.csv, rq1_per_student.csv
"""
import argparse, json, math, os, random
import numpy as np
import pandas as pd

import survey_parser as sp
from chart_loader import normalize_name as _nn
from recommender import (cb_scores, cf_scores, popularity_scores, popularity_smoothed, hybrid_scores, rank,
                         chart_hybrid_scores, CHART_TERM, MAX_TERM, load_chart_info, prereq_ok,
                         cb_scores_graded, cf_scores_graded, popularity_graded, preference, PREF_MAX)

KS = (3, 5)
MIN_CANDIDATES = 6      # a test with fewer candidates than this is dropped (otherwise random guessing also succeeds)
ALPHA = 0.3             # weight of CB in the hybrid
BETA = 1.0              # base weight of the chart prior
# the 10 electives actually offered in terms 4041, 4042, 4051 (and the summers); the other 18 surveyed courses are not offered
OFFERED_ELECTIVES = ["شبکه‌های کامپیوتری", "زبان‌های برنامه‌سازی", "گرافیک کامپیوتری", "پایگاه داده‌ها", "مباحثی در علوم کامپیوتر",
                     "کامپایلر", "شبیه‌سازی کامپیوتری", "احتمال ۱", "مقدمه‌ای بر داده‌کاوی", "فیزیک عمومی (۲)"]
METHODS = ["Random", "GlobalMean", "Popularity", "PopSmooth", "CB", "CF", "Hybrid", "Chart"]
PREREQS, TERMS = load_chart_info()
RELEVANCE = "binary"    # binary: relevant = satisfaction 4 or 5 (the report); graded: two-stage definition 'taken + satisfaction' (recommender.preference)


def straightliners(R, frac=0.9, min_rated=10):
    out = []
    for u in R.index:
        v = R.loc[u].dropna()
        if len(v) >= min_rated and v.value_counts().iloc[0] / len(v) >= frac:
            out.append(u)
    return out


def _entry_term(year_cell):
    digits = "".join(ch for ch in str(year_cell) if ch.isdigit())
    return min(MAX_TERM, max(1, (1405 - int(digits)) * 2 + 1)) if digits else MAX_TERM


def make_tests(R, interests, years, courses, term_mode="entry", relevance="binary"):
    """List of tests: (student, held-out course, term, candidates). Independent of the model and of the training set,
    so all methods and all training sizes are measured on exactly the same tests.
    relevance="binary": only courses with satisfaction >= 4 are held out (the report's definition).
    relevance="graded": every taken course is held out; satisfaction >= 3 is a positive test (weight = preference / max preference)
    and satisfaction 1-2 is a negative test (measures how often the method puts a disliked course in the top 3)."""
    tests = []
    for u in R.index:
        if relevance == "graded":
            held_list = [(c, R.loc[u, c]) for c in courses if not pd.isna(R.loc[u, c])]
        else:
            held_list = [(c, R.loc[u, c]) for c in courses if not pd.isna(R.loc[u, c]) and R.loc[u, c] >= 4]
        for held, rv in held_list:
            ru = R.loc[u].copy(); ru[held] = np.nan
            if ru.notna().sum() < 1:
                continue
            # term_mode="entry" (leak-free): the term comes from the student's entry year, independent of the held-out course.
            # term_mode="held" (old, leaky method): the chart term of the held-out course itself -> only for comparison with the earlier result.
            term = (CHART_TERM.get(held) or _entry_term(years[u])) if term_mode == "held" else _entry_term(years[u])
            passed = {_nn(c) for c in courses if not pd.isna(ru[c])}
            cands = [c for c in courses if pd.isna(ru[c])]
            cands = [c for c in cands if c == held or prereq_ok(c, passed, term, PREREQS, TERMS)]
            if len(cands) < MIN_CANDIDATES:
                continue
            t = {"u": u, "held": held, "term": term, "cands": cands, "ru": ru}
            if relevance == "graded":
                t["neg"] = bool(rv <= 2)
                t["w"] = max(0.0, preference(rv)) / PREF_MAX
            tests.append(t)
    return tests


def score_all(test, train, interests, courses):
    """Scores of all methods for one test. train = DataFrame of ratings of the OTHER students (without u)."""
    u, cands, ru = test["u"], test["cands"], test["ru"]
    # table for CF: training students + row u (held-out course = NaN). u only enters as the 'query'.
    R2 = pd.concat([train, ru.to_frame(name=u).T])
    graded = RELEVANCE == "graded"
    cb = cb_scores_graded(ru, interests[u], cands) if graded else cb_scores(ru, interests[u], cands)
    cf = cf_scores_graded(R2, u, cands) if graded else cf_scores(R2, u, cands)
    hyb = hybrid_scores(cb, cf, ALPHA)
    glob = float(np.nanmean(train.values)) if train.size else 0.0
    return {
        "GlobalMean": {c: glob for c in cands},                       # all equal -> fixed order (uninformed guess)
        "Popularity": popularity_graded(train, cands) if graded else popularity_scores(train, cands),
        "PopSmooth": popularity_graded(train, cands) if graded else popularity_smoothed(train, cands),   # in graded mode this equals Popularity
        "CB": cb, "CF": cf, "Hybrid": hyb,
        "Chart": chart_hybrid_scores(hyb, test["term"], BETA),
    }


def metrics_from_rank(r, n_cands):
    """r = rank of the held-out course (starting at 1). One relevant course -> Precision@K = hit/K, Recall@K = hit."""
    m = {"MRR": 1.0 / r}
    for k in KS:
        hit = 1.0 if r <= k else 0.0
        p, rc = hit / k, hit
        m[f"P@{k}"], m[f"R@{k}"] = p, rc
        m[f"F1@{k}"] = 2 * p * rc / (p + rc) if p + rc else 0.0
        m[f"NDCG@{k}"] = 1 / math.log2(r + 1) if r <= k else 0.0
    return m


def random_metrics(n):
    """Expected value of random guessing (exact, no sampling): every rank 1..n is equally likely."""
    acc = {}
    for r in range(1, n + 1):
        for k, v in metrics_from_rank(r, n).items():
            acc[k] = acc.get(k, 0.0) + v / n
    return acc


def augment_rows(train, k, seed, flip_p=0.2, drop_p=0.15):
    """k synthetic copies of every TRAINING student (never of the test student): each rating moves one step up/down
    with probability flip_p (within 1..5) and is dropped with probability drop_p. They are only added to the training set;
    the test is always on a real student."""
    rng = np.random.default_rng(seed)
    rows, names = [], []
    for u in train.index:
        base = train.loc[u].values.astype(float)
        for j in range(k):
            v = base.copy()
            for i in range(len(v)):
                if np.isnan(v[i]):
                    continue
                if rng.random() < drop_p:
                    v[i] = np.nan
                elif rng.random() < flip_p:
                    v[i] = min(5.0, max(1.0, v[i] + rng.choice([-1.0, 1.0])))
            if np.isfinite(v).sum() >= 2:
                rows.append(v); names.append(f"{u}_aug{j}")
    if not rows:
        return train
    return pd.concat([train, pd.DataFrame(rows, index=names, columns=train.columns)])


def evaluate(tests, pool_ids, R, interests, courses, rng=None, train_size=None, aug=0, seed=0):
    """Per-student mean. train_size=None means all other students; otherwise a random subset."""
    per = {}   # (u) -> method -> list of metric dicts
    negs = {}
    for t in tests:
        others = [x for x in pool_ids if x != t["u"]]
        if train_size is not None:
            if train_size > len(others):
                return None
            others = rng.sample(others, train_size)
        train = R.loc[others]
        if aug > 0:
            train = augment_rows(train, aug, seed + hash(str(t['u'])) % 10007)
        sc = score_all(t, train, interests, courses)
        d = per.setdefault(t["u"], {m: [] for m in METHODS})
        if t.get("neg"):
            # negative test (graded only): the held-out disliked course should not be in the top 3 (FP@3; lower is better)
            nd = negs.setdefault(t["u"], {m: [] for m in METHODS})
            n_ = len(t["cands"])
            nd["Random"].append(min(3, n_) / n_)
            for m, s in sc.items():
                nd[m].append(1.0 if rank(s, courses).index(t["held"]) + 1 <= 3 else 0.0)
            continue
        rm = random_metrics(len(t["cands"]))
        if "w" in t: rm["WR@3"] = rm["R@3"] * t["w"]
        d["Random"].append(rm)
        for m, s in sc.items():
            r = rank(s, courses).index(t["held"]) + 1
            mm = metrics_from_rank(r, len(t["cands"]))
            if "w" in t: mm["WR@3"] = mm["R@3"] * t["w"]     # Recall@3 weighted by the preference of the held-out course
            d[m].append(mm)
    # per-student mean
    rows = []
    for u, d in per.items():
        for m, lst in d.items():
            if not lst:
                continue      # a student who only has negative tests
            row = {"student": u, "method": m, "n_tests": len(lst)}
            for k in lst[0]:
                row[k] = float(np.mean([x[k] for x in lst]))
            if u in negs and negs[u][m]:
                row["FP@3"] = float(np.mean(negs[u][m]))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(per_student_df):
    cols = [c for c in per_student_df.columns if c not in ("student", "method", "n_tests")]
    g = per_student_df.groupby("method")[cols]
    mean, sd = g.mean(), g.std(ddof=1)
    out = pd.DataFrame(index=METHODS)
    for c in cols:
        out[c + "_mean"] = mean[c].reindex(METHODS)
        out[c + "_sd"] = sd[c].reindex(METHODS)
    return out


def paired_p(per_student_df, a, b, metric, seed=0):
    """Paired two-sided sign-flip permutation test on the per-student means; no scipy needed.
    Exact up to 16 students (all 2^n sign patterns), otherwise 20,000 random permutations."""
    import itertools
    x = per_student_df[per_student_df.method == a].set_index("student")[metric]
    y = per_student_df[per_student_df.method == b].set_index("student")[metric]
    d = (x - y.reindex(x.index)).dropna().values
    n = len(d)
    if n < 5 or np.allclose(d, 0):
        return None
    obs = abs(d.mean())
    if n <= 16:
        signs = np.array(list(itertools.product([1, -1], repeat=n)))
    else:
        signs = np.random.default_rng(seed).choice([1, -1], size=(20000, n))
    return float((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--no-straightliners", action="store_true", help="drop straight-line respondents (>=90%% the same number)")
    ap.add_argument("--draws", type=int, default=30, help="number of random training-set draws in the learning curve")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", default="5,10,15,20", help="training-set sizes for the learning curve")
    ap.add_argument("--augment", type=int, default=0, help="number of synthetic copies per training student (0 = no augmentation); only enlarges the training set")
    ap.add_argument("--term-mode", choices=["entry", "held"], default="entry",
                    help="entry (default, leak-free): term from the entry year; held: chart term of the held-out course itself (leaky, for comparison only)")
    ap.add_argument("--relevance", choices=["binary", "graded"], default="binary",
                    help="binary (report): relevant = satisfaction 4 or 5. graded: two-stage preference (taken + satisfaction, recommender.preference); "
                         "besides Recall/NDCG it also reports WR@3 (weighted) and FP@3 (disliked course in the top 3)")
    ap.add_argument("--offered-only", action="store_true", help="evaluate only the 10 actually offered courses (OFFERED_ELECTIVES)")
    ap.add_argument("--min-candidates", type=int, default=None, help="minimum number of candidates per test (default 6)")
    ap.add_argument("--out", default="rq1_out")
    a = ap.parse_args()
    global RELEVANCE, MIN_CANDIDATES
    RELEVANCE = a.relevance
    if a.min_candidates is not None:
        MIN_CANDIDATES = a.min_candidates
    os.makedirs(a.out, exist_ok=True)

    tidy = sp.load_survey(a.csv)
    courses = [c for c in sp.SPECIALIZED_COURSES if c in tidy.columns]
    if a.offered_only:
        courses = [c for c in sp.SPECIALIZED_COURSES if c in OFFERED_ELECTIVES and c in tidy.columns]   # neutral order (questionnaire order)
    R = tidy[courses].astype(float)
    interests, years = tidy["علاقه‌مندی‌ها"], tidy["سال_ورود"]
    flagged = straightliners(R)
    if a.no_straightliners:
        keep = [u for u in R.index if u not in flagged]
        R, interests, years = R.loc[keep], interests.loc[keep], years.loc[keep]
    pool_ids = list(R.index)

    tests = make_tests(R, interests, years, courses, a.term_mode, a.relevance)
    tested = sorted({t["u"] for t in tests})
    print(f"Students: {len(R)} | straight-liners: {flagged}{' (dropped)' if a.no_straightliners else ''}")
    print(f"Tests: {len(tests)} on {len(tested)} students | mean number of candidates: {np.mean([len(t['cands']) for t in tests]):.1f}")

    # -- main result: all other students are the training set --
    per = evaluate(tests, pool_ids, R, interests, courses)
    per.to_csv(os.path.join(a.out, "rq1_per_student.csv"), index=False, encoding="utf-8-sig")
    main_tab = summarize(per)
    main_tab.to_csv(os.path.join(a.out, "rq1_main.csv"), encoding="utf-8-sig")
    show = ["R@3_mean", "R@3_sd", "NDCG@5_mean", "NDCG@5_sd", "F1@3_mean", "MRR_mean"]
    if a.relevance == "graded":
        show += ["WR@3_mean", "FP@3_mean"]
        nneg = sum(1 for t in tests if t.get("neg"))
        print(f"relevance=graded: {len(tests) - nneg} positive tests (satisfaction >= 3) + {nneg} negative tests (satisfaction 1-2)")
    print("\n=== Main result (mean +/- SD across students) ===")
    print(main_tab[show].round(3).to_string())

    pvals = {}
    for other in ("Popularity", "CF", "CB"):
        for tgt in ("Hybrid", "Chart"):
            for met in ("R@3", "NDCG@5"):
                pvals[f"{tgt} vs {other} [{met}]"] = paired_p(per, tgt, other, met)
    # Holm-Bonferroni multiple-comparison correction over all the p-values above
    valid = sorted([(v, k) for k, v in pvals.items() if v is not None])
    m_ = len(valid); holm = {}; run = 0.0
    for i, (v, k) in enumerate(valid):
        run = max(run, min(1.0, (m_ - i) * v)); holm[k] = run
    print(f"\nPaired permutation test (raw p and Holm-corrected p; term mode: {a.term_mode}):")
    for k, v in pvals.items():
        print(f"  {k}: {'—' if v is None else round(v, 3)}   Holm: {'—' if k not in holm else round(holm[k], 3)}")


    # -- ablation: augmentation on the training set only --
    if a.augment > 0:
        per_a = evaluate(tests, pool_ids, R, interests, courses, aug=a.augment, seed=a.seed)
        per_a.to_csv(os.path.join(a.out, "rq1_per_student_aug.csv"), index=False, encoding="utf-8-sig")
        tab_a = summarize(per_a)
        tab_a.to_csv(os.path.join(a.out, "rq1_main_aug.csv"), encoding="utf-8-sig")
        chk = augment_rows(R, a.augment, a.seed)
        print(f"\n=== With augmentation x{a.augment} (training only; tests on real students) ===")
        print(f"Sanity check: overall mean rating real {np.nanmean(R.values):.3f} / synthetic {np.nanmean(chk.drop(index=R.index).values):.3f}; "
              f"mean absolute difference of per-course means: {float((chk.drop(index=R.index).mean() - R.mean()).abs().mean()):.3f}")
        cmp_ = pd.DataFrame({"R@3_real": main_tab["R@3_mean"], "R@3_aug": tab_a["R@3_mean"],
                             "NDCG@5_real": main_tab["NDCG@5_mean"], "NDCG@5_aug": tab_a["NDCG@5_mean"]})
        print(cmp_.round(3).to_string())

    # -- learning curve --
    sizes = [int(x) for x in a.sizes.split(",")] + [len(R) - 1]
    sizes = sorted(set(s for s in sizes if s <= len(R) - 1))
    rng = random.Random(a.seed)
    curve = []
    for n in sizes:
        draws = 1 if n == len(R) - 1 else a.draws          # no repeats needed when using all other students
        accum = []
        for _ in range(draws):
            p = evaluate(tests, pool_ids, R, interests, courses, rng=rng, train_size=n)
            if p is not None:
                accum.append(summarize(p)[["R@3_mean", "NDCG@5_mean"]])
        if not accum:
            continue
        mean_tab = sum(accum) / len(accum)
        sd_tab = pd.concat(accum).groupby(level=0).std(ddof=1) if len(accum) > 1 else None
        for m in METHODS:
            row = {"train_size": n, "method": m, "draws": len(accum),
                   "R@3": float(mean_tab.loc[m, "R@3_mean"]), "NDCG@5": float(mean_tab.loc[m, "NDCG@5_mean"])}
            if sd_tab is not None:
                row["R@3_sd_over_draws"] = float(sd_tab.loc[m, "R@3_mean"])
                row["NDCG@5_sd_over_draws"] = float(sd_tab.loc[m, "NDCG@5_mean"])
            curve.append(row)
    cdf = pd.DataFrame(curve)
    cdf.to_csv(os.path.join(a.out, "rq1_learning_curve.csv"), index=False, encoding="utf-8-sig")
    print("\n=== Learning curve: Recall@3 (mean over draws) ===")
    print(cdf.pivot(index="train_size", columns="method", values="R@3")[METHODS].round(3).to_string())

    json.dump({"n_students": len(R), "n_students_tested": len(tested), "n_cases": len(tests),
               "straightliners": [int(x) for x in flagged], "dropped_straightliners": a.no_straightliners,
               "alpha": ALPHA, "beta": BETA, "seed": a.seed, "draws": a.draws,
               "main": main_tab.round(4).to_dict(orient="index"), "wilcoxon_p": pvals, "holm_p": holm, "term_mode": a.term_mode, "relevance": a.relevance,
               "learning_curve": curve},
              open(os.path.join(a.out, "rq1_results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=2)
    print(f"\nFiles saved in {a.out}/.")


if __name__ == "__main__":
    main()
