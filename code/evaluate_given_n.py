# -*- coding: utf-8 -*-
"""
evaluate_given_n.py - "Given-n" evaluation on the 10 elective courses actually offered (OFFERED_ELECTIVES)

Why? With only 10 courses a student has rated most of them, so in the "one held-out course + unrated courses as
candidates" design (evaluate_rq1.py) almost no candidates are left (with at least 6 candidates: zero tests).
Given-n design: for every target student u (LOSO: u is not in the training set):
  1) n of u's rated courses are randomly "given"; the rest of u's ratings are hidden.
  2) Candidates = all non-given courses (prerequisite filter based on the student's real history).
  3) Relevant courses = hidden ones the student rated 4 or 5 (there may be several).
  4) Each method ranks the candidates; Recall@3 = relevant in the top 3 / min(3, number of relevant), NDCG@5, MRR.
  For stability every student is measured DRAWS times with different random choices and averaged; statistics are over students.
Run:  python evaluate_given_n.py form_responses_23.csv --n 3 --draws 50 --out given_out
"""
import argparse, json, math, os, itertools
import numpy as np, pandas as pd
import survey_parser as sp
from chart_loader import normalize_name as _nn
import evaluate_rq1 as e
from recommender import rank, prereq_ok

METHODS = ["Random", "GlobalMean", "Popularity", "PopSmooth", "CB", "CF", "Hybrid", "Count"]

# Count = number of training students who passed the course (baseline 'popularity by taking', ignoring satisfaction)
_orig_score_all = e.score_all
def _score_all_with_count(t, train, interests, courses):
    s = _orig_score_all(t, train, interests, courses)
    s["Count"] = {c: float(train[c].notna().sum()) for c in t["cands"]}
    return s
e.score_all = _score_all_with_count


def dcg(ranks, k):
    return sum(1 / math.log2(r + 1) for r in ranks if r <= k)


def metrics_multi(ranks, nrel, ncand):
    top3 = sum(1 for r in ranks if r <= 3)
    ndcg = dcg(ranks, 5) / sum(1 / math.log2(i + 1) for i in range(1, min(nrel, 5) + 1))
    return {"R@3": top3 / min(3, nrel), "NDCG@5": ndcg, "MRR": 1.0 / min(ranks)}


def random_expected(nrel, ncand, rng, sims=300):
    acc = {"R@3": 0, "NDCG@5": 0, "MRR": 0}
    for _ in range(sims):
        pos = rng.permutation(ncand) + 1
        rel_ranks = sorted(pos[:nrel])
        m = metrics_multi(rel_ranks, nrel, ncand)
        for k in acc: acc[k] += m[k] / sims
    return acc


def build_cases(R, years, courses, n_given, draws, seed, min_cands, passed_true, targets=None):
    targets = targets or courses     # courses that are candidates and evaluation targets; the rest of `courses` are only 'context' (their ratings stay known)
    rng = np.random.default_rng(seed)
    cases = []
    for u in R.index:
        rated = [c for c in targets if not pd.isna(R.loc[u, c])]
        if len(rated) < n_given + 1:
            continue
        term = e._entry_term(years[u])
        for _ in range(draws):
            given = list(rng.choice(rated, size=n_given, replace=False))
            hidden_rated = [c for c in rated if c not in given]
            rel = [c for c in hidden_rated if R.loc[u, c] >= 4]
            if not rel:
                continue
            cands = [c for c in targets if c not in given]
            cands = [c for c in cands if c in rel or prereq_ok(c, passed_true[u], term, e.PREREQS, e.TERMS)]
            if len(cands) < min_cands:
                continue
            ru = R.loc[u].copy()
            for c in targets:
                if c not in given: ru[c] = np.nan
            cases.append({"u": u, "given": given, "rel": rel, "cands": cands, "ru": ru, "term": term})
    return cases


def run(R, interests, years, courses, n_given, draws, seed, min_cands, targets=None):
    passed_true = {u: {_nn(c) for c in courses if not pd.isna(R.loc[u, c])} for u in R.index}
    cases = build_cases(R, years, courses, n_given, draws, seed, min_cands, passed_true, targets)
    rng = np.random.default_rng(seed + 1)
    acc = {}
    for t in cases:
        train = R.loc[[x for x in R.index if x != t["u"]]]
        sc = e.score_all(t, train, interests, courses)
        d = acc.setdefault(t["u"], {m: [] for m in METHODS})
        d["Random"].append(random_expected(len(t["rel"]), len(t["cands"]), rng))
        for m in METHODS[1:]:
            order = rank(sc[m], courses)
            ranks = sorted(order.index(c) + 1 for c in t["rel"])
            d[m].append(metrics_multi(ranks, len(t["rel"]), len(t["cands"])))
    rows = []
    for u, d in acc.items():
        for m, lst in d.items():
            rows.append({"student": u, "method": m, "n_tests": len(lst),
                         **{k: float(np.mean([x[k] for x in lst])) for k in lst[0]}})
    return cases, pd.DataFrame(rows)


def summarize(per):
    out = per.groupby("method")[["R@3", "NDCG@5", "MRR"]].agg(["mean", "std"])
    out.columns = [f"{a}_{'mean' if b == 'mean' else 'sd'}" for a, b in out.columns]
    return out.reindex(METHODS)


def boot_ci(per, method, metric, B=5000, seed=0):
    v = per[per.method == method].set_index("student")[metric].values
    rng = np.random.default_rng(seed)
    bm = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(B)]
    return float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def holm(ps):
    valid = sorted((v, k) for k, v in ps.items() if v is not None)
    m, run_, out = len(valid), 0.0, {}
    for i, (v, k) in enumerate(valid):
        run_ = max(run_, min(1.0, (m - i) * v)); out[k] = run_
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--draws", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-candidates", type=int, default=4)
    ap.add_argument("--context-all", action="store_true", help="keep the ratings of all 28 courses as context (training and profile), but candidates and targets are only the ten offered courses")
    ap.add_argument("--without-offered", action="store_true", help="like --context-all, but targets and candidates are only the 18 NOT offered courses (without the ten)")
    ap.add_argument("--out", default="given_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tidy = sp.load_survey(a.csv)
    all28 = [c for c in sp.SPECIALIZED_COURSES if c in tidy.columns]
    courses = [c for c in sp.SPECIALIZED_COURSES if c in e.OFFERED_ELECTIVES and c in tidy.columns]   # neutral order (questionnaire order) so ties do not favour any method
    R = tidy[courses].astype(float); interests, years = tidy["علاقه‌مندی‌ها"], tidy["سال_ورود"]
    targets = None
    if a.context_all:
        targets = courses; courses = all28; R = tidy[courses].astype(float)
    if a.without_offered:
        targets = [c for c in all28 if c not in courses]; courses = all28; R = tidy[courses].astype(float)
    cases, per = run(R, interests, years, courses, a.n, a.draws, a.seed, a.min_candidates, targets)
    tested = sorted({t["u"] for t in cases})
    print(f"Given-{a.n} | students: {len(R)} | students with tests: {len(tested)} | tests (draws): {len(cases)} | "
          f"mean candidates: {np.mean([len(t['cands']) for t in cases]):.1f} | mean relevant: {np.mean([len(t['rel']) for t in cases]):.2f}")
    tab = summarize(per); print(tab.round(3).to_string())
    per.to_csv(os.path.join(a.out, f"given{a.n}_per_student.csv"), index=False, encoding="utf-8-sig")
    tab.to_csv(os.path.join(a.out, f"given{a.n}_main.csv"), encoding="utf-8-sig")
    ps = {}
    pairs = [("Hybrid", "Popularity"), ("Hybrid", "CF"), ("Hybrid", "CB")]
    for x, y in pairs:
        for met in ("R@3", "NDCG@5"):
            ps[f"{x} vs {y} [{met}]"] = e.paired_p(per, x, y, met)
    h = holm(ps)
    print("\nPaired permutation test (pre-planned, 6 tests) raw p / Holm:")
    for k, v in ps.items(): print(f"  {k}: {v} / {h.get(k)}")
    ps2 = {}
    for x, y in [("CF", "Popularity"), ("CB", "Popularity")]:
        for met in ("R@3", "NDCG@5"):
            ps2[f"{x} vs {y} [{met}]"] = e.paired_p(per, x, y, met)
    h2 = holm(ps2)
    print("\nExploratory:")
    for k, v in ps2.items(): print(f"  {k}: {v} / {h2.get(k)}")
    ps3 = {}
    for x, y in [("Hybrid", "Count"), ("CF", "Count"), ("Count", "Popularity")]:
        for met in ("R@3", "NDCG@5"):
            ps3[f"{x} vs {y} [{met}]"] = e.paired_p(per, x, y, met)
    h3 = holm(ps3)
    print("\nExploratory (count-of-passers baseline):")
    for k, v in ps3.items(): print(f"  {k}: {v} / {h3.get(k)}")
    ci = {m: {met: boot_ci(per, m, met) for met in ("R@3", "NDCG@5")} for m in METHODS}
    print("\nCI 95%:"); [print(" ", m, {k: [round(x, 2) for x in v] for k, v in d.items()}) for m, d in ci.items()]
    json.dump({"n_given": a.n, "students_tested": len(tested), "cases": len(cases), "main": tab.round(4).to_dict(orient="index"),
               "p": ps, "holm": h, "p_expl": ps2, "holm_expl": h2, "p_count": ps3, "holm_count": h3, "ci": ci}, open(os.path.join(a.out, f"given{a.n}_results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
