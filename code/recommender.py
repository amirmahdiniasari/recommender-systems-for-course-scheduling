# -*- coding: utf-8 -*-
"""
recommender.py
--------------
Three course-recommendation approaches on the survey satisfaction table (student x course, 1-5 or NaN):

  1. Content-based (CB): cosine similarity between the student's topic profile
     (course_topics.student_profile_vector: declared interests + satisfaction with taken courses) and each course's topic vector.
  2. Collaborative (CF): student-student neighbourhood; similarity = cosine over the mean-centred ratings of the common courses,
     prediction = student mean + weighted mean of the neighbours' rating deviations.
  3. Hybrid: alpha * CB + (1 - alpha) * CF, after min-max normalisation of each over the candidate list.
Plus a comparison baseline (Popularity: mean satisfaction of all students with that course).
Candidates of a student: the courses the student has not rated ("did not take").
"""
import math
import numpy as np
import pandas as pd

from course_topics import TOPICS, course_vector, student_profile_vector

MIN_COMMON = 2   # minimum number of common courses to compute the similarity of two students
K_NEIGHBORS = 5


def _cos(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def cb_scores(ratings_row, interests, candidates):
    """ratings_row: Series course -> satisfaction (NaN = not taken)."""
    rated = {c: int(v) for c, v in ratings_row.items() if not pd.isna(v)}
    prof = student_profile_vector(sorted(interests or []), rated)
    return {c: _cos(prof, course_vector(c)) for c in candidates}


def cf_scores(R, u, candidates, k=K_NEIGHBORS):
    """R: student x course DataFrame; u: row index of the target student."""
    means = R.mean(axis=1)
    mu = means[u]
    if pd.isna(mu): mu = float(np.nanmean(R.values))
    sims = {}
    ru = R.loc[u]
    for v in R.index:
        if v == u: continue
        common = [c for c in R.columns if not pd.isna(ru[c]) and not pd.isna(R.loc[v, c])]
        if len(common) < MIN_COMMON: continue
        a = np.array([ru[c] - mu for c in common]); b = np.array([R.loc[v, c] - means[v] for c in common])
        s = _cos(a, b)
        if s > 0: sims[v] = s
    top = sorted(sims.items(), key=lambda x: -x[1])[:k]
    out = {}
    for c in candidates:
        num = den = 0.0
        for v, s in top:
            if not pd.isna(R.loc[v, c]):
                num += s * (R.loc[v, c] - means[v]); den += abs(s)
        if den > 0: out[c] = mu + num / den
        else:
            col = R[c].dropna()
            out[c] = float(col.mean()) if len(col) else mu   # no neighbours: course mean
    return out


def popularity_scores(R, candidates):
    return {c: (float(R[c].mean()) if R[c].notna().any() else 0.0) for c in candidates}


def _minmax(d):
    v = list(d.values()); lo, hi = min(v), max(v)
    return {k: (x - lo) / (hi - lo) if hi > lo else 0.0 for k, x in d.items()}


def hybrid_scores(cb, cf, alpha):
    a, b = _minmax(cb), _minmax(cf)
    return {c: alpha * a[c] + (1 - alpha) * b[c] for c in cb}


# -- two-stage "preference" definition (taking + satisfaction) ---------------------------------------
# Stage 1: taking a course is itself a sign of interest (+PREF_TAKEN_BASE) and not taking it carries a soft penalty (PREF_NOT_TAKEN).
# Stage 2: the satisfaction given to a taken course raises or lowers this: low satisfaction (1-2) is a hard penalty,
#          high satisfaction (4-5) a heavy reward (PREF_SAT_ADJ). (The survey has no grades; the 1-5 satisfaction plays the role of the grade.)
# All coefficients are in one place; changing them changes every component (CB, CF, popularity, evaluation) at once.
PREF_NOT_TAKEN = -0.25
PREF_TAKEN_BASE = 1.0
PREF_SAT_ADJ = {1: -3.0, 2: -1.5, 3: 0.0, 4: 1.5, 5: 3.0}
PREF_MAX = PREF_TAKEN_BASE + max(PREF_SAT_ADJ.values())


def preference(rating):
    """Preference for one course: NaN (not taken) -> soft penalty; otherwise taking-base + satisfaction effect."""
    if rating is None or pd.isna(rating):
        return PREF_NOT_TAKEN
    return PREF_TAKEN_BASE + PREF_SAT_ADJ[int(min(5, max(1, round(float(rating)))))]


def preference_matrix(R):
    """Student x course satisfaction matrix (NaN = not taken) -> preference matrix (no NaN)."""
    return R.apply(lambda col: col.map(preference))


def cb_scores_graded(ratings_row, interests, candidates, weight_stated=2.0):
    """Like cb_scores, but the profile is built from the preference of all courses (untaken ones with the soft penalty)."""
    vec = np.zeros(len(TOPICS))
    for t in (interests or []):
        if t in TOPICS:
            vec[TOPICS.index(t)] += weight_stated
    for c, r in ratings_row.items():
        vec += np.asarray(course_vector(c), float) * preference(r)
    return {c: _cos(vec, course_vector(c)) for c in candidates}


def cf_scores_graded(R, u, candidates, k=K_NEIGHBORS):
    """CF on the preference matrix; because the matrix is full, the pattern of what students have taken also affects student similarity."""
    return cf_scores(preference_matrix(R), u, candidates, k)


def popularity_graded(R, candidates):
    """Popularity based on the mean preference (taking + satisfaction) across all students."""
    P = preference_matrix(R)
    return {c: float(P[c].mean()) for c in candidates}


def rank(scores, seed_order):
    """Descending ranking; ties are broken by a fixed order (seed_order) so the result is deterministic."""
    return sorted(scores, key=lambda c: (-scores[c], seed_order.index(c)))


# -- suggested-chart prior ------------------------------------------------------------------------
# Suggested term of each course according to the "suggested chart" (only courses that are in the survey and have a chart term);
# electives with no term in the chart get a neutral prior.
CHART_TERM = {
    "فیزیک عمومی (۲)": 4, "احتمال ۱": 5, "کامپایلر": 6,
    "شبکه‌های کامپیوتری": 7, "گرافیک کامپیوتری": 7, "زبان‌های برنامه‌سازی": 7,
    "پایگاه داده‌ها": 8, "مقدمه‌ای بر داده‌کاوی": 8, "مباحثی در علوم کامپیوتر": 8,
    "شبیه‌سازی کامپیوتری": 8, "نظریه کدگذاری": 8,
}
MAX_TERM = 8
NEUTRAL_PRIOR = 0.5


def chart_prior(course, term):
    """1 for a course the chart places exactly in this term; 1/3 less for every term of distance (minimum 0); no term in the chart: neutral."""
    t = CHART_TERM.get(course)
    if t is None: return NEUTRAL_PRIOR
    return max(0.0, 1 - abs(t - term) / 3)


def chart_weight(beta0, term):
    """Chart weight is large in the first terms and small in the last terms: beta0 * (1 - (term-1)/(MAX_TERM))."""
    return beta0 * max(0.0, 1 - (min(term, MAX_TERM) - 1) / MAX_TERM)


def chart_hybrid_scores(hybrid, term, beta0, eligible=None):
    """Final score = CB/CF combination (min-max normalised) + chart_weight(term) * chart_prior.
    eligible (optional): set of courses whose prerequisites the student has. The chart effect applies only to these;
    the others get a neutral prior (neither reward nor penalty). None = previous behaviour (the report)."""
    w = chart_weight(beta0, term)
    return {c: s + w * (chart_prior(c, term) if (eligible is None or c in eligible) else NEUTRAL_PRIOR)
            for c, s in hybrid.items()}


# -- prerequisite filter (candidates whose prerequisites the student lacks are removed) -----------------
import os, re as _re


def load_chart_info(path=None):
    """From the chart: prerequisites of every course (load_chart) and the chart term of every course (sheet "چارت پیشنهادی").
    Keys are unified with normalize_name."""
    import openpyxl
    from chart_loader import load_chart, normalize_name as nn
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "chart_computer_science.xlsx")
    df = load_chart(path)
    prereqs = {nn(r["نام درس"]): [nn(p) for p in (r["پیش‌نیاز"] or [])] for _, r in df.iterrows()}
    terms, cur = {}, None
    for row in openpyxl.load_workbook(path)["چارت پیشنهادی"].iter_rows(values_only=True):
        a = row[0]
        if not isinstance(a, str): continue
        m = _re.match(r"ترم\s*([۰-۹0-9]+)", a)
        if m: cur = int(m.group(1).translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))); continue
        if cur and a != "عنوان درس" and not a.startswith("این ترم"):
            for part in _re.split(r"\s+یا\s+", a): terms[nn(part)] = cur
    return prereqs, terms


def prereq_ok(course, passed, term, prereqs, terms):
    """Are the prerequisites of `course` satisfied?
    passed: set of (normalised) names of the survey courses the student has passed;
    a prerequisite outside the survey is assumed passed if its chart term is before `term` (or it has no chart term)."""
    from chart_loader import normalize_name as nn
    from course_topics import CHOICE_COURSES
    survey = {nn(c) for c in CHOICE_COURSES}
    for p in prereqs.get(nn(course), []):
        if p in survey:
            if p not in passed: return False
        elif p in terms and terms[p] >= term: return False
    return True


# -- smoothing a course mean towards the mean of its topic group (for courses with few ratings) -----------
def popularity_smoothed(R, candidates, m=3.0):
    """Course mean with shrinkage towards the mean of its topic group: (n*mean + m*groupmean)/(n+m)."""
    from course_topics import COURSE_TOPIC_MAP
    glob = float(np.nanmean(R.values))
    def grp(c):
        t = COURSE_TOPIC_MAP.get(c, ("", []))[1]
        return t[0] if t else "—"
    gmean = {}
    for g in {grp(c) for c in R.columns}:
        vals = np.concatenate([R[c].dropna().values for c in R.columns if grp(c) == g] or [np.array([])])
        gmean[g] = float(vals.mean()) if len(vals) else glob
    out = {}
    for c in candidates:
        col = R[c].dropna(); n = len(col)
        mu = float(col.mean()) if n else gmean.get(grp(c), glob)
        out[c] = (n * mu + m * gmean.get(grp(c), glob)) / (n + m)
    return out
