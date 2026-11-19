# -*- coding: utf-8 -*-
"""
recommender.py
--------------
سه رویکرد پیشنهاد درس روی جدول رضایت پرسش‌نامه (دانشجو × درس، ۱ تا ۵ یا NaN):

  ۱. مبتنی بر محتوا (CB): شباهت کسینوسی بین «پروفایل موضوعی دانشجو» (course_topics.student_profile_vector:
     علاقه‌ی اعلام‌شده + رضایت از درس‌های گذرانده) و بردار موضوعی هر درس.
  ۲. مبتنی بر همکاری (CF): همسایگی دانشجو–دانشجو؛ شباهت = کسینوس روی رتبه‌های میانگین‌صفرشده‌ی درس‌های
     مشترک، پیش‌بینی = میانگین دانشجو + میانگین وزنی انحراف رتبه‌ی همسایه‌ها.
  ۳. ترکیبی: alpha * CB + (1 - alpha) * CF، پس از نرمال‌سازی min-max هر کدام روی فهرست کاندیدها.
به‌علاوه‌ی یک پایه‌ی مقایسه (Popularity: میانگین رضایت همه‌ی دانشجوها از آن درس).
کاندیدهای هر دانشجو: درس‌هایی که برایشان رتبه‌ای نداده («نگذراندم»).
"""
import math
import numpy as np
import pandas as pd

from course_topics import TOPICS, course_vector, student_profile_vector

MIN_COMMON = 2   # حداقل درس مشترک برای محاسبه‌ی شباهت دو دانشجو
K_NEIGHBORS = 5


def _cos(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def cb_scores(ratings_row, interests, candidates):
    """ratings_row: Series درس->رضایت (NaN = نگذرانده)."""
    rated = {c: int(v) for c, v in ratings_row.items() if not pd.isna(v)}
    prof = student_profile_vector(sorted(interests or []), rated)
    return {c: _cos(prof, course_vector(c)) for c in candidates}


def cf_scores(R, u, candidates, k=K_NEIGHBORS):
    """R: DataFrame دانشجو×درس؛ u: اندیس سطر دانشجوی هدف."""
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
            out[c] = float(col.mean()) if len(col) else mu   # بدون همسایه: میانگین درس
    return out


def popularity_scores(R, candidates):
    return {c: (float(R[c].mean()) if R[c].notna().any() else 0.0) for c in candidates}


def _minmax(d):
    v = list(d.values()); lo, hi = min(v), max(v)
    return {k: (x - lo) / (hi - lo) if hi > lo else 0.0 for k, x in d.items()}


def hybrid_scores(cb, cf, alpha):
    a, b = _minmax(cb), _minmax(cf)
    return {c: alpha * a[c] + (1 - alpha) * b[c] for c in cb}


# ── تعریف «علاقه» دو مرحله‌ای (گذراندن + رضایت) ───────────────────────────────────────────────
# مرحله‌ی ۱: خودِ گذراندن یک درس نشانه‌ی علاقه است (+PREF_TAKEN_BASE) و نگذراندنش جریمه‌ی نرم دارد (PREF_NOT_TAKEN).
# مرحله‌ی ۲: رضایتی که دانشجو از درس گذرانده‌شده داده این علاقه را کم یا زیاد می‌کند: رضایت پایین (۱–۲) جریمه‌ی سخت،
#            رضایت بالا (۴–۵) پاداش سنگین (PREF_SAT_ADJ). (پرسش‌نامه «نمره» ندارد؛ همین رضایت ۱ تا ۵ نقش نمره را دارد.)
# همه‌ی ضریب‌ها همین‌جا و در یک جا هستند؛ تغییرشان همه‌ی اجزا (CB، CF، محبوبیت، ارزیابی) را هم‌زمان عوض می‌کند.
PREF_NOT_TAKEN = -0.25
PREF_TAKEN_BASE = 1.0
PREF_SAT_ADJ = {1: -3.0, 2: -1.5, 3: 0.0, 4: 1.5, 5: 3.0}
PREF_MAX = PREF_TAKEN_BASE + max(PREF_SAT_ADJ.values())


def preference(rating):
    """علاقه‌ی یک درس: NaN (نگذرانده) → جریمه‌ی نرم؛ وگرنه پایه‌ی گذراندن + اثر رضایت."""
    if rating is None or pd.isna(rating):
        return PREF_NOT_TAKEN
    return PREF_TAKEN_BASE + PREF_SAT_ADJ[int(min(5, max(1, round(float(rating)))))]


def preference_matrix(R):
    """ماتریس دانشجو×درس رضایت (NaN = نگذرانده) → ماتریس «علاقه» (بدون NaN)."""
    return R.apply(lambda col: col.map(preference))


def cb_scores_graded(ratings_row, interests, candidates, weight_stated=2.0):
    """مثل cb_scores ولی پروفایل از «علاقه»ی همه‌ی درس‌ها ساخته می‌شود (نگذرانده‌ها هم با جریمه‌ی نرم)."""
    vec = np.zeros(len(TOPICS))
    for t in (interests or []):
        if t in TOPICS:
            vec[TOPICS.index(t)] += weight_stated
    for c, r in ratings_row.items():
        vec += np.asarray(course_vector(c), float) * preference(r)
    return {c: _cos(vec, course_vector(c)) for c in candidates}


def cf_scores_graded(R, u, candidates, k=K_NEIGHBORS):
    """CF روی ماتریس «علاقه»؛ چون ماتریس پر است، الگوی «چه چیزی را گذرانده‌اند» هم در شباهت دانشجوها اثر دارد."""
    return cf_scores(preference_matrix(R), u, candidates, k)


def popularity_graded(R, candidates):
    """محبوبیت بر اساس میانگین «علاقه» (گذراندن + رضایت) میان همه‌ی دانشجوها."""
    P = preference_matrix(R)
    return {c: float(P[c].mean()) for c in candidates}


def rank(scores, seed_order):
    """رتبه‌بندی نزولی؛ برای امتیاز مساوی ترتیب ثابت (seed_order) تا نتیجه قطعی باشد."""
    return sorted(scores, key=lambda c: (-scores[c], seed_order.index(c)))


# ── پیش‌فرض چارت پیشنهادی ────────────────────────────────────────────────────
# ترم پیشنهادی هر درس طبق «چارت پیشنهادی» (فقط درس‌هایی که در پرسش‌نامه‌اند و در چارت ترم دارند)؛
# درس‌های اختیاری بدون ترم در چارت، پیش‌فرض خنثی می‌گیرند.
CHART_TERM = {
    "فیزیک عمومی (۲)": 4, "احتمال ۱": 5, "کامپایلر": 6,
    "شبکه‌های کامپیوتری": 7, "گرافیک کامپیوتری": 7, "زبان‌های برنامه‌سازی": 7,
    "پایگاه داده‌ها": 8, "مقدمه‌ای بر داده‌کاوی": 8, "مباحثی در علوم کامپیوتر": 8,
    "شبیه‌سازی کامپیوتری": 8, "نظریه کدگذاری": 8,
}
MAX_TERM = 8
NEUTRAL_PRIOR = 0.5


def chart_prior(course, term):
    """۱ برای درسی که چارت دقیقا برای این ترم گذاشته؛ با هر ترم فاصله ۱/۳ کمتر (حداقل ۰)؛ بدون ترم در چارت: خنثی."""
    t = CHART_TERM.get(course)
    if t is None: return NEUTRAL_PRIOR
    return max(0.0, 1 - abs(t - term) / 3)


def chart_weight(beta0, term):
    """وزن چارت در ترم‌های اول زیاد و در ترم‌های آخر کم: beta0 * (1 - (term-1)/(MAX_TERM))."""
    return beta0 * max(0.0, 1 - (min(term, MAX_TERM) - 1) / MAX_TERM)


def chart_hybrid_scores(hybrid, term, beta0, eligible=None):
    """امتیاز نهایی = ترکیب CB/CF (min-max‌شده) + وزن_چارت(ترم) * پیش‌فرض_چارت.
    eligible (اختیاری): مجموعه‌ی درس‌هایی که دانشجو پیش‌نیازشان را دارد. اثر چارت فقط روی همین‌ها می‌افتد؛
    بقیه پیش‌فرض خنثی می‌گیرند (نه پاداش، نه جریمه‌ی چارت). None = رفتار قبلی (گزارش)."""
    w = chart_weight(beta0, term)
    return {c: s + w * (chart_prior(c, term) if (eligible is None or c in eligible) else NEUTRAL_PRIOR)
            for c, s in hybrid.items()}


# ── فیلتر پیش‌نیاز (کاندیدهایی که دانشجو پیش‌نیازشان را ندارد حذف می‌شوند) ───────────────────
import os, re as _re


def load_chart_info(path=None):
    """از چارت: پیش‌نیاز هر درس (load_chart) و ترم چارتی هر درس (شیت «چارت پیشنهادی»).
    کلیدها با normalize_name یکدست می‌شوند."""
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
    """پیش‌نیازهای «درس» ارضا شده‌اند؟
    passed: مجموعه‌ی نام‌های (normalize‌شده) درس‌های پرسش‌نامه‌ای که دانشجو گذرانده؛
    پیش‌نیاز خارج از پرسش‌نامه گذرانده فرض می‌شود اگر ترم چارتی‌اش قبل از «term» باشد (یا ترم چارتی نداشته باشد)."""
    from chart_loader import normalize_name as nn
    from course_topics import CHOICE_COURSES
    survey = {nn(c) for c in CHOICE_COURSES}
    for p in prereqs.get(nn(course), []):
        if p in survey:
            if p not in passed: return False
        elif p in terms and terms[p] >= term: return False
    return True


# ── هموارسازی میانگین درس با میانگین گروه موضوعی (برای درس‌های کم‌امتیاز) ─────────────────
def popularity_smoothed(R, candidates, m=3.0):
    """میانگین درس با shrinkage به‌سمت میانگین همان گروه موضوعی: (n*mean + m*groupmean)/(n+m)."""
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
