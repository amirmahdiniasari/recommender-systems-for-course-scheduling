"""آزمایش‌های تکمیلی روی ارزیابی اصلی (بخش‌های ۵.۱۰، ۵.۱۱ و ۵.۱۲ گزارش).
اجرا:  python evaluate_extra.py form_responses_23.csv
خروجی: چهار حالت (پایه، پیش‌نیاز روی درس هدف، بدون علاقه‌مندی‌ها، آموزش بدون پاسخ‌دهنده‌های «همه‌ی ۲۸ درس») با خط پایه‌ی
Count (تعداد گذرانندگان درس در دانشجوهای آموزش) و p جایگشتی جفت‌شده‌ی اکتشافی (بدون تصحیح)."""
import sys, json
import numpy as np, pandas as pd
import evaluate_rq1 as ev, survey_parser as sp

csv = sys.argv[1] if len(sys.argv) > 1 else "form_responses_23.csv"
tidy = sp.load_survey(csv)
courses = [c for c in sp.SPECIALIZED_COURSES if c in tidy.columns]
R = tidy[courses].astype(float)
interests, years = tidy["علاقه‌مندی‌ها"], tidy["سال_ورود"]
pool = list(R.index)
ev.METHODS = list(ev.METHODS) + ["Count"]
_orig = ev.score_all
def _sa(t, train, it, cs):
    s = _orig(t, train, it, cs)
    s["Count"] = {c: float(train[c].notna().sum()) for c in t["cands"]}
    return s
ev.score_all = _sa
tests = ev.make_tests(R, interests, years, courses)
tested = sorted({t["u"] for t in tests})
full = [u for u in R.index if R.loc[u].notna().all()]          # پاسخ‌دهنده‌هایی که هر ۲۸ درس را امتیاز داده‌اند (بدون آزمون)
PAIRS = [("Hybrid", "Popularity"), ("Hybrid", "Count"), ("Count", "Popularity"), ("CF", "Count"), ("CB", "Count")]

def run(name, tests_, it, pool_):
    per = ev.evaluate(tests_, pool_, R, it, courses)
    tab = ev.summarize(per)
    print(f"\n== {name}: {len(tests_)} آزمون، {per.student.nunique()} دانشجو، آموزش از {len(pool_)} نفر")
    print(tab[["R@3_mean", "NDCG@5_mean", "MRR_mean"]].round(3).to_string())
    P = {f"{a} vs {b} [{m}]": ev.paired_p(per, a, b, m) for a, b in PAIRS for m in ("R@3", "NDCG@5")}
    print({k: (None if v is None else round(v, 3)) for k, v in P.items()})
    return {"table": tab.round(4).to_dict("index"), "p": P}

out = {"full28_students": [int(u) for u in full]}
out["base"] = run("پایه", tests, interests, pool)
kept = []
for t in tests:
    passed = {ev._nn(c) for c in courses if not pd.isna(t["ru"][c])}
    if ev.prereq_ok(t["held"], passed, t["term"], ev.PREREQS, ev.TERMS): kept.append(t)
out["held_prereq"] = run("پیش‌نیاز روی درس هدف هم اعمال شد", kept, interests, pool)
noint = pd.Series([set() for _ in R.index], index=R.index)
out["no_interests"] = run("بدون علاقه‌مندی‌های اعلام‌شده", tests, noint, pool)
pool2 = [u for u in pool if u not in set(full) or u in set(tested)]
out["train_without_full"] = run(f"آموزش بدون {len(pool) - len(pool2)} پاسخ‌دهنده‌ی «همه‌ی ۲۸ درس»", tests, interests, pool2)
json.dump(out, open("extra_results.json", "w", encoding="utf8"), ensure_ascii=False, indent=2, default=float)
