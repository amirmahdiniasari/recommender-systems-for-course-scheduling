# -*- coding: utf-8 -*-
"""
survey_parser.py
------------------
Explains and implements how the answers of the public survey form (after collection) become data that CF,
content-based filtering and constraint_engine.py can use directly.

Change from the previous version: the "grade of each course" table (5 descriptive bands) was removed from the public
form, because it could not be collected reliably from the general student population. Instead:
  - the "satisfaction with each course" table (1 to 5) replaces it as the main input of CF; this is exactly what
    student_profile_vector in course_topics.py already expected (rated_courses_with_satisfaction).

Another change: the separate checkbox questions "passed courses" and "failed courses" were removed from the public form
(to shorten it):
  - "passed courses" is no longer asked, because it can be inferred from the satisfaction table: every course the
    student gave a numeric score (1 to 5) to has been passed.
  - "failed courses (repeating this term)" was deliberately not added. A sixth option ("I failed") could have been added to
    the satisfaction table, but the signal is useless for this form's practical decision: a failed course is soon retaken,
    and the form cannot know whether it is taken exactly this term, which is what the co-requisite rule actually needs.
    Consequence: failed_courses stays empty for all survey students and that specific rule is only visible in
    hand-written code examples - a deliberate limitation (section 6.2 of the report).
  - The option "I did not take it" stays in the satisfaction table, because the question is starred (mandatory) and the student
    must be able to fill in the row of a course they never took without leaving it empty.

Google Forms exports CSV. For a multiple-choice grid question (the satisfaction table) every course (row) becomes a separate
CSV column named like:
    "How satisfied were you with this course?... [Data Structures and Algorithms]"
so the course name is always at the end of the column title, inside brackets. For this reason a column is matched to a
course by searching for the course name inside the column title instead of exact matching (robust against small changes of
the question text).

Half-space / space matching: whoever types the form in Google Forms may use an ordinary space instead of a half-space
(ZWNJ) in places (for example "سیستم های کامپیوتری" instead of "سیستم‌های کامپیوتری"), so matching course names / keywords against
the column text is done on the normalised version (normalize_name from chart_loader.py, which unifies half-space/space and
Arabic/Persian yeh and kaf), not on the raw text - likewise for matching every option of the "interests" checkbox against the TOPICS list.
"""
import re
import pandas as pd

from course_topics import CHOICE_COURSES, course_vector, student_profile_vector, TOPICS
from chart_loader import normalize_name

# The 28 courses the form actually surveys - only those the student is really free to take
# (CHOICE_COURSES in course_topics.py), not the whole chart. The reason for this restriction is explained
# in the docstring at the top of course_topics.py: satisfaction with a mandatory course reflects teaching
# quality/difficulty more than real interest.
SPECIALIZED_COURSES = list(CHOICE_COURSES)

# known spelling mistakes in the column text of the published form (correct course name -> wrong spelling in the form)
FORM_TYPOS = {"کامپایلر": "کامپابلر"}

# map from the normalised version of every topic -> original TOPICS text, to match the
# half-space/space of the "interests" checkbox options.
_TOPICS_BY_NORM = {normalize_name(t): t for t in TOPICS}

# satisfaction options -> numeric score 1 to 5 (input of CF). "نگذراندم" (did not take) means we have no rating
# for this course (it takes part neither in training nor in the content profile).
# WARNING: these texts must be exactly the same as the option texts of the multiple-choice-grid column of the
# Google form (as typed in the published form), because parsing is done by exact matching of the raw CSV text
# against this dictionary, not by extracting the leading number of the string.
SATISFACTION_LEVELS = [
    "1 - اصلا راضی نبودم",
    "2 - کم راضی بودم",
    "3 - متوسط",
    "4 - راضی بودم",
    "5 - خیلی راضی بودم",
]
SATISFACTION_TO_SCORE = {level: i + 1 for i, level in enumerate(SATISFACTION_LEVELS)}
SATISFACTION_TO_SCORE["نگذراندم"] = None


def find_grid_columns(csv_columns, course_list=SPECIALIZED_COURSES):
    """For every course, finds the exact name of its column in the Google Forms CSV
    (چه برای جدول رضایت، چه برای هر Multiple choice grid دیگری روی همین لیست
    درس‌ها). تطبیق روی نسخه‌ی normalize_name انجام می‌شود تا تفاوت
    نیم‌فاصله/فاصله‌ی معمولی در متن سؤال، درس را گم نکند."""
    mapping = {}
    norm_cols = [(col, normalize_name(col)) for col in csv_columns]
    for course in course_list:
        norm_course = normalize_name(course)
        for col, norm_col in norm_cols:
            if norm_course in norm_col:
                mapping[course] = col
                break
        if course not in mapping:
            # looser matching (without any space) for differences like "غیر خطی" and "غیرخطی",
            # and the known spelling mistakes of the form text (FORM_TYPOS)
            lk = course.replace(" ", "")
            lk = normalize_name(course).replace(" ", "")
            alt = FORM_TYPOS.get(course)
            for col, norm_col in norm_cols:
                nc = norm_col.replace(" ", "")
                if lk in nc or (alt and normalize_name(alt).replace(" ", "") in nc):
                    mapping[course] = col
                    break
    missing = [c for c in course_list if c not in mapping]
    if missing:
        print("WARNING: these courses were not found in the CSV columns (the question text may differ):")
        for m in missing:
            print("   -", m)
    return mapping


def _find_single_column(raw_columns, keyword):
    norm_keyword = normalize_name(keyword)
    matches = [c for c in raw_columns if norm_keyword in normalize_name(c)]
    return matches[0] if matches else None


def _split_checkbox_cell(cell):
    """Google Forms joins the selected checkbox options with ", "."""
    if pd.isna(cell) or not str(cell).strip():
        return set()
    return {p.strip() for p in re.split(r"\s*,\s*|\s*،\s*", str(cell)) if p.strip()}


def _normalize_interest_set(raw_set):
    """Maps every raw option of the "interests" checkbox (with half-space/space matching) to the
    exact TOPICS text in course_topics.py; an option that matches none of the 10 topics
    is ignored (it raises no error and is not reported)."""
    out = set()
    for item in raw_set:
        canon = _TOPICS_BY_NORM.get(normalize_name(item))
        if canon:
            out.add(canon)
    return out


def load_survey(csv_path):
    """Reads the raw Google Forms output and builds a "clean" student x course table:
    one row per student, one column per course = satisfaction score 1 to 5 or NaN (not taken),
    plus side columns (passed / interests / entry year / GPA band). "Passed courses"
    is no longer a separate question - it is inferred directly from the satisfaction table; "failed
    courses" is deliberately not asked in the public form (see the explanation at the top of the file)."""
    raw = pd.read_csv(csv_path)
    col_map = find_grid_columns(list(raw.columns))

    tidy = pd.DataFrame(index=raw.index)
    for course, col in col_map.items():
        tidy[course] = raw[col].map(SATISFACTION_TO_SCORE)

    interest_col = _find_single_column(raw.columns, "علاقه")
    year_col = _find_single_column(raw.columns, "سال ورود")
    gpa_col = _find_single_column(raw.columns, "معدل")

    # passed = every course the student gave a numeric score (other than "did not take").
    rated_cols = list(col_map.keys())
    tidy["درس‌های_گذرانده"] = tidy[rated_cols].apply(
        lambda r: {c for c in rated_cols if not pd.isna(r[c])}, axis=1
    ) if rated_cols else [set()] * len(raw)
    # failed courses are no longer asked; always empty (deliberate limitation, section 6.2 of the report).
    tidy["درس‌های_افتاده"] = [set()] * len(raw)
    tidy["علاقه‌مندی‌ها"] = (
        raw[interest_col].map(_split_checkbox_cell).map(_normalize_interest_set)
        if interest_col else [set()] * len(raw)
    )
    if year_col:
        tidy["سال_ورود"] = raw[year_col]
    if gpa_col:
        # only to describe the sample; not used in any calculation (the survey asks the
        # overall GPA, while the unit cap needs the previous term's GPA).
        tidy["بازه_معدل"] = raw[gpa_col]

    return tidy


def student_to_engine_state(row):
    """Converts one row of the clean table into passed_courses/failed_courses -
    exactly the fields constraint_engine.StudentState needs.
    passed is inferred from the satisfaction table; failed is always empty, because the public
    form does not ask this question (see the explanation at the top of the file)."""
    return set(row.get("درس‌های_گذرانده", set())), set(row.get("درس‌های_افتاده", set()))


def build_cf_training_table(tidy_df):
    """For CF: long format (student_idx, course, satisfaction) - only the rows where the
    student really took that course and gave a satisfaction score (not NaN)."""
    rows = []
    for student_idx, srow in tidy_df.iterrows():
        for course in SPECIALIZED_COURSES:
            score = srow.get(course)
            if pd.isna(score):
                continue
            rows.append({"دانشجو": student_idx, "درس": course, "رضایت": int(score)})
    return pd.DataFrame(rows)


def build_student_profiles(tidy_df):
    """For content-based filtering: the 10-dimensional vector of each student (from course_topics.py),
    a combination of declared interests + the real satisfaction with courses."""
    profiles = {}
    for student_idx, srow in tidy_df.iterrows():
        ratings = {
            course: int(srow[course])
            for course in SPECIALIZED_COURSES
            if not pd.isna(srow.get(course))
        }
        profiles[student_idx] = student_profile_vector(
            interest_topics=srow.get("علاقه‌مندی‌ها", set()),
            rated_courses_with_satisfaction=ratings,
        )
    return profiles


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python survey_parser.py <path of the Google Forms CSV export>")
        print("No form data yet, so a synthetic sample is built to test the structure...")
        import io
        # ستون‌های نمونه از CHOICE_COURSES انتخاب شدند (نه دروس اجباری)، چون
        # فرم فقط رضایت درس‌های واقعا‌انتخابی را می‌پرسد (بالای فایل).
        # ردیف اول عمدا با فاصله‌ی معمولی به‌جای نیم‌فاصله تایپ شده («پایگاه
        # داده ها») — دقیقا همان خطایی که موقع تایپ فرم در گوگل‌فرم پیش
        # می‌آید؛ هدف این نمونه، اثبات این است که normalize_name آن را درست
        # تشخیص می‌دهد و گم نمی‌شود.
        fake_csv = (
            "Timestamp,"
            "چقدر از این درس راضی بودید؟... [پایگاه داده ها],"
            "چقدر از این درس راضی بودید؟... [گرافیک کامپیوتری],"
            "علاقه‌مندی‌ها,تقریبا معدل کلی فعلی‌تان در کدام بازه است؟\n"
            "2026-10-07,5 - خیلی راضی بودم,نگذراندم,برنامه نویسی وب,15 تا 18\n"
            "2026-10-07,2 - کم راضی بودم,3 - متوسط,شبکه,کمتر از 12\n"
        )
        tidy = load_survey(io.StringIO(fake_csv))
        print("\nClean table of the sample:")
        print(tidy[["پایگاه داده‌ها", "گرافیک کامپیوتری", "درس‌های_گذرانده", "علاقه‌مندی‌ها", "بازه_معدل"]].to_string())
        p, f = student_to_engine_state(tidy.iloc[0])
        print(f"\nدانشجوی اول -> passed={p}, failed={f}")
    else:
        tidy = load_survey(sys.argv[1])
        tidy.to_csv("survey_clean.csv", index=False, encoding="utf-8-sig")
        print(f"{len(tidy)} responses processed; output in survey_clean.csv")
