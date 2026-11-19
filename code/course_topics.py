# -*- coding: utf-8 -*-
"""
course_topics.py
-------------------
Manual topic labels for the courses - the basis of the content vector of content-based filtering
(section 2.2.1 of the report).

The previous version of this file used the same 9 topics as the form's "interests" question, but three courses
(physics, research methods, workshop) fitted none of them. To fix this, two topics were changed/added (10 instead of 9):
  - "pure mathematics" was renamed "mathematics and basic sciences" so that it also covers physics.
  - "research and practical skills" was added to cover research methods and the workshop.
WARNING: because these topics changed, the form's "interests" question (which used to have 9 options)
must also be aligned with these 10 topics.

A more important change: COURSE_TOPIC_MAP is no longer itself the "list of form questions" - it covers all
courses of the chart, mandatory or elective. The form's grid question comes only from the subset CHOICE_COURSES (further down
in this file): the courses the student is really free to choose
(based on the basic analysis showing that satisfaction with a mandatory course reflects teaching
quality/difficulty more than real interest, see section 2.2.1 of the report). This list was built by
filtering the "status" column of the chart (every row containing "elective") plus the whole
sheet "دروس اختیاری", except two important exceptions: "Foundations of Mathematical Analysis" and
"Foundations of Algebra", which according to the chart's own footnote are in practice mandatory despite the title "elective" (so they are not real
choices and were left out of CHOICE_COURSES), and three explanatory non-course rows (sheet footnotes) that were not course names at all.
"""

TOPICS = [
    "هوش مصنوعی", "شبکه", "امنیت", "برنامه‌نویسی وب", "گرافیک کامپیوتری",
    "پایگاه‌داده و داده‌کاوی", "نرم‌افزار و معماری", "ریاضیات و علوم پایه",
    "سخت‌افزار و سیستم", "مهارت‌های پژوهشی و عملی",
]

# course name -> (category, [topics])   - in the same order 1 to 31 of the Google form
COURSE_TOPIC_MAP = {
    "اصول سیستم‌های عامل":            ("تخصصی",  ["سخت‌افزار و سیستم"]),
    "اصول سیستم‌های کامپیوتری":        ("تخصصی",  ["سخت‌افزار و سیستم"]),
    "اصول طراحی نرم‌افزار":            ("تخصصی",  ["نرم‌افزار و معماری"]),
    "بهینه‌سازی خطی":                  ("تخصصی",  ["ریاضیات و علوم پایه"]),
    "جبر خطی عددی":                   ("تخصصی",  ["ریاضیات و علوم پایه"]),
    "ساختمان داده‌ها و الگوریتم‌ها":    ("تخصصی",  ["نرم‌افزار و معماری"]),
    "شبکه‌های کامپیوتری":              ("تخصصی",  ["شبکه"]),
    "طراحی و تحلیل الگوریتم‌ها":       ("تخصصی",  ["نرم‌افزار و معماری"]),
    "مباحثی در علوم کامپیوتر":         ("تخصصی",  ["نرم‌افزار و معماری"]),
    "مبانی آنالیز عددی":              ("تخصصی",  ["ریاضیات و علوم پایه"]),
    "مبانی احتمال":                   ("تخصصی",  ["ریاضیات و علوم پایه"]),
    "مبانی ترکیبیات":                 ("تخصصی",  ["ریاضیات و علوم پایه"]),
    "مبانی علوم ریاضی":               ("تخصصی",  ["ریاضیات و علوم پایه"]),
    "مبانی ماتریس‌ها و جبر خطی":       ("تخصصی",  ["ریاضیات و علوم پایه"]),
    "مبانی منطق و نظریه مجموعه‌ها":    ("تخصصی",  ["ریاضیات و علوم پایه"]),
    "هوش مصنوعی":                     ("تخصصی",  ["هوش مصنوعی"]),
    "مبانی جبر":                      ("تخصصی",  ["ریاضیات و علوم پایه"]),
    "مبانی نظریه محاسبه":             ("تخصصی",  ["ریاضیات و علوم پایه", "نرم‌افزار و معماری"]),
    "نظریه محاسبه":                   ("تخصصی",  ["ریاضیات و علوم پایه", "نرم‌افزار و معماری"]),
    "پایگاه داده‌ها":                 ("تخصصی",  ["پایگاه‌داده و داده‌کاوی"]),
    "برنامه‌سازی پیشرفته":            ("پایه",    ["نرم‌افزار و معماری"]),
    "کامپایلر":                       ("تخصصی",  ["نرم‌افزار و معماری", "سخت‌افزار و سیستم"]),
    "گرافیک کامپیوتری":               ("اختیاری", ["گرافیک کامپیوتری"]),  # according to the chart it really is in the "elective courses" sheet, not a mandatory specialised course
    "ریاضی 1":                        ("پایه",    ["ریاضیات و علوم پایه"]),
    "ریاضی 2":                        ("پایه",    ["ریاضیات و علوم پایه"]),
    "معادلات دیفرانسیل":              ("پایه",    ["ریاضیات و علوم پایه"]),
    "روش تحقیق و گزارش‌نویسی":         ("پایه",    ["مهارت‌های پژوهشی و عملی"]),
    "کارگاه":                         ("پایه",    ["مهارت‌های پژوهشی و عملی"]),
    "مقدمه‌ای بر داده‌کاوی":           ("اختیاری", ["پایگاه‌داده و داده‌کاوی", "هوش مصنوعی"]),
    "زبان‌های برنامه‌سازی":            ("اختیاری", ["نرم‌افزار و معماری"]),
    "فیزیک 1":                        ("پایه",    ["ریاضیات و علوم پایه"]),

    # --- "real elective" courses that were just added (explained at the top of the file) ---
    # elective group of 4 base courses (you must take 1):
    "فیزیک عمومی (۲)":                ("پایه",    ["ریاضیات و علوم پایه"]),
    "اصول مدیریت":                    ("پایه",    []),
    "اصول حسابداری و هزینه‌یابی":      ("پایه",    []),
    "مبانی اقتصاد":                   ("پایه",    []),
    # elective group of 7 specialised courses (you must take at least 2):
    "آنالیز عددی":                    ("تخصصی",  ["ریاضیات و علوم پایه"]),
    "نظریه گراف و کاربردها":          ("تخصصی",  ["ریاضیات و علوم پایه", "نرم‌افزار و معماری"]),
    "بهینه‌سازی غیرخطی":              ("تخصصی",  ["ریاضیات و علوم پایه"]),
    # sheet "elective courses" (completely free):
    "منطق":                           ("اختیاری", ["ریاضیات و علوم پایه"]),
    "مبانی کارآفرینی":                ("اختیاری", []),
    "مبانی بیوانفورماتیک":            ("اختیاری", ["پایگاه‌داده و داده‌کاوی"]),
    "کارآموزی ۱":                    ("اختیاری", []),
    "کارآموزی ۲":                    ("اختیاری", []),
    # the form asks these two as one question (below, in CHOICE_COURSES); the two courses
    # above were kept to cover the chart completely.
    "کارآموزی ۱ و ۲":                 ("اختیاری", []),
    "مباحثی در الگوریتم‌ها":          ("اختیاری", ["نرم‌افزار و معماری"]),
    "نظریه کدگذاری":                 ("اختیاری", ["شبکه", "ریاضیات و علوم پایه"]),
    "طراحی هندسی کامپیوتری":          ("اختیاری", ["گرافیک کامپیوتری"]),
    "کاربرد کامپیوتر در سیستم‌های تجاری": ("اختیاری", ["پایگاه‌داده و داده‌کاوی"]),
    "سیستم‌های اطلاعات مدیریت":       ("اختیاری", ["پایگاه‌داده و داده‌کاوی"]),
    "زیست‌شناسی سلولی و مولکولی":     ("اختیاری", []),
    "ریاضیات مهندسی":                ("اختیاری", ["ریاضیات و علوم پایه"]),
    "بهینه‌سازی ترکیبیاتی و آنالیز شبکه‌ها": ("اختیاری", ["ریاضیات و علوم پایه", "نرم‌افزار و معماری"]),
    "احتمال ۱":                      ("اختیاری", ["ریاضیات و علوم پایه"]),
    "شبیه‌سازی کامپیوتری":            ("اختیاری", ["نرم‌افزار و معماری", "ریاضیات و علوم پایه"]),
    "پروژه":                          ("اختیاری", []),
}

assert len(COURSE_TOPIC_MAP) == 55, f"انتظار ۵۵ درس بود، {len(COURSE_TOPIC_MAP)} تا پیدا شد"

# Courses the student is really free to take (full explanation at the top of the file):
# elective groups of base/specialised courses + the whole sheet "elective courses", except two courses that
# are in practice mandatory according to the chart footnote. Only the form's grid question is built from this list;
# the full COURSE_TOPIC_MAP, for other uses (such as course_vector), covers both kinds.
# "Project" was removed (it is in practice mandatory for everyone, so asking satisfaction is meaningless);
# "Internship 1" and "Internship 2" were merged into one question ("کارآموزی ۱ و ۲").
CHOICE_COURSES = [
    # elective group of base courses
    "فیزیک عمومی (۲)", "اصول مدیریت", "اصول حسابداری و هزینه‌یابی", "مبانی اقتصاد",
    # elective group of specialised courses
    "کامپایلر", "پایگاه داده‌ها", "شبکه‌های کامپیوتری", "مباحثی در علوم کامپیوتر",
    "بهینه‌سازی غیرخطی", "نظریه گراف و کاربردها", "آنالیز عددی",
    # completely free elective courses
    "منطق", "مقدمه‌ای بر داده‌کاوی", "مبانی کارآفرینی", "مبانی بیوانفورماتیک",
    "کارآموزی ۱ و ۲", "مباحثی در الگوریتم‌ها", "نظریه کدگذاری",
    "طراحی هندسی کامپیوتری", "کاربرد کامپیوتر در سیستم‌های تجاری",
    "سیستم‌های اطلاعات مدیریت", "زیست‌شناسی سلولی و مولکولی", "زبان‌های برنامه‌سازی",
    "ریاضیات مهندسی", "بهینه‌سازی ترکیبیاتی و آنالیز شبکه‌ها", "گرافیک کامپیوتری",
    "احتمال ۱", "شبیه‌سازی کامپیوتری",
]
assert len(CHOICE_COURSES) == 28, f"انتظار ۲۸ درس انتخابی بود، {len(CHOICE_COURSES)} تا پیدا شد"
assert all(c in COURSE_TOPIC_MAP for c in CHOICE_COURSES), "یک درس انتخابی تگ موضوعی ندارد"


def course_vector(course_name):
    """Multi-valued one-hot vector over the 10 topics, for computing cosine similarity."""
    _, topics = COURSE_TOPIC_MAP.get(course_name, ("نامشخص", []))
    return [1 if t in topics else 0 for t in TOPICS]


def student_profile_vector(interest_topics, rated_courses_with_satisfaction, weight_stated=2.0):
    """
    interest_topics: فهرست حوزه‌هایی که دانشجو در سؤال «علاقه‌مندی‌ها»ی فرم انتخاب کرده.
    rated_courses_with_satisfaction: دیکشنری {نام‌درس: امتیاز ۱ تا ۵} از جدول رضایت فرم.
    خروجی: بردار ۱۰بعدی پروفایل دانشجو (ترکیب علاقه‌ی اعلام‌شده + رفتار واقعی).
    """
    vec = [0.0] * len(TOPICS)
    for t in interest_topics:
        if t in TOPICS:
            vec[TOPICS.index(t)] += weight_stated
    for course, score in rated_courses_with_satisfaction.items():
        cv = course_vector(course)
        centered = score - 3  # 3 out of 5 is treated as neutral
        for i, v in enumerate(cv):
            vec[i] += v * centered
    return vec


if __name__ == "__main__":
    import math
    def cosine(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        na, nb = math.sqrt(sum(x*x for x in a)), math.sqrt(sum(x*x for x in b))
        return dot / (na*nb) if na and nb else 0.0

    print(f"Number of labelled courses: {len(COURSE_TOPIC_MAP)}")
    print(f"Number of topics: {len(TOPICS)} -> {TOPICS}\n")

    profile = student_profile_vector(
        interest_topics=["هوش مصنوعی"],
        rated_courses_with_satisfaction={"مبانی احتمال": 5, "شبکه‌های کامپیوتری": 2, "کارگاه": 4},
    )
    print("Profile vector of the sample student:", dict(zip(TOPICS, profile)))
    print()
    for course in ["هوش مصنوعی", "پایگاه داده‌ها", "شبکه‌های کامپیوتری", "فیزیک 1", "کارگاه"]:
        sim = cosine(profile, course_vector(course))
        print(f"Similarity with \"{course}\": {sim:.2f}")
