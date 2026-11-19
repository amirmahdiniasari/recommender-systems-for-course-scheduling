# -*- coding: utf-8 -*-
"""
survey_parser.py
------------------
این فایل توضیح می‌دهد و پیاده می‌کند که جواب‌های فرم عمومی (بعد از
جمع‌آوری) چطور به داده‌ای تبدیل می‌شوند که CF، content-based و
constraint_engine.py می‌توانند مستقیم از آن استفاده کنند.

⚠ تغییر نسبت به نسخه‌ی قبلی: جدول «نمره‌ی هر درس» (۵ بازه‌ی توصیفی) از فرم
عمومی حذف شد، چون امکان جمع‌آوری قابل‌اتکای آن از جامعه‌ی عمومی دانشجوها
فراهم نشد. به‌جایش:
  - جدول «رضایت از هر درس» (۱ تا ۵) جایگزین ورودی اصلی CF شده — این دقیقا
    همان چیزی‌ست که student_profile_vector در course_topics.py از قبل
    انتظارش را داشت (rated_courses_with_satisfaction).

⚠ تغییر دیگر: سؤال‌های جداگانه‌ی Checkbox «درس‌های گذرانده» و «درس‌های افتاده»
از فرم عمومی حذف شدند (برای کوتاه‌ترکردن فرم):
  - «درس‌های گذرانده» دیگر پرسیده نمی‌شود، چون از همان جدول رضایت قابل‌استنتاج
    است: هر درسی که دانشجو برایش امتیازی عددی (۱ تا ۵) زده، یعنی گذرانده.
  - «درس‌های افتاده (در حال تکرار همین ترم)» عمدا اضافه نشد. اگرچه می‌شد یک
    گزینه‌ی ششم («افتاده‌ام») به جدول رضایت اضافه کرد، اما این سیگنال برای
    تصمیم عملی این فرم بی‌فایده است: درس افتاده را دانشجو زودیادیر دوباره
    می‌گیرد، و فرم نمی‌داند که آیا دقیقا همین ترم دارد می‌گیرد یا نه—همان چیزی
    که قاعده‌ی هم‌نیازی واقعا به آن نیاز دارد. پیامد: failed_courses برای
    همه‌ی دانشجویان داده‌ی فرم همیشه خالی می‌ماند و آن قاعده‌ی خاص فقط در
    مثال‌های دستی کد قابل‌مشاهده می‌ماند—یک محدودیت آگاهانه (بخش ۶.۲).
  - گزینه‌ی «نگذراندم» در جدول رضایت باقی می‌ماند، چون سؤال ستاره‌دار
    (الزامی) است و دانشجو باید بتواند برای درسی که اصلا نگرفته، ردیفش را
    بدون باقی‌گذاشتن خالی پر کند.

گوگل‌فرم خروجی را به‌صورت CSV می‌دهد. برای سؤال Multiple choice grid
(جدول رضایت)، هر درس (ردیف) یک ستون جدا در CSV می‌شود، با نامی شبیه:
    "چقدر از این درس راضی بودید؟... [ساختمان داده‌ها و الگوریتم‌ها]"
یعنی نام درس همیشه ته عنوان ستون، داخل کروشه می‌آید — برای همین تطبیق‌دادن
ستون به نام درس، به‌جای مطابقت دقیق، با جست‌وجوی نام درس داخل عنوان ستون
انجام می‌شود (مقاوم در برابر تغییر جزئی متن سؤال).

⚠ تطبیق نیم‌فاصله/فاصله: چون کسی که فرم را در گوگل‌فرم تایپ می‌کند ممکن است
جابه‌جا فاصله‌ی معمولی به‌جای نیم‌فاصله بزند (مثلا «سیستم های کامپیوتری»
به‌جای «سیستم‌های کامپیوتری»)، تطبیق نام درس/کلیدواژه با متن ستون روی نسخه‌ی
نرمال‌شده (normalize_name از chart_loader.py، که نیم‌فاصله/فاصله/ي‌-ی/ك-ک را
یکسان می‌کند) انجام می‌شود، نه روی متن خام — همین‌طور برای تطبیق هر گزینه‌ی
چک‌باکس «علاقه‌مندی‌ها» با لیست TOPICS.
"""
import re
import pandas as pd

from course_topics import CHOICE_COURSES, course_vector, student_profile_vector, TOPICS
from chart_loader import normalize_name

# ۲۸ درسی که فرم واقعا رضایت‌سنجی می‌کند — فقط آن‌هایی که دانشجو در گرفتن‌شان
# واقعا آزاد است (CHOICE_COURSES در course_topics.py)، نه کل چارت. دلیل این
# محدودسازی در docstring بالای course_topics.py توضیح داده شده: رضایت از یک
# درس اجباری بیشتر کیفیت تدریس/سختی را نشان می‌دهد تا علاقه‌ی واقعی.
SPECIALIZED_COURSES = list(CHOICE_COURSES)

# غلط‌های املایی شناخته‌شده در متن ستون‌های فرم منتشرشده (نام صحیح درس -> نوشته‌ی غلط در فرم)
FORM_TYPOS = {"کامپایلر": "کامپابلر"}

# نگاشت نسخه‌ی نرمال‌شده‌ی هر حوزه -> متن اصلی TOPICS، برای تطبیق
# نیم‌فاصله/فاصله‌ی گزینه‌های چک‌باکس «علاقه‌مندی‌ها».
_TOPICS_BY_NORM = {normalize_name(t): t for t in TOPICS}

# بازه‌های رضایت -> امتیاز عددی ۱ تا ۵ (ورودی CF). «نگذراندم» یعنی رتبه‌ای
# برای این درس نداریم (نه در ترینینگ شرکت می‌کند، نه در پروفایل محتوایی).
# ⚠ این متن‌ها دقیقا باید با متن گزینه‌های ستون Multiple-choice-grid فرم
# گوگل یکی باشند (همان‌طور که در فرم منتشرشده تایپ شده‌اند)، چون پارس‌کردن
# بر پایه‌ی تطبیق دقیق متن خام CSV با این دیکشنری انجام می‌شود، نه با
# استخراج عدد ابتدای رشته.
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
    """برای هر درس، نام دقیق ستون متناظرش را در CSV گوگل‌فرم پیدا می‌کند
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
            # تطبیق سست‌تر (بدون هیچ فاصله) برای تفاوت‌هایی مثل «غیر خطی» و «غیرخطی»،
            # و غلط‌های املایی شناخته‌شده‌ی متن فرم (FORM_TYPOS)
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
        print("⚠ این درس‌ها در ستون‌های CSV پیدا نشدند (شاید متن سؤال فرق دارد):")
        for m in missing:
            print("   -", m)
    return mapping


def _find_single_column(raw_columns, keyword):
    norm_keyword = normalize_name(keyword)
    matches = [c for c in raw_columns if norm_keyword in normalize_name(c)]
    return matches[0] if matches else None


def _split_checkbox_cell(cell):
    """گوگل‌فرم گزینه‌های چک‌باکس انتخاب‌شده را با ", " به هم می‌چسباند."""
    if pd.isna(cell) or not str(cell).strip():
        return set()
    return {p.strip() for p in re.split(r"\s*,\s*|\s*،\s*", str(cell)) if p.strip()}


def _normalize_interest_set(raw_set):
    """هر گزینه‌ی خام چک‌باکس «علاقه‌مندی‌ها» را (با تطبیق نیم‌فاصله/فاصله) به
    متن دقیق TOPICS در course_topics.py نگاشت می‌کند؛ گزینه‌ای که با هیچ‌کدام
    از ۱۰ حوزه مطابق نشود، نادیده گرفته می‌شود (و نه خطا می‌دهد، نه گزارش)."""
    out = set()
    for item in raw_set:
        canon = _TOPICS_BY_NORM.get(normalize_name(item))
        if canon:
            out.add(canon)
    return out


def load_survey(csv_path):
    """خروجی خام گوگل‌فرم را می‌خواند و یک جدول «تمیز» دانشجو×درس می‌سازد:
    هر سطر یک دانشجو، هر ستون درس = امتیاز رضایت ۱ تا ۵ یا NaN (نگذرانده)،
    به‌علاوه‌ی ستون‌های جانبی (گذرانده/علاقه/سال‌ورود/معدل). «درس‌های گذرانده»
    دیگر سؤال جداگانه نیست — مستقیم از جدول رضایت استنتاج می‌شود؛ «درس‌های
    افتاده» عمدا در فرم عمومی پرسیده نمی‌شود (توضیح بالای فایل)."""
    raw = pd.read_csv(csv_path)
    col_map = find_grid_columns(list(raw.columns))

    tidy = pd.DataFrame(index=raw.index)
    for course, col in col_map.items():
        tidy[course] = raw[col].map(SATISFACTION_TO_SCORE)

    interest_col = _find_single_column(raw.columns, "علاقه")
    year_col = _find_single_column(raw.columns, "سال ورود")
    gpa_col = _find_single_column(raw.columns, "معدل")

    # گذرانده = هر درسی که دانشجو امتیازی عددی (غیر از «نگذرانده‌ام») زده.
    rated_cols = list(col_map.keys())
    tidy["درس‌های_گذرانده"] = tidy[rated_cols].apply(
        lambda r: {c for c in rated_cols if not pd.isna(r[c])}, axis=1
    ) if rated_cols else [set()] * len(raw)
    # افتاده دیگر پرسیده نمی‌شود؛ همیشه خالی (محدودیت آگاهانه، بخش ۶.۲ گزارش).
    tidy["درس‌های_افتاده"] = [set()] * len(raw)
    tidy["علاقه‌مندی‌ها"] = (
        raw[interest_col].map(_split_checkbox_cell).map(_normalize_interest_set)
        if interest_col else [set()] * len(raw)
    )
    if year_col:
        tidy["سال_ورود"] = raw[year_col]
    if gpa_col:
        # فقط برای توصیف نمونه؛ در هیچ محاسبه‌ای استفاده نمی‌شود (پرسش‌نامه
        # معدل کل را می‌پرسد، سقف واحد به معدل ترم قبل نیاز دارد).
        tidy["بازه_معدل"] = raw[gpa_col]

    return tidy


def student_to_engine_state(row):
    """یک سطر از جدول تمیز را به passed_courses/failed_courses تبدیل می‌کند —
    دقیقا همان فیلدهایی که constraint_engine.StudentState نیاز دارد.
    passed از جدول رضایت استنتاج می‌شود؛ failed همیشه خالی است، چون فرم
    عمومی این سؤال را نمی‌پرسد (توضیح بالای فایل)."""
    return set(row.get("درس‌های_گذرانده", set())), set(row.get("درس‌های_افتاده", set()))


def build_cf_training_table(tidy_df):
    """برای CF: فرمت بلند (student_idx, درس, رضایت) — فقط ردیف‌هایی که
    دانشجو واقعا آن درس را گذرانده و رضایتش را زده (نه NaN)."""
    rows = []
    for student_idx, srow in tidy_df.iterrows():
        for course in SPECIALIZED_COURSES:
            score = srow.get(course)
            if pd.isna(score):
                continue
            rows.append({"دانشجو": student_idx, "درس": course, "رضایت": int(score)})
    return pd.DataFrame(rows)


def build_student_profiles(tidy_df):
    """برای content-based: بردار ۱۰بعدی هر دانشجو (از course_topics.py)،
    ترکیب علاقه‌مندی‌های اعلام‌شده + رضایت واقعی از درس‌ها."""
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
        print("استفاده: python survey_parser.py <مسیر CSV خروجی گوگل‌فرم>")
        print("هنوز فرم پر نشده، پس نمونه‌ی ساختگی برای تست ساختار می‌سازیم...")
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
        print("\nجدول تمیز نمونه:")
        print(tidy[["پایگاه داده‌ها", "گرافیک کامپیوتری", "درس‌های_گذرانده", "علاقه‌مندی‌ها", "بازه_معدل"]].to_string())
        p, f = student_to_engine_state(tidy.iloc[0])
        print(f"\nدانشجوی اول -> passed={p}, failed={f}")
    else:
        tidy = load_survey(sys.argv[1])
        tidy.to_csv("survey_clean.csv", index=False, encoding="utf-8-sig")
        print(f"{len(tidy)} پاسخ پردازش شد؛ خروجی در survey_clean.csv")
