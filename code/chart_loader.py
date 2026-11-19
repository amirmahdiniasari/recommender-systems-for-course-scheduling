# -*- coding: utf-8 -*-
"""
chart_loader.py
----------------
Loads the digitized curriculum chart (chart_computer_science.xlsx) into a
single, clean, machine-readable table of courses + prerequisites, plus the
two "offered this term" sheets.

This is the data layer for the constraint-filter engine (قدم ۴ از برنامه).
"""
import re
import openpyxl
import pandas as pd

TITLE_COL_NAMES = {"نام درس", "عنوان درس"}
UNIT_COL_NAMES = {"واحد", "واحد (نظری-عملی)", "تعداد واحد (نظری – عملی)"}
PREREQ_COL_NAMES = {"پیش‌نیاز", "پیش نیاز"}
STATUS_COL_NAMES = {"وضعیت", "توضیح"}


_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_LATIN_DIGITS = "0123456789"
_DIGIT_MAP = {ord(a): b for a, b in zip(_PERSIAN_DIGITS + _ARABIC_DIGITS, _LATIN_DIGITS * 2)}


def normalize_name(name):
    """Normalize a course-title string so that the same course written
    differently in the chart vs. the offered-courses export (Persian vs.
    Latin digits, extra spaces, ZWNJ, parentheses, ي/ك vs ی/ک) compares
    equal. Used only for matching — never for display."""
    if name is None:
        return ""
    s = str(name)
    s = s.translate(_DIGIT_MAP)
    s = s.replace("\u200c", " ")           # ZWNJ -> space
    s = s.replace("ي", "ی").replace("ك", "ک")
    s = re.sub(r"[()]", " ", s)            # drop parentheses, keep content
    s = re.sub(r"[‌،,]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def loose_key(name):
    """An even looser key (all whitespace stripped) for cases like
    'سیستم‌های' (chart) vs 'سیستمهای' (offered-list export) that differ
    only in where a space/ZWNJ was typed."""
    return normalize_name(name).replace(" ", "")


def _parse_units(raw):
    """'3-0' or '3-1' or '3' or 18 -> float total units."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s in ("", "-"):
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$", s)
    if m:
        return float(m.group(1)) + float(m.group(2))
    try:
        return float(s)
    except ValueError:
        return None


def _parse_prereqs(raw):
    """Split a prereq cell into a list of individual prerequisite names/rules.
    Handles '-', '،' separated lists, and 'گذراندن N واحد' rules (kept as-is,
    flagged separately so the engine can check against total units passed).
    """
    if raw is None:
        return []
    s = str(raw).strip()
    if s in ("", "-"):
        return []
    parts = [p.strip() for p in re.split("[،,]", s) if p.strip()]
    return parts


def _sheet_to_records(ws, category):
    """Scan a worksheet for one or more sub-tables (title + header + rows)
    and yield unified course records."""
    records = []
    rows = list(ws.iter_rows(values_only=True))
    header_idx = None
    col_map = {}
    for i, row in enumerate(rows):
        cells = [c for c in row if c is not None]
        if not cells:
            continue
        # detect a header row
        row_set = {str(c).strip() for c in row if c is not None}
        if row_set & TITLE_COL_NAMES:
            header_idx = i
            col_map = {}
            for ci, val in enumerate(row):
                if val is None:
                    continue
                key = str(val).strip()
                col_map[key] = ci
            continue
        if header_idx is None:
            continue
        # a data row belonging to the current header
        title_col = next((col_map[k] for k in TITLE_COL_NAMES if k in col_map), None)
        if title_col is None or title_col >= len(row):
            continue
        title = row[title_col]
        if title is None or str(title).strip() in ("", "-"):
            continue
        unit_col = next((col_map[k] for k in UNIT_COL_NAMES if k in col_map), None)
        prereq_col = next((col_map[k] for k in PREREQ_COL_NAMES if k in col_map), None)
        status_col = next((col_map[k] for k in STATUS_COL_NAMES if k in col_map), None)

        units = _parse_units(row[unit_col]) if unit_col is not None and unit_col < len(row) else None
        prereqs = _parse_prereqs(row[prereq_col]) if prereq_col is not None and prereq_col < len(row) else []
        status = row[status_col] if status_col is not None and status_col < len(row) else ""

        records.append({
            "نام درس": re.sub(r"\s+", " ", str(title).replace("،", " ")).strip(),  # ویرگول داخل نام درس («تاریخ، فرهنگ و تمدن…») حذف می‌شود
            "واحد": units,
            "پیش‌نیاز": prereqs,
            "دسته": category,
            "وضعیت": str(status).strip() if status else "",
        })
    return records


def load_chart(xlsx_path):
    """Returns a pandas DataFrame: one row per course in the curriculum,
    with columns نام درس, واحد, پیش‌نیاز (list), دسته, وضعیت."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    all_records = []
    sheet_to_category = {
        "دروس عمومی": "عمومی",
        "دروس پایه": "پایه",
        "دروس پیش‌دانشگاهی": "پیش‌نیاز",
        "دروس تخصصی": "تخصصی",
        "دروس اختیاری": "اختیاری",
        "دروس معارف": "معارف",
    }
    for sheet_name, category in sheet_to_category.items():
        if sheet_name not in wb.sheetnames:
            continue
        all_records.extend(_sheet_to_records(wb[sheet_name], category))

    df = pd.DataFrame(all_records)
    # a handful of rows in the raw chart sheets are footnotes/group-notes
    # ("لیست دروس تخصصی اجباری (گروه دوم) — جمع ۲۷ واحد...", "• در مجموع
    # باید حداقل ۳۹ واحد...") that happen to sit in the title column under
    # an active header block, so _sheet_to_records scoops them up as if
    # they were course rows. Every real course in this chart has a parsed
    # unit count, so dropping rows with no units removes these notes
    # without touching any actual course.
    df = df[df["واحد"].notna()].reset_index(drop=True)
    # «زبان انگلیسی عمومی» (۳ واحدی، چارت قدیمی) از انتخاب‌ها حذف شد: برای همه‌ی ورودی‌ها
    # الگوی درست همان زنجیره‌ی پیش‌دانشگاهی ترکیبی (۲) ← ترکیبی ۱ ← ۲ ← ۳ (هرکدام ۱ واحد، جمعا ۵ واحد،
    # در ترم‌های جدا به شرط پاس‌شدن پیش‌نیاز) است.
    df = df[df["نام درس"].map(normalize_name) != normalize_name("زبان انگلیسی عمومی")].reset_index(drop=True)
    # de-duplicate: a course may appear in more than one sub-table
    # (e.g. once as a named row, once inside an elective group) — keep the
    # richest (non-null units / longer prereq list) version.
    df["_prereq_len"] = df["پیش‌نیاز"].apply(len)
    df = (
        df.sort_values(["نام درس", "_prereq_len"], ascending=[True, False])
        .drop_duplicates(subset=["نام درس"], keep="first")
        .drop(columns=["_prereq_len"])
        .reset_index(drop=True)
    )
    df.attrs["chart_terms"] = _load_chart_terms(wb)
    return df


def _load_chart_terms(wb):
    """ترم پیشنهادی هر درس طبق شیت «چارت پیشنهادی» (کلید: normalize_name). اگر شیت نبود، {}."""
    import re as _re
    terms = {}
    try:
        cur = None
        for row in wb["چارت پیشنهادی"].iter_rows(values_only=True):
            a = row[0]
            if not isinstance(a, str):
                continue
            m = _re.match(r"ترم\s*([۰-۹0-9]+)", a)
            if m:
                cur = int(m.group(1).translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")))
                continue
            if cur and a != "عنوان درس" and not a.startswith(("این ترم", "چارت پیشنهادی")):
                for part in _re.split(r"\s+یا\s+", a):
                    terms[normalize_name(part)] = cur
    except Exception:
        return {}
    return terms


_OFFERED_RENAMES = {
    normalize_name("زبان انگلیسی پیش دانشگاهی-ترکیبی"): "زبان انگلیسی پیش‌دانشگاهی ترکیبی",
    normalize_name("زبان انگلیسی عمومی-ترکیبی(1)"): "زبان انگلیسی ترکیبی ۱",
    normalize_name("زبان انگلیسی عمومی-ترکیبی(2)"): "زبان انگلیسی ترکیبی ۲",
    normalize_name("زبان انگلیسی عمومی-ترکیبی(3)"): "زبان انگلیسی ترکیبی ۳",
    normalize_name("ریاضیات پایه(پیش نیاز دانشگاهی)"): "ریاضی پایه پیش‌دانشگاهی",
    normalize_name("آیین زندگی (اخلاق کاربردی)"): "آیین زندگی",
    normalize_name("اندیشه اسلامی 1 (مبدا ومعاد)"): "اندیشه اسلامی ۱",
    normalize_name("اندیشه اسلامی 2 (نبوت و امامت)"): "اندیشه اسلامی ۲",
    normalize_name("شبیه سازی کامپیوتر"): "شبیه‌سازی کامپیوتری",
    normalize_name("کارگاه کامپیوتر 1"): "کارگاه کامپیوتر",
    normalize_name("کارآموزی طرح پویش (1) کاردانی و کارشناسی"): "کارآموزی ۱",
    # در آموزشیار نام این درس بلندتر است و با نام چارت فرق دارد؛ بدون این تطبیق هیچ بخشی برایش ساخته نمی‌شد.
    normalize_name("اندیشه های امامین انقلاب اسلامی و وصایای حضرت امام خمینی(ره)"): "اندیشه‌ها و وصایای امام خمینی",
}


def load_offered(xlsx_path, summer=False):
    """Returns a pandas DataFrame of every offered class-section this term,
    combining the specialized and general offered-course sheets. Not every
    sheet necessarily has a 'کد درس' / 'کد ارائه کلاس' column (depends on
    what the source export included) — missing ones are filled with None."""
    frames = []
    suffix = " تابستان" if summer else " این ترم"
    for sheet in ["دروس ارائه‌شده" + suffix, "دروس عمومی ارائه‌شده" + suffix]:
        try:
            # کدها باید «متن» خوانده شوند: pandas رشته‌ی عددی را عدد می‌کند و صفرهای اول کد ارائه
            # (مثل 0599 برای تربیت‌بدنی) از دست می‌رود.
            t = pd.read_excel(xlsx_path, sheet_name=sheet, header=3,
                              dtype={"کد درس": str, "کد ارائه کلاس": str, "شماره کلاس": str})
        except Exception:
            continue
        t = t.dropna(how="all")
        for col in ("کد درس", "کد ارائه کلاس"):
            if col not in t.columns:
                t[col] = None
        t["_src"] = "specialized" if sheet.startswith("دروس ارائه‌شده") else "general"
        frames.append(t)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["نام درس"].notna()]
    # «علوم و معارف دفاع مقدس و مقاومت» درس جداگانه‌ای در چارت نیست و جز «آشنایی با دفاع مقدس» حساب نمی‌شود؛ از فهرست ارائه‌ها حذف می‌شود
    df = df[~df["نام درس"].map(lambda n: normalize_name(n).startswith("علوم و معارف دفاع مقدس"))].reset_index(drop=True)
    # همه‌ی دروس برگه‌ی «دروس ارائه‌شده این ترم» (تخصصی/پایه/اختیاری رشته) در دانشکده‌ی فنی برگزار می‌شوند؛
    # در ۱۳۱ بخشی که مکانشان ثبت شده، هر ۱۳۱ تا «فنی» است. پس مکان خالی این برگه «فنی» حساب می‌شود.
    # (دروس عمومی/معارف/زبان در برگه‌ی دیگر جدا مدیریت می‌شوند و مکان خالی‌شان «نامشخص» می‌ماند.)
    if "ساختمان" not in df.columns:
        df["ساختمان"] = None
    df.loc[(df["_src"] == "specialized") & df["ساختمان"].isna(), "ساختمان"] = "فنی"
    # همه‌ی کلاس‌های زبان (انگلیسی/خارجی/فارسی) مجازی‌اند، نه حضوری در یک ساختمان.
    _lang = df["نام درس"].map(lambda n: normalize_name(n).startswith(("زبان انگلیسی", "زبان خارجی", "زبان فارسی")))
    df.loc[_lang, "ساختمان"] = "مجازی"
    # تربیت‌بدنی و ورزش: مکان در HTML نیامده؛ پیش‌فرض دانشکده‌ی پیامبر.
    _pe = df["نام درس"].map(lambda n: normalize_name(n).startswith(("تربیت بدنی", "ورزش")) and "ویژه" not in n) & df["ساختمان"].isna()
    df.loc[_pe, "ساختمان"] = "پیامبر"
    df = df.drop(columns=["_src"])
    # نام زنجیره‌ی زبان انگلیسی در سامانه ("زبان انگلیسی عمومی-ترکیبی(1)" و ...) با نام چارت
    # ("زبان انگلیسی ترکیبی ۱" و ...) فرق دارد و تطبیق پیشوندی آن را اشتباه به درس ۳واحدی
    # «زبان انگلیسی عمومی» وصل می‌کرد. اینجا نام‌ها به نام چارت یکسان می‌شوند.
    df["نام درس"] = df["نام درس"].map(lambda n: _OFFERED_RENAMES.get(normalize_name(n), n))
    # نام درس در آموزشیار گاهی فقط در فاصله/نیم‌فاصله با چارت فرق دارد («ساختمان داده هاوالگوریتمها»
    # در برابر «ساختمان داده‌ها و الگوریتم‌ها»)؛ چنین نام‌هایی به نام چارت یکسان می‌شوند تا
    # در همه‌ی ماژول‌ها (اعتبارسنجی دستی، بهینه‌ساز، دستیار) درس در چارت پیدا شود.
    try:
        chart_names = load_chart(xlsx_path)["نام درس"].tolist()
        by_loose = {loose_key(n): n for n in chart_names}
        df["نام درس"] = df["نام درس"].map(lambda n: by_loose.get(loose_key(n), n))
    except Exception:
        pass
    return df.reset_index(drop=True)


if __name__ == "__main__":
    path = "/mnt/user-data/outputs/chart_computer_science.xlsx"
    chart = load_chart(path)
    offered = load_offered(path)
    print(f"چارت: {len(chart)} درس منحصربه‌فرد بارگذاری شد.")
    print(chart.head(10).to_string())
    print()
    print(f"دروس ارائه‌شده‌ی این ترم: {len(offered)} بخش/کلاس.")
    print(offered[["نام درس", "استاد"]].head(5).to_string())
