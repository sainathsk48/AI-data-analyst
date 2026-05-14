import csv
import difflib
import io
import json
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="AI Data Analyst", layout="wide")

APP_VERSION = "2026-05-14-common-summary-v5"


def inject_professional_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --surface: #ffffff;
                --surface-soft: #f7f9fc;
                --border: #d9e1ec;
                --text-muted: #64748b;
                --accent: #1d4ed8;
                --accent-soft: #eff6ff;
            }

            .stApp {
                background: #f4f7fb;
            }

            section.main > div {
                max-width: 1280px;
                padding-top: 2.25rem;
            }

            [data-testid="stSidebar"] {
                background: #eef3f8;
                border-right: 1px solid var(--border);
            }

            h1, h2, h3 {
                letter-spacing: 0;
            }

            h1 {
                font-size: 2.2rem;
                line-height: 1.15;
                margin-bottom: 0.25rem;
            }

            [data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 1rem 1.1rem;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            }

            [data-testid="stMetricLabel"] {
                color: var(--text-muted);
                font-size: 0.82rem;
            }

            [data-testid="stMetricValue"] {
                font-weight: 700;
                color: #0f172a;
            }

            .analyst-header {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 1.2rem 1.35rem;
                margin-bottom: 1rem;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            }

            .analyst-kicker {
                color: var(--accent);
                font-weight: 700;
                font-size: 0.82rem;
                margin-bottom: 0.25rem;
                text-transform: uppercase;
            }

            .analyst-subtitle {
                color: var(--text-muted);
                margin-top: 0.35rem;
                max-width: 760px;
            }

            .source-note {
                background: var(--accent-soft);
                border: 1px solid #bfdbfe;
                border-radius: 8px;
                color: #1e3a8a;
                padding: 0.8rem 1rem;
                margin: 0.75rem 0 1rem;
            }

            div[data-testid="stTabs"] button {
                font-weight: 600;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@dataclass
class ParseResult:
    dataframe: pd.DataFrame
    encoding: str
    delimiter: str
    skipped_bad_lines: bool


ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin1")
DELIMITERS: tuple[str | None, ...] = (None, ",", ";", "\t", "|")
NUMERIC_WORDS = ("average", "avg", "mean", "sum", "total", "minimum", "min", "maximum", "max", "median")
AGGREGATE_WORDS = NUMERIC_WORDS + ("top", "most", "common", "count", "many")
MAX_WORDS = ("highest", "largest", "maximum", "max", "biggest", "greatest")
MIN_WORDS = ("lowest", "smallest", "minimum", "min", "least")
QUESTION_STOPWORDS = {
    "a",
    "about",
    "all",
    "an",
    "and",
    "answer",
    "are",
    "assigned",
    "csv",
    "data",
    "did",
    "file",
    "find",
    "for",
    "get",
    "give",
    "got",
    "has",
    "have",
    "i",
    "in",
    "is",
    "me",
    "my",
    "number",
    "of",
    "option",
    "options",
    "please",
    "project",
    "record",
    "row",
    "show",
    "tell",
    "the",
    "this",
    "to",
    "value",
    "what",
    "which",
    "who",
    "with",
}
IDENTITY_COLUMN_HINTS = ("name", "email", "student", "candidate", "employee", "user", "person")
IDENTIFIER_COLUMN_HINTS = ("id", "email", "phone", "mobile", "url", "link", "address")
MONEY_METRIC_HINTS = ("sales", "revenue", "amount", "price", "profit", "cost", "spend", "income", "total")
ADDITIVE_METRIC_HINTS = MONEY_METRIC_HINTS + ("quantity", "qty", "units", "count")
RATE_METRIC_HINTS = ("discount", "rate", "ratio", "percent", "percentage", "margin")
ENTITY_ALIASES = {
    "customer": ("customer", "client", "buyer", "name"),
    "product": ("product", "item", "sku"),
    "region": ("region", "state", "city", "country", "market", "location"),
    "category": ("category", "segment", "type", "group"),
    "student": ("student", "name", "candidate"),
    "employee": ("employee", "name", "staff"),
}
COLUMN_SYNONYMS = {
    "option": {"option", "options", "choice", "choices", "project", "projects", "assigned", "allocation", "preference"},
    "project": {"project", "projects", "assignment", "assigned", "allocation", "topic", "task"},
    "group": {"group", "groups", "team", "section"},
    "batch": {"batch", "cohort"},
    "course": {"course", "class", "program", "subject"},
    "name": {"name", "student", "person", "candidate", "customer", "employee", "who"},
    "email": {"email", "mail", "gmail"},
    "score": {"score", "scores", "mark", "marks", "grade", "grades", "point", "points"},
    "amount": {"amount", "value", "price", "cost", "sales", "revenue", "total"},
}
PREFERRED_GEMINI_MODELS = (
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.5-flash-lite",
    "models/gemini-2.0-flash-lite",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro",
)


def dedupe_columns(columns: list[Any]) -> list[str]:
    seen: dict[str, int] = {}
    cleaned: list[str] = []

    for index, column in enumerate(columns, start=1):
        base = str(column).strip() if column is not None else ""
        base = base or f"Column {index}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        cleaned.append(base if count == 1 else f"{base}_{count}")

    return cleaned


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = dedupe_columns(list(df.columns))
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    for column in df.select_dtypes(include=["object", "string"]).columns:
        series = df[column]
        non_empty = series.dropna().astype(str).str.strip()
        if non_empty.empty:
            continue

        numeric_text = (
            non_empty.str.replace(r"[\$,]", "", regex=True)
            .str.replace("%", "", regex=False)
            .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        )
        converted = pd.to_numeric(numeric_text, errors="coerce")
        if converted.notna().mean() >= 0.85:
            full_text = (
                series.astype(str)
                .str.strip()
                .str.replace(r"[\$,]", "", regex=True)
                .str.replace("%", "", regex=False)
                .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
            )
            df[column] = pd.to_numeric(full_text, errors="coerce")

    return df.reset_index(drop=True)


def candidate_delimiters(text: str) -> list[str | None]:
    sample = text[:65536]
    candidates: list[str | None] = []

    try:
        detected = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        candidates.append(detected)
    except csv.Error:
        candidates.append(None)

    candidates.extend(DELIMITERS)

    deduped: list[str | None] = []
    for delimiter in candidates:
        if delimiter not in deduped:
            deduped.append(delimiter)
    return deduped


def delimiter_label(delimiter: str) -> str:
    labels = {
        ",": "comma (,)",
        ";": "semicolon (;)",
        "\t": "tab",
        "|": "pipe (|)",
        "auto-detected": "auto-detected",
    }
    return labels.get(delimiter, delimiter or "auto-detected")


def read_csv_bytes(raw: bytes) -> ParseResult:
    attempts: list[str] = []

    for encoding in ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError as exc:
            attempts.append(f"{encoding}: {exc}")
            continue

        best: ParseResult | None = None
        best_score = -1
        for delimiter in candidate_delimiters(text):
            try:
                df = pd.read_csv(
                    io.StringIO(text),
                    sep=delimiter,
                    engine="python",
                    on_bad_lines="skip",
                    skip_blank_lines=True,
                )
            except Exception as exc:  # noqa: BLE001 - collect parser attempts for a useful user error.
                label = "auto" if delimiter is None else repr(delimiter)
                attempts.append(f"{encoding}/{label}: {exc}")
                continue

            df = clean_dataframe(df)
            if df.empty or df.shape[1] == 0:
                continue

            score = (df.shape[1] * 1000) + min(df.shape[0], 1000)
            if df.shape[1] == 1:
                score -= 500
            if score > best_score:
                best = ParseResult(
                    dataframe=df,
                    encoding=encoding,
                    delimiter="auto-detected" if delimiter is None else delimiter,
                    skipped_bad_lines=True,
                )
                best_score = score

        if best is not None:
            return best

    message = "Could not read this CSV. Check that the file is a valid comma, semicolon, tab, or pipe separated text file."
    if attempts:
        message += "\n\nParser details:\n" + "\n".join(attempts[:6])
    raise ValueError(message)


@st.cache_data(show_spinner=False)
def load_csv(raw: bytes) -> ParseResult:
    return read_csv_bytes(raw)


@st.cache_data(show_spinner=False)
def profile_csv(df: pd.DataFrame) -> dict[str, Any]:
    return build_profile(df)


def format_value(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if abs(number) >= 100:
            return f"{number:,.0f}"
        return f"{number:,.2f}".rstrip("0").rstrip(".")
    return str(value)


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([A-Za-z])", r"\1 \2", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_tokens(value: Any) -> set[str]:
    return {token for token in normalize_text(value).split() if len(token) > 1 or token.isdigit()}


def display_name(column: str) -> str:
    text = re.sub(r"[_\-]+", " ", str(column)).strip()
    return text or str(column)


def entity_display_name(column: str) -> str:
    text = display_name(column).lower()
    return text[:-5] if text.endswith(" name") else text


def pluralize(label: str, count: int) -> str:
    if count == 1:
        return label
    if label.endswith("y"):
        return f"{label[:-1]}ies"
    if label.endswith("s"):
        return label
    return f"{label}s"


def column_contains(column: str, hints: tuple[str, ...]) -> bool:
    normalized = normalize_text(column)
    tokens = set(normalized.split())
    return any(hint in tokens or hint in normalized for hint in hints)


def is_identifier_like(column: str, series: pd.Series | None = None) -> bool:
    if column_contains(column, IDENTIFIER_COLUMN_HINTS):
        return True

    normalized = normalize_text(column)
    if normalized in {"name", "full name"} or normalized.endswith(" name"):
        if series is None:
            return True
        non_empty = series.dropna().astype(str).str.strip()
        return not non_empty.empty and (non_empty.nunique() / max(len(non_empty), 1)) > 0.7

    if series is not None and not series.empty and not pd.api.types.is_numeric_dtype(series):
        non_empty = series.dropna().astype(str).str.strip()
        if not non_empty.empty and (non_empty.nunique() / max(len(non_empty), 1)) > 0.9:
            return True

    return False


def is_rate_metric(column: str, series: pd.Series | None = None) -> bool:
    if not column_contains(column, RATE_METRIC_HINTS):
        return False
    if series is None:
        return True
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return numeric.empty or numeric.abs().quantile(0.95) <= 1


def is_additive_metric(column: str) -> bool:
    return column_contains(column, ADDITIVE_METRIC_HINTS) and not column_contains(column, RATE_METRIC_HINTS)


def format_metric_value(column: str, value: Any, series: pd.Series | None = None, include_raw_rate: bool = False) -> str:
    if pd.isna(value):
        return "n/a"
    if is_rate_metric(column, series):
        percentage = f"{float(value) * 100:.1f}".rstrip("0").rstrip(".")
        if include_raw_rate:
            return f"{format_value(value)} ({percentage}%)"
        return f"{percentage}%"
    return format_value(value)


def metric_priority(column: str) -> int:
    normalized = normalize_text(column)
    score = 0
    for priority, hint in enumerate(("sales", "revenue", "profit", "amount", "total", "quantity", "discount", "price", "score")):
        if hint in normalized:
            score += 100 - priority
    if is_identifier_like(column):
        score -= 100
    return score


def choose_primary_metric(df: pd.DataFrame, numeric_columns: list[str]) -> str | None:
    candidates = [column for column in numeric_columns if not is_identifier_like(column, df[column])]
    if not candidates:
        return numeric_columns[0] if numeric_columns else None
    return sorted(candidates, key=metric_priority, reverse=True)[0]


def useful_categorical_columns(df: pd.DataFrame, categorical_columns: list[str], include_constants: bool = False) -> list[str]:
    useful: list[str] = []
    row_count = max(len(df), 1)

    for column in categorical_columns:
        series = df[column].dropna().astype(str).str.strip()
        series = series[series != ""]
        if series.empty or is_identifier_like(column, series):
            continue

        unique_count = series.nunique()
        if unique_count == 1 and include_constants:
            useful.append(column)
        elif 1 < unique_count <= min(50, max(2, row_count // 20)):
            useful.append(column)

    return useful


def infer_dataset_kind(profile: dict[str, Any]) -> str:
    names = " ".join(normalize_text(column) for column in profile["column_names"])
    if any(word in names for word in ("sales", "revenue", "order", "customer", "product", "discount")):
        return "sales or customer activity data"
    if any(word in names for word in ("student", "course", "batch", "university", "marks", "score")):
        return "student or course data"
    if any(word in names for word in ("employee", "salary", "department", "role")):
        return "employee data"
    if any(word in names for word in ("date", "time", "month", "year")):
        return "time-based records"
    return "tabular business data"


def detect_date_columns(df: pd.DataFrame) -> dict[str, pd.Series]:
    detected: dict[str, pd.Series] = {}

    for column in df.columns:
        series = df[column].dropna()
        if series.empty:
            continue

        if pd.api.types.is_datetime64_any_dtype(df[column]):
            parsed = pd.to_datetime(df[column], errors="coerce")
        elif pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            sample = series.astype(str).str.strip()
            if sample.nunique(dropna=True) < 2:
                continue
            date_like_share = sample.head(500).str.contains(r"\d{1,4}[-/]\d{1,2}|\d{1,2}[-/]\d{1,4}", regex=True).mean()
            if not column_contains(column, ("date", "time", "month", "year")) and date_like_share < 0.6:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed_sample = pd.to_datetime(sample.head(500), errors="coerce")
            if parsed_sample.notna().mean() < 0.75:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(df[column], errors="coerce")
        else:
            continue

        if parsed.notna().sum() >= 2:
            detected[column] = parsed

    return detected


def numeric_stats(df: pd.DataFrame, numeric_columns: list[str]) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for column in numeric_columns[:12]:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outliers = 0
        if iqr > 0:
            outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())

        stats.append(
            {
                "column": column,
                "count": int(series.count()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "min": float(series.min()),
                "max": float(series.max()),
                "std": float(series.std(ddof=0)),
                "outliers": outliers,
            }
        )

    return stats


def categorical_stats(df: pd.DataFrame, categorical_columns: list[str]) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    row_count = max(len(df), 1)

    for column in useful_categorical_columns(df, categorical_columns, include_constants=True)[:10]:
        series = df[column].dropna().astype(str).str.strip()
        series = series[series != ""]
        if series.empty:
            continue

        top_values = series.value_counts().head(5)
        stats.append(
            {
                "column": column,
                "unique": int(series.nunique()),
                "top_values": [
                    {
                        "value": str(index),
                        "count": int(count),
                        "share": round((int(count) / row_count) * 100, 1),
                    }
                    for index, count in top_values.items()
                ],
            }
        )

    return stats


def correlation_pairs(df: pd.DataFrame, numeric_columns: list[str]) -> list[dict[str, Any]]:
    if len(numeric_columns) < 2:
        return []

    corr = df[numeric_columns].corr(numeric_only=True)
    pairs: list[dict[str, Any]] = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(value) >= 0.6:
                pairs.append({"left": left, "right": right, "correlation": float(value)})

    return sorted(pairs, key=lambda item: abs(item["correlation"]), reverse=True)[:8]


def trend_stats(df: pd.DataFrame, date_columns: dict[str, pd.Series], numeric_columns: list[str]) -> list[dict[str, Any]]:
    trends: list[dict[str, Any]] = []
    if not date_columns or not numeric_columns:
        return trends

    date_column, parsed_dates = next(iter(date_columns.items()))
    working = df.copy()
    working["_detected_date"] = parsed_dates
    working = working.dropna(subset=["_detected_date"]).sort_values("_detected_date")
    if len(working) < 4:
        return trends

    window = max(1, len(working) // 5)
    early = working.head(window)
    recent = working.tail(window)

    for column in numeric_columns[:5]:
        early_mean = pd.to_numeric(early[column], errors="coerce").mean()
        recent_mean = pd.to_numeric(recent[column], errors="coerce").mean()
        if pd.isna(early_mean) or pd.isna(recent_mean) or early_mean == 0:
            continue

        change_pct = ((recent_mean - early_mean) / abs(early_mean)) * 100
        trends.append(
            {
                "date_column": date_column,
                "metric": column,
                "early_mean": float(early_mean),
                "recent_mean": float(recent_mean),
                "change_pct": float(change_pct),
            }
        )

    return trends[:5]


def build_profile(df: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
    date_columns = detect_date_columns(df)
    categorical_columns = [
        column
        for column in df.columns
        if column not in numeric_columns and column not in date_columns
    ]

    missing = df.isna().sum()
    missing_summary = [
        {
            "column": column,
            "missing": int(count),
            "missing_pct": round((int(count) / max(len(df), 1)) * 100, 1),
        }
        for column, count in missing.sort_values(ascending=False).items()
        if count > 0
    ][:10]

    date_summary = []
    for column, parsed in date_columns.items():
        valid = parsed.dropna()
        if valid.empty:
            continue
        date_summary.append(
            {
                "column": column,
                "start": valid.min().date().isoformat(),
                "end": valid.max().date().isoformat(),
                "valid_dates": int(valid.count()),
            }
        )

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "date_columns": list(date_columns.keys()),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_summary": missing_summary,
        "numeric_stats": numeric_stats(df, numeric_columns),
        "categorical_stats": categorical_stats(df, categorical_columns),
        "date_summary": date_summary,
        "correlations": correlation_pairs(df, numeric_columns),
        "trends": trend_stats(df, date_columns, numeric_columns),
    }


def choose_label_column(df: pd.DataFrame, question: str = "", target_column: str | None = None) -> str | None:
    question_tokens = text_tokens(question)
    scored: list[tuple[int, str]] = []

    for column in df.columns:
        if column == target_column:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            continue

        normalized = normalize_text(column)
        column_tokens = set(normalized.split())
        score = 0

        if normalized in {"name", "full name"} or normalized.endswith(" name"):
            score += 24

        for alias, hints in ENTITY_ALIASES.items():
            if alias in question_tokens and any(hint in normalized or hint in column_tokens for hint in hints):
                score += 30

        if any(hint in normalized for hint in IDENTITY_COLUMN_HINTS):
            score += 12
        if column_contains(column, ("customer", "product", "region", "category", "segment", "course", "batch")):
            score += 10
        if is_identifier_like(column, series):
            score -= 4

        unique_count = series.dropna().astype(str).str.strip().nunique()
        if 1 < unique_count < max(len(df) * 0.9, 2):
            score += 3

        if score > 0:
            scored.append((score, column))

    if not scored:
        return None
    return sorted(scored, key=lambda item: item[0], reverse=True)[0][1]


def describe_extreme_row(df: pd.DataFrame, metric: str, direction: str) -> str | None:
    numeric = pd.to_numeric(df[metric], errors="coerce").dropna()
    if numeric.empty:
        return None

    value = numeric.max() if direction == "highest" else numeric.min()
    matches = df.loc[numeric[numeric == value].index]
    label_column = choose_label_column(df, target_column=metric)
    value_text = format_metric_value(metric, value, df[metric])

    if label_column and not matches.empty:
        labels = matches[label_column].dropna().astype(str).str.strip()
        labels = labels[labels != ""].drop_duplicates().head(3).tolist()
        if labels:
            label_text = ", ".join(f"**{label}**" for label in labels)
            if len(matches) > len(labels):
                return f"The {direction} **{display_name(metric)}** is **{value_text}**. Examples with this value include {label_text}."
            return f"The {direction} **{display_name(metric)}** is **{value_text}**, seen for {label_text}."

    return f"The {direction} **{display_name(metric)}** is **{value_text}**."


def summary_value(column: str, value: Any, df: pd.DataFrame) -> str:
    text = format_metric_value(column, value, df[column], include_raw_rate=True)
    if len(text) > 120:
        return text[:117].rstrip() + "..."
    return text


def build_small_file_summary(profile: dict[str, Any], df: pd.DataFrame) -> str:
    rows = profile["rows"]
    columns = profile["columns"]
    lines = [
        "### Summary",
        f"This is a small file with **{rows:,} row{'s' if rows != 1 else ''}** and **{columns:,} column{'s' if columns != 1 else ''}**.",
    ]

    if profile["missing_cells"]:
        lines.append(f"It has **{profile['missing_cells']:,} blank value{'s' if profile['missing_cells'] != 1 else ''}**.")
    else:
        lines.append("Nothing is blank.")

    lines.append("\n#### What It Contains")
    for row_number, (_, row) in enumerate(df.head(5).iterrows(), start=1):
        values = [
            f"**{display_name(column)}**: {summary_value(column, row[column], df)}"
            for column in df.columns[:10]
        ]
        prefix = "The entry contains" if rows == 1 else f"Row {row_number} contains"
        lines.append(f"- {prefix}: " + "; ".join(values) + ".")

    if columns > 10:
        lines.append(f"- There are {columns - 10:,} more columns not shown in this short summary.")

    lines.append("\n#### Plain Meaning")
    if rows == 1:
        lines.append("This file is not a trend or report yet. It is one entry, so the useful summary is the exact values above.")
    else:
        lines.append("This file has only a few rows, so the most useful view is to read each row and compare the values directly.")

    return "\n".join(lines)


def build_plain_english_summary(profile: dict[str, Any], df: pd.DataFrame) -> str:
    rows = profile["rows"]
    columns = profile["columns"]
    if rows <= 5:
        return build_small_file_summary(profile, df)

    numeric_columns = profile["numeric_columns"]
    categorical_columns = profile["categorical_columns"]
    primary_metric = choose_primary_metric(df, numeric_columns)
    lines = [
        "### Summary",
        f"This file has **{rows:,} rows** and **{columns:,} columns**.",
    ]

    if profile["date_summary"]:
        item = profile["date_summary"][0]
        lines.append(f"It covers **{item['start']} to {item['end']}** based on **{display_name(item['column'])}**.")

    if profile["missing_cells"]:
        missing_cols = profile["missing_summary"][:5]
        readable = ", ".join(
            f"**{display_name(item['column'])}** ({item['missing_pct']}%)" for item in missing_cols
        )
        lines.append(f"It has **{profile['missing_cells']:,} blank cells**. Check these columns first: {readable}.")
    else:
        lines.append("Nothing is blank.")

    if profile["duplicate_rows"]:
        lines.append(f"It has **{profile['duplicate_rows']:,} duplicate rows**, so those rows should be checked.")

    takeaways: list[str] = []
    if primary_metric:
        numeric = pd.to_numeric(df[primary_metric], errors="coerce").dropna()
        if not numeric.empty:
            metric_name = display_name(primary_metric)
            if is_additive_metric(primary_metric):
                takeaways.append(
                    f"Total **{metric_name}** is **{format_metric_value(primary_metric, numeric.sum(), df[primary_metric])}**. "
                    f"The average row has **{format_metric_value(primary_metric, numeric.mean(), df[primary_metric])}**."
                )
            else:
                takeaways.append(
                    f"**{metric_name}** ranges from **{format_metric_value(primary_metric, numeric.min(), df[primary_metric], include_raw_rate=True)}** "
                    f"to **{format_metric_value(primary_metric, numeric.max(), df[primary_metric], include_raw_rate=True)}**."
                )
            highest = describe_extreme_row(df, primary_metric, "highest")
            if highest:
                takeaways.append(highest)

    useful_categories = useful_categorical_columns(df, categorical_columns)
    if useful_categories and primary_metric:
        for category in useful_categories[:3]:
            grouped = (
                df[[category, primary_metric]]
                .dropna()
                .groupby(category, dropna=True)[primary_metric]
                .sum()
                .sort_values(ascending=False)
                .head(1)
            )
            if not grouped.empty:
                takeaways.append(
                    f"By **{display_name(category)}**, **{grouped.index[0]}** has the highest total "
                    f"**{display_name(primary_metric)}**: **{format_metric_value(primary_metric, grouped.iloc[0], df[primary_metric])}**."
                )

    if profile["categorical_stats"]:
        for item in profile["categorical_stats"][:3]:
            top = item["top_values"][0]
            if item["unique"] == 1:
                takeaways.append(f"Every row has **{display_name(item['column'])}** as **{top['value']}**.")
            else:
                takeaways.append(
                    f"Most common **{display_name(item['column'])}** is **{top['value']}** "
                    f"({top['count']:,} rows, {top['share']}%)."
                )

    if takeaways:
        lines.append("\n#### Key Points")
        lines.extend(f"- {item}" for item in takeaways[:8])

    number_notes = []
    for item in profile["numeric_stats"][:5]:
        number_notes.append(
            f"**{display_name(item['column'])}**: lowest **{format_metric_value(item['column'], item['min'], df[item['column']], include_raw_rate=True)}**, "
            f"highest **{format_metric_value(item['column'], item['max'], df[item['column']], include_raw_rate=True)}**"
        )

    if number_notes:
        lines.append("\n#### Number Ranges")
        lines.extend(f"- {note}" for note in number_notes)

    if profile["trends"]:
        lines.append("\n#### Time Movement")
        for item in profile["trends"][:4]:
            direction = "went up" if item["change_pct"] > 0 else "went down"
            lines.append(
                f"- **{display_name(item['metric'])}** {direction} by about **{abs(item['change_pct']):.1f}%** from the first part of the file to the last."
            )

    lines.append("\n#### What To Check Next")
    if profile["missing_cells"]:
        lines.append("Check the blank cells first, then use the Ask tab to ask about a specific person, product, project, or column.")
    else:
        lines.append("Use the Ask tab to ask about a specific person, product, project, or column. The answer will be calculated from the CSV.")

    return "\n".join(lines)


def find_secret_value(*names: str) -> str | None:
    for name in names:
        env_value = os.getenv(name)
        if env_value:
            return env_value

        try:
            value = st.secrets.get(name)
        except Exception:  # noqa: BLE001 - Streamlit raises when no secrets file exists.
            value = None
        if value:
            return str(value)
    return None


def gemini_api_key() -> str | None:
    return find_secret_value("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY")


def choose_gemini_model(genai: Any, api_key: str) -> str:
    genai.configure(api_key=api_key)
    available: list[str] = []

    try:
        for model in genai.list_models():
            methods = set(getattr(model, "supported_generation_methods", []) or [])
            name = getattr(model, "name", "")
            if name and "generateContent" in methods:
                available.append(name)
    except Exception:  # noqa: BLE001 - fall back to preferred current model names.
        available = []

    if available:
        available_set = set(available)
        short_available = {name.removeprefix("models/"): name for name in available}
        for candidate in PREFERRED_GEMINI_MODELS:
            if candidate in available_set:
                return candidate
            short_candidate = candidate.removeprefix("models/")
            if short_candidate in short_available:
                return short_available[short_candidate]
        return available[0]

    return PREFERRED_GEMINI_MODELS[0]


def generate_gemini_summary(profile: dict[str, Any], df: pd.DataFrame, api_key: str) -> str:
    import google.generativeai as genai

    model = genai.GenerativeModel(choose_gemini_model(genai, api_key))
    sample_csv = df.head(30).to_csv(index=False)
    prompt = f"""
You are a careful data analyst. Write a concise, understandable English summary of this CSV.
Avoid hallucinating. Base your answer only on the supplied profile and sample rows.
Include the dataset purpose if it is obvious, the key patterns, data quality warnings, and 3 next actions.

Profile JSON:
{json.dumps(profile, default=str)}

First rows as CSV:
{sample_csv}
"""
    response = model.generate_content(prompt)
    return (getattr(response, "text", "") or "").strip()


def column_alias_tokens(column: str) -> set[str]:
    tokens = text_tokens(column)
    normalized = normalize_text(column)
    if "option" in normalized:
        tokens.update(COLUMN_SYNONYMS["option"])
    for canonical, synonyms in COLUMN_SYNONYMS.items():
        if canonical in tokens or canonical in normalized:
            tokens.update(synonyms)
    if "number" in tokens:
        tokens.update({"no", "num"})
    if "no" in tokens or "num" in tokens:
        tokens.add("number")
    return tokens


def option_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if "option" in column_alias_tokens(column)
    ]


def numbered_column_base(column: str) -> str | None:
    normalized = normalize_text(column)
    match = re.match(r"(.+?)\s+[0-9]+$", normalized)
    if not match:
        return None
    return match.group(1).strip()


def numbered_family_columns(df: pd.DataFrame, question: str) -> list[str]:
    question_tokens = text_tokens(question)
    families: dict[str, list[str]] = {}

    for column in df.columns:
        base = numbered_column_base(column)
        if base:
            families.setdefault(base, []).append(column)

    matches: list[str] = []
    for base, columns in families.items():
        if len(columns) < 2:
            continue
        base_tokens = text_tokens(base)
        base_aliases = set(base_tokens)
        for canonical, synonyms in COLUMN_SYNONYMS.items():
            if canonical in base_tokens:
                base_aliases.update(synonyms)
        if question_tokens & base_aliases:
            matches.extend(columns)

    return filter_option_columns_by_question(matches, question) if matches else []


def direct_project_columns(df: pd.DataFrame, question: str) -> list[str]:
    question_tokens = text_tokens(question)
    if not ({"project", "projects", "assignment", "assigned", "allocation", "topic", "task"} & question_tokens):
        return []

    candidates = []
    for column in df.columns:
        aliases = column_alias_tokens(column)
        if "project" in aliases and "option" not in aliases:
            candidates.append((column_score(column, question), column))

    candidates = [(score, column) for score, column in candidates if score >= 3]
    if not candidates:
        return []

    best_score = max(score for score, _ in candidates)
    return [column for score, column in candidates if score == best_score]


def requested_option_numbers(question: str) -> set[str]:
    normalized = normalize_text(question)
    numbers = set(re.findall(r"\boption\s*([0-9]+)\b", normalized))
    if not numbers:
        tokens = normalized.split()
        for index, token in enumerate(tokens[:-1]):
            if token in {"option", "options", "project", "projects"} and tokens[index + 1].isdigit():
                numbers.add(tokens[index + 1])
    return numbers


def filter_option_columns_by_question(columns: list[str], question: str) -> list[str]:
    numbers = requested_option_numbers(question) | {token for token in text_tokens(question) if token.isdigit()}
    if not numbers:
        return columns
    filtered = [
        column
        for column in columns
        if text_tokens(column) & numbers
    ]
    return filtered or columns


def column_score(column: str, question: str) -> int:
    normalized_question = normalize_text(question)
    normalized_column = normalize_text(column)
    question_tokens = text_tokens(question)
    alias_tokens = column_alias_tokens(column)
    score = 0

    if normalized_column and normalized_column in normalized_question:
        score += 10

    overlap = question_tokens & alias_tokens
    score += len(overlap) * 3
    if overlap:
        score += 2

    if {"project", "number"} <= question_tokens and "project" in alias_tokens:
        score += 5
    if {"project", "no"} <= question_tokens and "project" in alias_tokens:
        score += 5
    if "project" in question_tokens and "option" in alias_tokens:
        score += 6

    close_matches = difflib.get_close_matches(normalized_column, [normalized_question], n=1, cutoff=0.82)
    if close_matches:
        score += 3

    return score


def column_lookup(df: pd.DataFrame, question: str) -> str | None:
    scored = sorted(
        ((column_score(column, question), column) for column in df.columns),
        key=lambda item: item[0],
        reverse=True,
    )
    if scored and scored[0][0] >= 3:
        return scored[0][1]
    return None


def matching_target_columns(df: pd.DataFrame, question: str) -> list[str]:
    question_tokens = text_tokens(question)
    real_project_columns = direct_project_columns(df, question)
    if real_project_columns:
        return real_project_columns

    family_columns = numbered_family_columns(df, question)
    if family_columns:
        return family_columns

    option_like_columns = option_columns(df)
    if option_like_columns and ({"option", "options", "choice", "choices"} & question_tokens):
        return filter_option_columns_by_question(option_like_columns, question)

    scored = [
        (column_score(column, question), column)
        for column in df.columns
    ]
    positive = [(score, column) for score, column in scored if score >= 3]
    if not positive:
        return []

    if "option" in question_tokens or "options" in question_tokens or "project" in question_tokens:
        positive_option_columns = [
            column
            for score, column in positive
            if "option" in column_alias_tokens(column)
        ]
        if positive_option_columns and not any(token.isdigit() for token in question_tokens):
            return positive_option_columns

    top_score = max(score for score, _ in positive)
    return [column for score, column in positive if score == top_score]


def question_entity_tokens(question: str, target_column: str | None) -> set[str]:
    target_tokens = column_alias_tokens(target_column or "")
    ignored = QUESTION_STOPWORDS | set(AGGREGATE_WORDS) | target_tokens
    return {token for token in text_tokens(question) if token not in ignored and not token.isdigit()}


def row_display_label(row: pd.Series, matched_columns: set[str], target_column: str) -> str:
    for column in row.index:
        normalized = normalize_text(column)
        if column != target_column and any(hint in normalized for hint in IDENTITY_COLUMN_HINTS):
            value = row[column]
            if pd.notna(value) and str(value).strip():
                return f"{column} **{format_value(value)}**"

    for column in matched_columns:
        value = row[column]
        if pd.notna(value) and str(value).strip():
            return f"{column} **{format_value(value)}**"

    return "the matching row"


def find_matching_rows(df: pd.DataFrame, question: str, target_column: str) -> tuple[pd.DataFrame, dict[int, set[str]]]:
    tokens = question_entity_tokens(question, target_column)
    if not tokens:
        return pd.DataFrame(), {}

    scores = pd.Series(0, index=df.index, dtype="int64")
    matched_columns_by_index: dict[int, set[str]] = {}
    search_columns = [
        column
        for column in df.columns
        if column != target_column and not pd.api.types.is_numeric_dtype(df[column])
    ]

    for column in search_columns:
        normalized_series = (
            df[column]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9]+", " ", regex=True)
        )
        column_hits = pd.Series(0, index=df.index, dtype="int64")
        for token in tokens:
            column_hits += normalized_series.str.contains(rf"\b{re.escape(token)}\b", regex=True).astype("int64")

        hit_indexes = column_hits[column_hits > 0].index
        scores += column_hits
        for index in hit_indexes[:1000]:
            matched_columns_by_index.setdefault(int(index), set()).add(column)

    if scores.max() <= 0:
        return pd.DataFrame(), {}

    best_score = scores.max()
    best_indexes = scores[scores == best_score].index
    return df.loc[best_indexes], matched_columns_by_index


def answer_extreme_question(question: str, df: pd.DataFrame, target_column: str) -> str | None:
    q_tokens = text_tokens(question)
    wants_max = bool(q_tokens & set(MAX_WORDS))
    wants_min = bool(q_tokens & set(MIN_WORDS))
    if not wants_max and not wants_min:
        return None

    series = pd.to_numeric(df[target_column], errors="coerce")
    valid = series.dropna()
    if valid.empty:
        return None

    direction = "highest" if wants_max or not wants_min else "lowest"
    entity_column = choose_label_column(df, question, target_column=target_column)

    if entity_column and is_additive_metric(target_column) and not is_rate_metric(target_column, df[target_column]):
        grouped = (
            df[[entity_column, target_column]]
            .dropna()
            .groupby(entity_column, dropna=True)[target_column]
            .sum()
        )
        if not grouped.empty:
            best_value = grouped.max() if direction == "highest" else grouped.min()
            best_entities = grouped[grouped == best_value].sort_values(ascending=direction != "highest")
            names = [str(name) for name in best_entities.head(8).index]
            entity_name = entity_display_name(entity_column)
            metric_name = display_name(target_column)
            name_text = ", ".join(f"**{name}**" for name in names)
            tie_note = "" if len(best_entities) == 1 else f" ({len(best_entities):,} {pluralize(entity_name, len(best_entities))} are tied)"
            return (
                f"The {entity_name} with the {direction} total **{metric_name}** is {name_text}{tie_note}. "
                f"The value is **{format_metric_value(target_column, best_value, df[target_column])}**."
            )

    extreme_value = valid.max() if direction == "highest" else valid.min()
    matches = df.loc[series == extreme_value]
    metric_name = display_name(target_column)
    value_text = format_metric_value(target_column, extreme_value, df[target_column], include_raw_rate=True)
    source_text = f"Calculated from all **{len(valid):,} valid rows**."

    if entity_column:
        entities = matches[entity_column].dropna().astype(str).str.strip()
        entities = entities[entities != ""].drop_duplicates()
        examples = ", ".join(f"**{name}**" for name in entities.head(8))
        entity_name = entity_display_name(entity_column)
        if len(entities) > 1:
            return (
                f"{source_text} The {direction} **{metric_name}** is **{value_text}**. "
                f"**{len(entities):,} {pluralize(entity_name, len(entities))}** have this value. Examples: {examples}."
            )
        if len(entities) == 1:
            return f"{source_text} The {entity_name} with the {direction} **{metric_name}** is {examples}. The value is **{value_text}**."

    return f"{source_text} The {direction} **{metric_name}** is **{value_text}**."


def answer_row_lookup(question: str, df: pd.DataFrame, target_column: str) -> str | None:
    matches, matched_columns = find_matching_rows(df, question, target_column)
    if matches.empty:
        return None

    if len(matches) == 1:
        row = matches.iloc[0]
        index = int(matches.index[0])
        value = row[target_column]
        label = row_display_label(row, matched_columns.get(index, set()), target_column)
        return f"For {label}, **{display_name(target_column)}** is **{format_metric_value(target_column, value, df[target_column], include_raw_rate=True)}**."

    lines = [f"I found {len(matches):,} matching rows. Here are the first matches:"]
    for index, row in matches.head(8).iterrows():
        label = row_display_label(row, matched_columns.get(int(index), set()), target_column)
        lines.append(
            f"- {label}: **{display_name(target_column)}** = "
            f"**{format_metric_value(target_column, row[target_column], df[target_column], include_raw_rate=True)}**"
        )
    return "\n".join(lines)


def answer_multi_column_lookup(question: str, df: pd.DataFrame, target_columns: list[str]) -> str | None:
    if not target_columns:
        return None

    matches, matched_columns = find_matching_rows(df, question, target_columns[0])
    if matches.empty:
        return None

    if len(matches) == 1:
        row = matches.iloc[0]
        index = int(matches.index[0])
        label = row_display_label(row, matched_columns.get(index, set()), target_columns[0])
        values = [
            f"**{display_name(column)}** = **{format_metric_value(column, row[column], df[column], include_raw_rate=True)}**"
            for column in target_columns
        ]
        return f"For {label}, " + ", ".join(values) + "."

    lines = [f"I found {len(matches):,} matching rows. Here are the first matches:"]
    for index, row in matches.head(8).iterrows():
        label = row_display_label(row, matched_columns.get(int(index), set()), target_columns[0])
        values = ", ".join(
            f"{display_name(column)} = {format_metric_value(column, row[column], df[column], include_raw_rate=True)}"
            for column in target_columns
        )
        lines.append(f"- {label}: {values}")
    return "\n".join(lines)


def informative_row_columns(question: str, df: pd.DataFrame, label_column: str | None) -> list[str]:
    question_tokens = text_tokens(question)
    if {"project", "projects", "option", "options", "choice", "choices"} & question_tokens:
        columns = option_columns(df)
        if columns:
            return filter_option_columns_by_question(columns, question)

    preferred_hints = ("course", "batch", "group", "option", "score", "mark", "sales", "discount", "project")
    preferred = [
        column
        for column in df.columns
        if column != label_column and column_contains(column, preferred_hints)
    ]
    if preferred:
        return preferred[:8]

    fallback = [
        column
        for column in df.columns
        if column != label_column and not column_contains(column, ("email", "phone", "url", "link"))
    ]
    return fallback[:8]


def answer_row_summary(question: str, df: pd.DataFrame) -> str | None:
    matches, matched_columns = find_matching_rows(df, question, "")
    if matches.empty:
        return None

    if len(matches) == 1:
        row = matches.iloc[0]
        index = int(matches.index[0])
        label_column = choose_label_column(df, question)
        label = row_display_label(row, matched_columns.get(index, set()), label_column or "")
        columns = informative_row_columns(question, df, label_column)
        if not columns:
            return None
        values = [
            f"**{display_name(column)}** = **{format_metric_value(column, row[column], df[column], include_raw_rate=True)}**"
            for column in columns
        ]
        return f"For {label}, " + ", ".join(values) + "."

    lines = [f"I found {len(matches):,} matching rows. Here are the first matches:"]
    label_column = choose_label_column(df, question)
    columns = informative_row_columns(question, df, label_column)
    for index, row in matches.head(8).iterrows():
        label = row_display_label(row, matched_columns.get(int(index), set()), label_column or "")
        values = ", ".join(
            f"{display_name(column)} = {format_metric_value(column, row[column], df[column], include_raw_rate=True)}"
            for column in columns
        )
        lines.append(f"- {label}: {values}")
    return "\n".join(lines)


def local_question_answer(question: str, df: pd.DataFrame, profile: dict[str, Any]) -> str:
    q = question.strip().lower()
    if not q:
        return "Ask a question about the uploaded CSV."

    if "how many row" in q or "number of row" in q or "record" in q:
        return f"The CSV has {profile['rows']:,} rows."

    if "how many column" in q or "number of column" in q or "columns" == q:
        return f"The CSV has {profile['columns']:,} columns: {', '.join(profile['column_names'])}."

    if "missing" in q or "null" in q or "blank" in q:
        if not profile["missing_summary"]:
            return "There are no missing cells in the uploaded CSV."
        parts = [f"{item['column']}: {item['missing']:,} missing ({item['missing_pct']}%)" for item in profile["missing_summary"][:8]]
        return "Columns with missing values: " + "; ".join(parts) + "."

    if "duplicate" in q:
        return f"The CSV has {profile['duplicate_rows']:,} duplicate rows."

    target_columns = matching_target_columns(df, question)
    if len(target_columns) > 1:
        multi_answer = answer_multi_column_lookup(question, df, target_columns)
        if multi_answer:
            return multi_answer

    matched_column = target_columns[0] if target_columns else column_lookup(df, question)
    if matched_column:
        extreme_answer = answer_extreme_question(question, df, matched_column)
        if extreme_answer:
            return extreme_answer

        row_answer = answer_row_lookup(question, df, matched_column)
        if row_answer:
            return row_answer

        series = df[matched_column]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                return f"I found **{matched_column}**, but it does not have numeric values to calculate."
            if any(word in q for word in ("sum", "total")):
                return f"The total of **{display_name(matched_column)}** is **{format_metric_value(matched_column, numeric.sum(), series, include_raw_rate=True)}**."
            if any(word in q for word in ("average", "avg", "mean")):
                return f"The average of **{display_name(matched_column)}** is **{format_metric_value(matched_column, numeric.mean(), series, include_raw_rate=True)}**."
            if "median" in q:
                return f"The median of **{display_name(matched_column)}** is **{format_metric_value(matched_column, numeric.median(), series, include_raw_rate=True)}**."
            if any(word in q for word in ("minimum", "min", "lowest", "smallest")):
                return f"The minimum value in **{display_name(matched_column)}** is **{format_metric_value(matched_column, numeric.min(), series, include_raw_rate=True)}**."
            if any(word in q for word in ("maximum", "max", "highest", "largest")):
                return f"The maximum value in **{display_name(matched_column)}** is **{format_metric_value(matched_column, numeric.max(), series, include_raw_rate=True)}**."
            return (
                f"**{display_name(matched_column)}** averages **{format_metric_value(matched_column, numeric.mean(), series, include_raw_rate=True)}**, "
                f"has a middle value of **{format_metric_value(matched_column, numeric.median(), series, include_raw_rate=True)}**, and ranges from "
                f"**{format_metric_value(matched_column, numeric.min(), series, include_raw_rate=True)}** to "
                f"**{format_metric_value(matched_column, numeric.max(), series, include_raw_rate=True)}**."
            )

        values = series.dropna().astype(str).str.strip()
        values = values[values != ""]
        if values.empty:
            return f"I found **{matched_column}**, but it is empty."
        top_values = values.value_counts().head(5)
        parts = [f"{index}: {count:,}" for index, count in top_values.items()]
        return f"The most common values in **{matched_column}** are " + "; ".join(parts) + "."

    if any(word in q for word in NUMERIC_WORDS) and profile["numeric_stats"]:
        first = profile["numeric_stats"][0]
        return (
            f"I could not match a specific column name. For **{display_name(first['column'])}**, the average is "
            f"**{format_metric_value(first['column'], first['mean'], df[first['column']])}**, the middle value is "
            f"**{format_metric_value(first['column'], first['median'], df[first['column']])}**, and the total is "
            f"**{format_metric_value(first['column'], pd.to_numeric(df[first['column']], errors='coerce').sum(), df[first['column']])}**."
        )

    row_summary = answer_row_summary(question, df)
    if row_summary:
        return row_summary

    return (
        "I could not map that question to a specific column. Try including the exact column name, "
        "for example: `average sales`, `top city`, `missing values`, or `duplicate rows`."
    )


def generate_gemini_answer(question: str, local_answer: str, profile: dict[str, Any], df: pd.DataFrame, api_key: str) -> str:
    import google.generativeai as genai

    model = genai.GenerativeModel(choose_gemini_model(genai, api_key))
    prompt = f"""
Rewrite the CSV-calculated answer in clear plain English.
Do not change any names, numbers, columns, or values from the CSV-calculated answer.
If the CSV-calculated answer says it could not map the question, explain what exact column wording the user should try.

Question:
{question}

CSV-calculated answer:
{local_answer}

Profile JSON:
{json.dumps(profile, default=str)}

First rows as CSV:
{df.head(60).to_csv(index=False)}
"""
    response = model.generate_content(prompt)
    return (getattr(response, "text", "") or "").strip()


def render_metric_cards(profile: dict[str, Any]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{profile['rows']:,}")
    col2.metric("Fields", f"{profile['columns']:,}")
    col3.metric("Blank Cells", f"{profile['missing_cells']:,}")
    col4.metric("Duplicate Rows", f"{profile['duplicate_rows']:,}")


def render_charts(df: pd.DataFrame, profile: dict[str, Any]) -> None:
    numeric_columns = [
        column for column in profile["numeric_columns"] if not is_identifier_like(column, df[column])
    ]
    categorical_columns = useful_categorical_columns(df, profile["categorical_columns"])
    date_columns = profile["date_columns"]
    primary_metric = choose_primary_metric(df, numeric_columns)

    if not numeric_columns and not categorical_columns:
        st.info("No chart-ready numeric or category columns were detected.")
        return

    st.subheader("Recommended Charts")

    if primary_metric and date_columns:
        date_column = date_columns[0]
        trend = df[[date_column, primary_metric]].copy()
        trend[date_column] = pd.to_datetime(trend[date_column], errors="coerce")
        trend[primary_metric] = pd.to_numeric(trend[primary_metric], errors="coerce")
        trend = trend.dropna()

        if not trend.empty:
            trend = trend.sort_values(date_column)
            period = "M" if trend[date_column].dt.date.nunique() > 90 else "D"
            trend["Period"] = trend[date_column].dt.to_period(period).dt.to_timestamp()
            aggregate = "sum" if is_additive_metric(primary_metric) else "mean"
            grouped = trend.groupby("Period", as_index=False)[primary_metric].agg(aggregate)
            title_action = "Total" if aggregate == "sum" else "Average"
            fig = px.line(
                grouped,
                x="Period",
                y=primary_metric,
                markers=True,
                title=f"{title_action} {display_name(primary_metric)} Over Time",
            )
            fig.update_layout(xaxis_title=display_name(date_column), yaxis_title=display_name(primary_metric))
            st.plotly_chart(fig, width="stretch")

    chart_cols = st.columns(2)

    if primary_metric and categorical_columns:
        category = chart_cols[0].selectbox("Compare by", categorical_columns)
        aggregate = "sum" if is_additive_metric(primary_metric) else "mean"
        grouped = (
            df[[category, primary_metric]]
            .dropna()
            .groupby(category, as_index=False)[primary_metric]
            .agg(aggregate)
            .sort_values(primary_metric, ascending=False)
            .head(15)
        )
        if not grouped.empty:
            grouped = grouped.sort_values(primary_metric, ascending=True)
            title_action = "Total" if aggregate == "sum" else "Average"
            fig = px.bar(
                grouped,
                x=primary_metric,
                y=category,
                orientation="h",
                title=f"Top {display_name(category)} by {title_action} {display_name(primary_metric)}",
            )
            fig.update_layout(xaxis_title=display_name(primary_metric), yaxis_title=display_name(category))
            chart_cols[0].plotly_chart(fig, width="stretch")

    if numeric_columns:
        default_index = numeric_columns.index(primary_metric) if primary_metric in numeric_columns else 0
        numeric_column = chart_cols[1].selectbox("Distribution", numeric_columns, index=default_index)
        sample = df[[numeric_column]].dropna()
        if len(sample) > 10000:
            sample = sample.sample(10000, random_state=7)
        fig = px.histogram(
            sample,
            x=numeric_column,
            nbins=40,
            title=f"How {display_name(numeric_column)} Is Spread Across Records",
        )
        fig.update_layout(xaxis_title=display_name(numeric_column), yaxis_title="Number of records")
        chart_cols[1].plotly_chart(fig, width="stretch")

    if len(numeric_columns) >= 2:
        st.subheader("Relationship Check")
        rel_cols = st.columns(2)
        x_axis = rel_cols[0].selectbox("X axis", numeric_columns, index=0, key="relationship_x_axis")
        y_options = [column for column in numeric_columns if column != x_axis]
        default_y = y_options.index(primary_metric) if primary_metric in y_options else 0
        y_axis = rel_cols[1].selectbox("Y axis", y_options, index=default_y, key="relationship_y_axis")
        scatter = df.loc[:, [x_axis, y_axis]].dropna().copy()
        if len(scatter) > 5000:
            scatter = scatter.sample(5000, random_state=7)
        fig = px.scatter(
            scatter,
            x=x_axis,
            y=y_axis,
            opacity=0.45,
            title=f"{display_name(y_axis)} Compared With {display_name(x_axis)}",
        )
        fig.update_layout(xaxis_title=display_name(x_axis), yaxis_title=display_name(y_axis))
        st.plotly_chart(fig, width="stretch")


def main() -> None:
    inject_professional_styles()
    if st.session_state.get("app_version") != APP_VERSION:
        st.session_state["app_version"] = APP_VERSION
        st.session_state.pop("last_question", None)
        st.session_state.pop("last_answer", None)

    st.markdown(
        """
        <div class="analyst-header">
            <div class="analyst-kicker">CSV Intelligence Workspace</div>
            <h1>AI Data Analyst</h1>
            <div class="analyst-subtitle">
                Upload a CSV, get business-readable insights, realistic charts, and direct answers calculated from the full dataset.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Dataset")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv", "tsv", "txt"])
        st.markdown("### Answer Mode")
        st.caption("CSV answers are calculated locally from all rows. Gemini is optional wording support only.")
        use_gemini = st.toggle("Show Gemini options", value=False)

    if uploaded_file is None:
        st.info("Upload a CSV file to generate a plain-English summary, charts, and direct answers.")
        return

    try:
        with st.spinner("Reading CSV..."):
            result = load_csv(uploaded_file.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        return

    df = result.dataframe
    with st.spinner("Preparing insights..."):
        profile = profile_csv(df)
    api_key = gemini_api_key()
    file_signature = (APP_VERSION, uploaded_file.name, uploaded_file.size)
    if st.session_state.get("file_signature") != file_signature:
        st.session_state["file_signature"] = file_signature
        st.session_state.pop("last_question", None)
        st.session_state.pop("last_answer", None)

    st.markdown(
        f"""
        <div class="source-note">
            Loaded <strong>{uploaded_file.name}</strong> with {result.encoding} encoding and {delimiter_label(result.delimiter)} delimiter.
            Answers in the Ask tab are calculated from the CSV, not guessed from an AI sample.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_metric_cards(profile)

    summary_tab, data_tab, charts_tab, ask_tab = st.tabs(["Summary", "Data", "Charts", "Ask"])

    with summary_tab:
        st.markdown(build_plain_english_summary(profile, df))
        if use_gemini:
            if not api_key:
                st.info("No Gemini API key was found. The summary above is generated directly from the CSV.")
            elif st.button("Generate optional Gemini wording"):
                with st.spinner("Generating Gemini summary..."):
                    try:
                        ai_summary = generate_gemini_summary(profile, df, api_key)
                    except Exception as exc:  # noqa: BLE001 - keep the app usable if the model/API fails.
                        st.warning(f"Gemini summary failed, but the local summary above is still available. Error: {exc}")
                    else:
                        if ai_summary:
                            st.markdown("### Gemini Summary")
                            st.markdown(ai_summary)

    with data_tab:
        if len(df) > 1000:
            max_preview = min(len(df), 5000)
            preview_rows = st.slider("Rows to preview", min_value=100, max_value=max_preview, value=1000, step=100)
        else:
            preview_rows = len(df)
        st.caption(f"Showing the first {preview_rows:,} rows. The download includes all {len(df):,} rows.")
        st.dataframe(df.head(preview_rows), width="stretch", hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download cleaned CSV", data=csv, file_name="cleaned_data.csv", mime="text/csv")

    with charts_tab:
        render_charts(df, profile)

    with ask_tab:
        with st.form("ask_csv_form"):
            question = st.text_input("Ask a question about the CSV")
            submitted = st.form_submit_button("Answer from CSV")

        if submitted and question:
            st.session_state["last_question"] = question
            st.session_state["last_answer"] = local_question_answer(question, df, profile)

        if st.session_state.get("last_answer"):
            st.markdown(st.session_state["last_answer"])
            if use_gemini and api_key and st.button("Ask Gemini for extra wording"):
                with st.spinner("Checking with Gemini..."):
                    try:
                        answer = generate_gemini_answer(
                            st.session_state["last_question"],
                            st.session_state["last_answer"],
                            profile,
                            df,
                            api_key,
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Gemini answer failed. Error: {exc}")
                    else:
                        if answer:
                            st.markdown("### Gemini Answer")
                            st.markdown(answer)
            elif use_gemini and not api_key:
                st.info("No Gemini API key was found. The answer above is calculated directly from the CSV.")


if __name__ == "__main__":
    main()
