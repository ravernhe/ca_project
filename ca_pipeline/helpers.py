# -*- coding: utf-8 -*-

from __future__ import annotations
import re, unicodedata, csv
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from .settings import DATE_COLUMNS

# ---------- normalization ----------
def norm(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_col_index(columns) -> dict:
    return {norm(c): c for c in columns}

def find_col(idx: dict, columns, *, any_of=(), all_tokens=(), regex: Optional[str]=None):
    for name in any_of:
        key = norm(name)
        if key in idx:
            return idx[key]
    if all_tokens:
        toks = [norm(t) for t in all_tokens]
        for c in columns:
            nc = norm(c)
            if all(t in nc for t in toks):
                return c
    if regex:
        cre = re.compile(regex)
        for c in columns:
            if cre.search(norm(c)):
                return c
    return None

# ---------- IO ----------
def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        except Exception:
            try:
                return pd.read_csv(path, sep=";", engine="python", encoding="utf-8-sig")
            except Exception:
                return pd.read_csv(path, sep=",", engine="python", encoding="utf-8-sig")
    raise ValueError(f"Unsupported file type: {path.suffix}")

def read_csv_loose(path: Path) -> pd.DataFrame:
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t,")
        delim = dialect.delimiter
    except Exception:
        delim = None

    try:
        return pd.read_csv(path, header=None, sep=None, engine="python",
                           quoting=csv.QUOTE_MINIMAL, quotechar='"', escapechar="\\",
                           on_bad_lines="skip", encoding="utf-8-sig")
    except Exception:
        pass

    if delim:
        try:
            return pd.read_csv(path, header=None, sep=delim, engine="python",
                               quoting=csv.QUOTE_MINIMAL, quotechar='"', escapechar="\\",
                               on_bad_lines="skip", encoding="utf-8-sig")
        except Exception:
            pass

    for d in [";", ",", "\t", "|"]:
        try:
            return pd.read_csv(path, header=None, sep=d, engine="python",
                               quoting=csv.QUOTE_MINIMAL, quotechar='"', escapechar="\\",
                               on_bad_lines="skip", encoding="utf-8-sig")
        except Exception:
            continue

    try:
        return pd.read_table(path, header=None, sep=r"[;,\t|]", engine="python",
                             on_bad_lines="skip", encoding="utf-8-sig")
    except Exception as e:
        raise ValueError(f"CSV illisible: {path.name} ({e})")

def export_excel(df: pd.DataFrame, out_path: Path, meta: dict):
    df_out = df.replace({pd.NA: "", "nan": "", "<NA>": "", None: ""})
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df_out.to_excel(xw, sheet_name="consolidation", index=False)
        resume = df_out.groupby("Origine rapport", dropna=False).size().rename("nb_lignes").reset_index()
        resume.to_excel(xw, sheet_name="resume", index=False)
        pd.DataFrame([meta]).to_excel(xw, sheet_name="parametres", index=False)

# ---------- transformations ----------
def format_date_columns(df: pd.DataFrame, style: str = "mdy_slash") -> pd.DataFrame:
    def _fmt_mdy_slash(dt):
        if pd.isna(dt): return ""
        return f"{dt.month}/{dt.day}/{str(dt.year)[-2:]}"
    for col in DATE_COLUMNS:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce", format="%m/%d/%y")
            df[col] = s.apply(_fmt_mdy_slash) if style == "mdy_slash" else s.astype(str)
    return df

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df
