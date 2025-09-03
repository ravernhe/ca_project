# -*- coding: utf-8 -*-

from __future__ import annotations
import re, unicodedata, csv
from pathlib import Path
from typing import List, Dict, Optional
from openpyxl import load_workbook
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

from openpyxl import load_workbook

from openpyxl import load_workbook
from datetime import datetime
import pandas as pd
from pathlib import Path

def export_excel(df: pd.DataFrame, out_path: Path, meta: dict):
    ws_name = "consolidation"

    # --- Colonnes à cibler ---
    date_cols = [
        "Date d'éxécution (produit sans session)",
        "Date de début",
        "Date de fin",
        "Date prévisionnelle de début de projet",
        "Date de fin de projet",
    ]
    price_cols = [
        "Prix Intra 1 standard (converti)",
        "Prix total",
        "CA session",
        "Facturation (convertie) N-2",
        "Facturation (convertie) N-1",
        "Facturation (convertie) N",
        "Facturation Y-2",
        "Facturation Y-1",
        "Facturation Y",
        "Facturation totale",
        "CA attendu",
        "CA avancement",
        "CA YTD",
        "CA EOY (backlog)",
        "FAE",
        "PCA",
        "Prix de vente (converti)",
        "Prix total (converti)",
    ]
    percent_cols = ["Taux d'avancement global", "Avancement EOY"]

    # --- Préparation du DataFrame ---
    df_out = df.copy()

    # convertir les colonnes de dates en datetime64
    for col in date_cols:
        if col in df_out.columns:
            df_out[col] = pd.to_datetime(df_out[col], errors="coerce")

    # remplacer NaN uniquement sur les colonnes non-dates
    non_date_cols = [c for c in df_out.columns if c not in date_cols]
    df_out[non_date_cols] = df_out[non_date_cols].replace(
        {pd.NA: "", "nan": "", "<NA>": "", None: ""}
    )

    # --- Écriture des feuilles ---
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df_out.to_excel(xw, sheet_name=ws_name, index=False)
        resume = (
            df_out.groupby("Origine rapport", dropna=False)
            .size()
            .rename("nb_lignes")
            .reset_index()
        )
        resume.to_excel(xw, sheet_name="resume", index=False)
        pd.DataFrame([meta]).to_excel(xw, sheet_name="parametres", index=False)

    # --- Mise en forme Excel ---
    wb = load_workbook(out_path)

    # auto-largeur en fonction de l'en-tête
    for ws in wb.worksheets:
        for cell in ws[1]:
            if cell.value is not None:
                col_letter = cell.column_letter
                ws.column_dimensions[col_letter].width = len(str(cell.value)) + 2

    ws = wb[ws_name]
    headers = {cell.value: cell.column for cell in ws[1] if cell.value}

    # Dates → DD/MM/YY
    for name in date_cols:
        if name in headers:
            ci = headers[name]
            for col in ws.iter_cols(
                min_col=ci, max_col=ci, min_row=2, values_only=False
            ):
                for cell in col:
                    if isinstance(cell.value, (datetime, pd.Timestamp)):
                        cell.number_format = "dd/mm/yy"

    # Prix → #
    for name in price_cols:
        if name in headers:
            ci = headers[name]
            for col in ws.iter_cols(
                min_col=ci, max_col=ci, min_row=2, values_only=False
            ):
                for cell in col:
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = "#"

    # Pourcentages → 0%
    for name in percent_cols:
        if name in headers:
            ci = headers[name]
            for col in ws.iter_cols(
                min_col=ci, max_col=ci, min_row=2, values_only=False
            ):
                for cell in col:
                    cell.number_format = "0%"

    wb.save(out_path)



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