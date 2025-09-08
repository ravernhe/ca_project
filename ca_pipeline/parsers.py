# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

from .settings import DATE_OUTPUT_STYLE, COLUMN_MAPPING
from .helpers import (
    read_any, normalize_headers, format_date_columns
)

# -------------------------------------------------------------------
# Utilitaires
# -------------------------------------------------------------------

_href_re = re.compile(r'href="([^"]+)"')
def extract_anchor_href(df: pd.DataFrame) -> pd.DataFrame:
    col = "Lien Vers le Produit d'opportunité"
    if col in df.columns:
        df[col] = df[col].astype(str).apply(
            lambda v: _href_re.search(v).group(1) if ("<a " in v and _href_re.search(v)) else v
        )
    return df

def apply_column_mapping(df: pd.DataFrame) -> pd.DataFrame:
    lower_to_actual = {c.lower(): c for c in df.columns}
    ren = {}
    for k_lower, target in COLUMN_MAPPING.items():
        if k_lower in lower_to_actual:
            ren[lower_to_actual[k_lower]] = target
    return df.rename(columns=ren)

def _fr_to_float(x) -> float:
    if pd.isna(x):
        return 0.0
    s = str(x)
    s = s.replace("\u202f", "").replace("\xa0", "").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        s2 = re.sub(r"[^0-9\.\-]+", "", s)
        try:
            return float(s2) if s2 not in ("", "-", ".", "-.") else 0.0
        except Exception:
            return 0.0

def _fmt_mdy(dt: pd.Timestamp) -> str:
    return f"{dt.month}/{dt.day}/{str(dt.year)[-2:]}" if pd.notna(dt) else ""

# -------------------------------------------------------------------
# Codes analytiques & Axes
# -------------------------------------------------------------------

def fix_code_analytique_fields(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    for col in ["Code analytique (pdt)", "Code analytique (cf session)"]:
        if col not in df.columns:
            df[col] = ""

    o = (origin or "").lower().strip()
    cols = [c for c in df.columns if str(c).lower().startswith("code analytique")]

    if o == "intra":
        src_pdt = "Code analytique" if "Code analytique" in df.columns else (cols[0] if cols else None)
        src_cf  = "Code analytique.1" if "Code analytique.1" in df.columns else (cols[1] if len(cols) > 1 else None)
        if src_pdt:
            df["Code analytique (pdt)"] = df["Code analytique (pdt)"].where(df["Code analytique (pdt)"].astype(str).ne(""), df[src_pdt])
        if src_cf:
            df["Code analytique (cf session)"] = df["Code analytique (cf session)"].where(df["Code analytique (cf session)"].astype(str).ne(""), df[src_cf])

    elif o == "inter":
        src = "Code analytique" if "Code analytique" in df.columns else (cols[0] if cols else None)
        if src:
            df["Code analytique (cf session)"] = df["Code analytique (cf session)"].where(df["Code analytique (cf session)"].astype(str).ne(""), df[src])

    elif o == "sans_session":
        src = "Code analytique" if "Code analytique" in df.columns else (cols[0] if cols else None)
        if src:
            df["Code analytique (pdt)"] = df["Code analytique (pdt)"].where(df["Code analytique (pdt)"].astype(str).ne(""), df[src])

    for tgt in ["Code analytique (pdt)", "Code analytique (cf session)"]:
        df[tgt] = df[tgt].astype(str).replace({"nan": "", "None": "", "<NA>": ""}).str.strip()

    cf  = df["Code analytique (cf session)"].fillna("").astype(str)
    pdt = df["Code analytique (pdt)"].fillna("").astype(str)
    df["Code analytique"] = (cf + ";" + pdt).str.replace(";;", ";", regex=False).str.strip(";").str.strip()
    return df

def route_axe(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    if "Axe" not in df.columns:
        return df
    if "Axe (sessions)" not in df.columns:
        df["Axe (sessions)"] = pd.NA
    if "Axe (produit d'opportunité)" not in df.columns:
        df["Axe (produit d'opportunité)"] = pd.NA
    if origin.lower() in ("intra", "inter"):
        df["Axe (sessions)"] = df["Axe (sessions)"].fillna(df["Axe"])
    else:
        df["Axe (produit d'opportunité)"] = df["Axe (produit d'opportunité)"].fillna(df["Axe"])
    return df

# -------------------------------------------------------------------
# Facturation
# -------------------------------------------------------------------

_YEAR_RANGE_RE = re.compile(r"(20\d{2}).{0,3}(20\d{2})")

def _extract_years_hint_from_excel_raw(raw: pd.DataFrame) -> list[int]:
    """Cherche une cellule avec un intervalle d'années (ex: '2023..2025')
       et retourne [2023, 2024, 2025]."""
    max_rows = min(12, len(raw))
    max_cols = min(8, raw.shape[1])
    for i in range(max_rows):
        row = raw.iloc[i, :max_cols].tolist()
        for v in row:
            s = "" if pd.isna(v) else str(v)
            m = _YEAR_RANGE_RE.search(s)
            if m:
                y1, y2 = int(m.group(1)), int(m.group(2))
                if y1 > y2:
                    y1, y2 = y2, y1
                return list(range(y1, y2 + 1))
    return []

def _read_fact_excel_b6(path: Path) -> tuple[pd.DataFrame, tuple[int|None,int|None]]:
    raw = pd.read_excel(path, header=None, dtype=object)
    years_hint = _extract_years_hint_from_excel_raw(raw)
    header = raw.iloc[5, 1:].tolist()
    data   = raw.iloc[6:, 1:].copy()
    data.columns = header
    return data, years_hint

def _read_fact_csv_b6(path: Path) -> tuple[pd.DataFrame, tuple[int|None,int|None]]:
    try:
        raw = pd.read_csv(path, header=None, sep=None, engine="python",
                          encoding="utf-8-sig", on_bad_lines="skip", skiprows=5)
    except Exception:
        raw = None
        for sep in [";", ",", "\t", "|"]:
            try:
                raw = pd.read_csv(path, header=None, sep=sep, engine="python",
                                  encoding="utf-8-sig", on_bad_lines="skip", skiprows=5)
                break
            except Exception:
                continue
        if raw is None:
            raise
    try:
        head = pd.read_csv(path, header=None, nrows=12, encoding="utf-8-sig", sep=None, engine="python")
    except Exception:
        head = None
        for sep in [";", ",", "\t", "|"]:
            try:
                head = pd.read_csv(path, header=None, nrows=12, encoding="utf-8-sig", sep=sep, engine="python")
                break
            except Exception:
                continue
    years_hint = _extract_years_hint_from_excel_raw(head) if head is not None else (None, None)

    header = raw.iloc[0, 1:].tolist()
    data   = raw.iloc[1:, 1:].copy()
    data.columns = header
    return data, years_hint

def parse_fact_file(path: Path, origin_label: str) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls"):
        data, years_hint = _read_fact_excel_b6(path)
    elif suf == ".csv":
        data, years_hint = _read_fact_csv_b6(path)
    else:
        raise ValueError(f"Type non supporté pour facturation: {path.suffix}")

    needed = ["Section Analytique - Code", "Date Ecriture", "Solde Tenue de Compte"]
    for col in needed:
        if col not in data.columns:
            raise ValueError(f"Colonne manquante: {col}. Vu: {list(map(str, data.columns))}")

    df = data[needed + ([c for c in ["Société", "Societe"] if c in data.columns])].copy()
    df["Section Analytique - Code"] = df["Section Analytique - Code"].astype(str).str.strip()
    dates = pd.to_datetime(df["Date Ecriture"], errors="coerce", dayfirst=False, yearfirst=False)
    df["Année"] = dates.dt.year.astype("Int64")
    df["Date"]  = dates.apply(lambda d: _fmt_mdy(d) if pd.notna(d) else "")
    df["Montant"] = pd.to_numeric(df["Solde Tenue de Compte"].map(_fr_to_float), errors="coerce").fillna(0.0)
    df["DateDT"] = dates

    out = pd.DataFrame({
        "Origine rapport": origin_label,
        "Code analytique": df["Section Analytique - Code"].astype(str).str.strip(),
        "Date": df["Date"],
        "Année": df["Année"],
        "Montant": df["Montant"],
        "DateDT": df["DateDT"],
    })

    if "Société" in df.columns:
        out["Filiale First Finance"] = df["Société"]
    elif "Societe" in df.columns:
        out["Filiale First Finance"] = df["Societe"]

    years = years_hint if isinstance(years_hint, (list, tuple)) else []
    years = [int(y) for y in years if y is not None]

    if years:
        years_sorted = sorted(set(years))
        # keep compatibility with existing pipeline columns:
        out["__Y_HINT_N"] = years_sorted[-1]              # most recent year
        if len(years_sorted) >= 2:
            out["__Y_HINT_N1"] = years_sorted[-2]         # previous year

        # optional: store the full range as a csv string for later logic (non-breaking)
        out["__Y_HINT_YEARS"] = ",".join(map(str, years_sorted))

    out["Source fichier"] = path.name
    return out

# -------------------------------------------------------------------
# Prix (simplified: always EUR)
# -------------------------------------------------------------------

PRICE_TO_DEV_COL = {
    "Prix Intra 1 standard (converti)": "Prix Intra 1 standard (converti) Devise",
    "Prix total (converti)": "Prix total (converti) Devise",
    "Prix de vente (converti)": "Prix de vente (converti) Devise",
}

PRICE_COL_ALIASES = {
    "Prix Intra 1 standard (converti)": [
        "Prix intra 1 standard (converti)", "Prix Intra 1 Standard (converti)",
        "Prix Intra 1 standard(converti)", "Prix intra 1 standard converti",
    ],
    "Prix total (converti)": [
        "Prix total (converti)", "Prix Total (converti)", "Prix total(converti)",
        "Prix total",
    ],
    "Prix de vente (converti)": [
        "Prix de vente (converti)", "Prix De Vente (converti)", "Prix de vente(converti)",
    ],
}

def _normalize_header_name(name: str) -> str:
    return str(name).replace("\u00A0", " ").replace("\u202F", " ").strip().replace("  ", " ")

def _resolve_present_price_cols(df: pd.DataFrame) -> dict[str, str]:
    present = { _normalize_header_name(c): c for c in df.columns }
    resolved = {}
    for canonical, variants in PRICE_COL_ALIASES.items():
        if canonical in present:
            resolved[canonical] = present[canonical]
            continue
        for v in variants:
            v_norm = _normalize_header_name(v)
            if v_norm in present:
                resolved[canonical] = present[v_norm]
                break
    return resolved

def normalize_price_columns(df: pd.DataFrame,
                            price_cols: list[str] | None = None,
                            create_missing_dev_col: bool = True,
                            debug: bool = False) -> pd.DataFrame:
    """Simplified currency handling:
    - only numeric coercion on price columns
    - fill associated devise columns with 'EUR' by default.
    """
    df = df.copy()
    resolved = _resolve_present_price_cols(df)
    if price_cols:
        resolved = {k: v for k, v in resolved.items() if k in price_cols}
    if debug:
        print("[price] resolved:", resolved)
    for canonical, actual_col in resolved.items():
        df[actual_col] = pd.to_numeric(df[actual_col], errors='coerce')
        dev_col = PRICE_TO_DEV_COL.get(canonical)
        if dev_col:
            df[dev_col] = 'EUR'
        elif create_missing_dev_col:
            det_col = f"{canonical} Devise détectée"
            df[det_col] = 'EUR'
    return df

def parse_file_with_origin(path: Path, origin_label: str) -> pd.DataFrame:
    if origin_label in ("fact_eur", "fact_hkd"):
        return parse_fact_file(path, origin_label)

    df = read_any(path)
    df = normalize_headers(df)
    df = apply_column_mapping(df)
    df = normalize_price_columns(df)
    df = format_date_columns(df, style="datetime_fr")
    df = normalize_price_columns(df)

    df["Origine rapport"] = origin_label
    df = extract_anchor_href(df)
    df = fix_code_analytique_fields(df, origin_label)
    df = route_axe(df, origin_label)
    df["Source fichier"] = path.name
    return df