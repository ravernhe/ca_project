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
    # supprime espaces fines/insécables et remplace la virgule par un point
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
    """Affecte les colonnes (pdt)/(cf session) selon l'origine, et crée 'Code analytique' = cf;pdt."""
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
# Facturation : lecture Excel (B6) et CSV (B6) + extraction des années
# -------------------------------------------------------------------

_YEAR_RANGE_RE = re.compile(r"(20\d{2}).{0,3}(20\d{2})")

def _extract_years_hint_from_excel_raw(raw: pd.DataFrame) -> tuple[int|None,int|None]:
    """Cherche une ligne avec 'Année' et 2 années (ex: 'Année', '2023..2024')."""
    max_rows = min(12, len(raw))
    max_cols = min(8, raw.shape[1])
    for i in range(max_rows):
        row = raw.iloc[i, :max_cols].tolist()
        for j, v in enumerate(row):
            s = "" if pd.isna(v) else str(v)
            if s.strip().lower().startswith("année"):
                # à droite si dispo
                if j + 1 < max_cols:
                    s2 = "" if pd.isna(row[j+1]) else str(row[j+1])
                    m = _YEAR_RANGE_RE.search(s2)
                    if m:
                        y1, y2 = int(m.group(1)), int(m.group(2))
                        if y1 > y2: y1, y2 = y2, y1
                        return (y1, y2)
                # sinon dans la même cellule
                m = _YEAR_RANGE_RE.search(s)
                if m:
                    y1, y2 = int(m.group(1)), int(m.group(2))
                    if y1 > y2: y1, y2 = y2, y1
                    return (y1, y2)
    # fallback C3
    try:
        cell = raw.iloc[2, 2]
        s = "" if pd.isna(cell) else str(cell)
        m = _YEAR_RANGE_RE.search(s)
        if m:
            y1, y2 = int(m.group(1)), int(m.group(2))
            if y1 > y2: y1, y2 = y2, y1
            return (y1, y2)
    except Exception:
        pass
    return (None, None)

def _read_fact_excel_b6(path: Path) -> tuple[pd.DataFrame, tuple[int|None,int|None]]:
    """Excel entête en B6 (ligne 7 Excel, 0-based index 6), données à partir de B7."""
    raw = pd.read_excel(path, header=None, dtype=object)
    years_hint = _extract_years_hint_from_excel_raw(raw)
    header = raw.iloc[6, 1:].tolist()   # B6
    data   = raw.iloc[7:, 1:].copy()    # B7+
    data.columns = header
    return data, years_hint

def _read_fact_csv_b6(path: Path) -> tuple[pd.DataFrame, tuple[int|None,int|None]]:
    """CSV avec entête positionnée en B6 : skip 5 lignes puis prendre colonnes B..fin."""
    # on tente la détection de séparateur
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
    # extraire hint directement depuis le head non-skippé
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

    header = raw.iloc[0, 1:].tolist()   # B6
    data   = raw.iloc[1:, 1:].copy()    # B7+
    data.columns = header
    return data, years_hint

def parse_fact_file(path: Path, origin_label: str) -> pd.DataFrame:
    """
    Parse un fichier de facturation (EUR/HKD) : Excel (.xlsx/.xls) ou CSV (B6).
    Extrait: 'Section Analytique - Code' => 'Code analytique', 'Date Ecriture' => 'Date', 'Montant'.
    Ajoute: 'Année' (depuis Date), 'Filiale First Finance' si 'Société' présent,
            et les hints '__Y_HINT_N1' / '__Y_HINT_N' si trouvés.
    """
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

    # Normalisations
    df["Section Analytique - Code"] = df["Section Analytique - Code"].astype(str).str.strip()
    dates = pd.to_datetime(df["Date Ecriture"], errors="coerce", dayfirst=False, yearfirst=False)
    df["Année"] = dates.dt.year.astype("Int64")
    df["Date"]  = dates.apply(lambda d: _fmt_mdy(d) if pd.notna(d) else "")

    df["Montant"] = pd.to_numeric(df["Solde Tenue de Compte"].map(_fr_to_float), errors="coerce").fillna(0.0)

    out = pd.DataFrame({
        "Origine rapport": origin_label,  # 'fact_eur' ou 'fact_hkd'
        "Code analytique": df["Section Analytique - Code"].astype(str).str.strip(),
        "Date": df["Date"],
        "Année": df["Année"],
        "Montant": df["Montant"],
    })

    if "Société" in df.columns:
        out["Filiale First Finance"] = df["Société"]
    elif "Societe" in df.columns:
        out["Filiale First Finance"] = df["Societe"]

    # Inject hints d'années si trouvés
    y1, y2 = years_hint  # y1 = N-1, y2 = N
    if y1 is not None and y2 is not None:
        out["__Y_HINT_N1"] = y1
        out["__Y_HINT_N"]  = y2

    out["Source fichier"] = path.name
    return out

# -------------------------------------------------------------------
# Parse générique (INTRA / INTER / SANS_SESSION)
# -------------------------------------------------------------------

def parse_file_with_origin(path: Path, origin_label: str) -> pd.DataFrame:
    if origin_label in ("fact_eur", "fact_hkd"):
        return parse_fact_file(path, origin_label)

    df = read_any(path)
    df = normalize_headers(df)
    df = apply_column_mapping(df)

    df["Origine rapport"] = origin_label
    df = format_date_columns(df, DATE_OUTPUT_STYLE)
    df = extract_anchor_href(df)
    df = fix_code_analytique_fields(df, origin_label)
    df = route_axe(df, origin_label)
    df["Source fichier"] = path.name
    return df
