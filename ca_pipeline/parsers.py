# -*- coding: utf-8 -*-

from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from .settings import DATE_OUTPUT_STYLE, COLUMN_MAPPING
from .helpers import (
    read_any, build_col_index, find_col,
    normalize_headers, format_date_columns
)

# ----------------------- Utils locaux (href, normalisation) ------------------

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

# ------------------ Routage code analytique & axes (métier) ------------------

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

# ----------------------------- Facturation --------------------

def _fr_to_float(x) -> float:
    """Convertit un montant format FR en float (tolérant aux espaces fines, NBSP, etc.)."""
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

def _read_fact_excel_b6(path: Path) -> pd.DataFrame:
    """
    Lit un Excel dont l'entête est en B6 et les données à partir de B7.
    Retourne un DataFrame avec la première ligne d'entête prise à B6 (iloc[5,1:]).
    """
    raw = pd.read_excel(path, header=None, dtype=object)
    if raw.shape[0] < 7 or raw.shape[1] < 2:
        raise ValueError(f"Fichier facturation Excel trop court/inattendu: {path.name}")
    header = raw.iloc[6, 1:].tolist()   # B6..fin
    data   = raw.iloc[7:, 1:].copy()    # B7..fin
    data.columns = header
    return data

def _read_fact_csv_b6(path: Path) -> pd.DataFrame:
    """
    Lit un CSV de facturation dont l'entête est en B6 (ligne 6, colonne B).
    -> skiprows=5 pour ignorer les 5 premières lignes
    -> on saute aussi la 1ère colonne (col A) car l'entête commence en B
    """
    # try auto-sep detection first
    try:
        raw = pd.read_csv(
            path, header=None, sep=None, engine="python",
            encoding="utf-8-sig", on_bad_lines="skip", skiprows=5
        )
    except Exception:
        raw = None
        for sep in [";", ",", "\t", "|"]:
            try:
                raw = pd.read_csv(
                    path, header=None, sep=sep, engine="python",
                    encoding="utf-8-sig", on_bad_lines="skip", skiprows=5
                )
                break
            except Exception:
                continue
        if raw is None:
            raise

    # La ligne immédiatement lue après skiprows=5 correspond à la ligne 6 du CSV → l'entête
    header = raw.iloc[0, 1:].tolist()     # ligne 6 (Excel) / ligne 1 après skip, colonnes B..fin
    data   = raw.iloc[1:, 1:].copy()      # données à partir de la ligne 7, colonnes B..fin
    data.columns = header
    return data


def parse_fact_file(path: Path, origin_label: str) -> pd.DataFrame:
    """
    Parse un fichier facturation (EUR/HKD) .xlsx/.xls (entête B6) ou .csv (entête première ligne).
    Extrait:
      - Section Analytique - Code  -> Code analytique
      - Date Ecriture              -> Date (m/d/yy)
      - Solde Tenue de Compte      -> Montant (float)
    Calcule aussi Année (depuis Date).
    Retourne un DataFrame avec colonnes: Origine rapport | Code analytique | Date | Année | Montant | [Filiale First Finance]
    """
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls"):
        data = _read_fact_excel_b6(path)
    elif suf == ".csv":
        data = _read_fact_csv_b6(path)
    else:
        raise ValueError(f"Type non supporté pour facturation: {path.suffix}")

    needed = ["Section Analytique - Code", "Date Ecriture", "Solde Tenue de Compte"]
    for col in needed:
        if col not in data.columns:
            raise ValueError(f"Colonne manquante: {col}. Vu: {list(map(str, data.columns))}")

    # On ne garde que ce qui est nécessaire
    df = data[needed + ([c for c in ["Société", "Societe"] if c in data.columns])].copy()

    # Normalisations
    df["Section Analytique - Code"] = df["Section Analytique - Code"].astype(str).str.strip()

    # Dates: input peut être '3/25/24' -> on suppose m/d/y (mdy) ; erreurs -> NaT
    dates = pd.to_datetime(df["Date Ecriture"], errors="coerce", format="%m/%d/%y")
    df["Année"] = dates.dt.year.astype("Int64")
    df["Date"]  = dates.apply(lambda d: f"{d.month}/{d.day}/{str(d.year)[-2:]}" if pd.notna(d) else "")

    # Montants FR
    df["Montant"] = pd.to_numeric(df["Solde Tenue de Compte"].map(_fr_to_float), errors="coerce").fillna(0.0)

    # Filtrage de base
    df = df[df["Section Analytique - Code"].astype(str).str.len() > 0]

    # Sortie standardisée
    out = pd.DataFrame({
        "Origine rapport": origin_label,
        "Code analytique": df["Section Analytique - Code"].astype(str).str.strip(),
        "Date": df["Date"],
        "Année": df["Année"],
        "Montant": df["Montant"],
    })

    # Filiale si présent
    if "Société" in df.columns:
        out["Filiale First Finance"] = df["Société"]
    elif "Societe" in df.columns:
        out["Filiale First Finance"] = df["Societe"]

    # On supprime les lignes sans code ou année
    out = out[out["Code analytique"].astype(str).str.len() > 0]
    # Année peut être NA si Date vide: on garde quand même (sera ignoré lors des agrégations si besoin)
    return out

# ------------------- Parse générique (intra / inter / ss) --------------------

def parse_file_with_origin(path: Path, origin_label: str) -> pd.DataFrame:
    if origin_label in ("fact_eur", "fact_hkd"):
        df_fact = parse_fact_file(path, origin_label)
        df_fact["Source fichier"] = path.name
        return df_fact

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
