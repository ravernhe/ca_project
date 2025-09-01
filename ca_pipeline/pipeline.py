# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
from typing import List, Optional
from .settings import TARGET_COLUMNS, INTERMEDIATE_MONETARY_COLS
from .helpers import export_excel

# Concat Intra / SS / Inter
def pipeline_concat(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dfs, ignore_index=True, sort=False) if dfs else pd.DataFrame()

def _map_series_on_two_codes(df: pd.DataFrame, ser: pd.Series, out_col: str) -> pd.DataFrame:
    """
    Left-join a Series (index = Code analytique, values = totals) onto both
    'Code analytique (cf session)' and 'Code analytique (pdt)' then sum.
    """
    if ser is None or ser.empty:
        df[out_col] = 0.0
        return df
    df = df.copy()
    for k in ["Code analytique (cf session)", "Code analytique (pdt)"]:
        if k in df.columns:
            df[k] = df[k].astype(str).str.strip().replace({"nan": "", "None": "", "<NA>": ""})
    ser = ser.copy()
    ser.index = ser.index.astype(str).str.strip()
    df = df.merge(ser.rename(f"__{out_col}_cf"), how="left",
                  left_on="Code analytique (cf session)", right_index=True)
    df = df.merge(ser.rename(f"__{out_col}_pdt"), how="left",
                  left_on="Code analytique (pdt)", right_index=True)
    df[out_col] = pd.to_numeric(df.get(f"__{out_col}_cf"), errors="coerce").fillna(0.0) + \
                  pd.to_numeric(df.get(f"__{out_col}_pdt"), errors="coerce").fillna(0.0)
    for c in [f"__{out_col}_cf", f"__{out_col}_pdt"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    return df

def _parse_rate(val) -> Optional[float]:
    """Accepts '0.11', '0,11', '11%' etc. Returns float or None."""
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9eE\+\-\.%]", "", s)
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None

def compute_facturation_from_external(df_main: pd.DataFrame,
                                      fact_eur: pd.DataFrame,
                                      fact_hkd: pd.DataFrame,
                                      hkd_to_eur_rate) -> pd.DataFrame:
    """
    - fact_eur / fact_hkd: DataFrames with columns ['Code analytique', 'Année', 'Montant'].
    - hkd_to_eur_rate: string/number like '0.11', '0,11'.
    Applies HKD->EUR rate to ALL HKD totals (N and N-1) if provided.
    """
    df = df_main.copy()

    # Determine latest years present across EUR/HKD fact tables
    years = []
    for sub in (fact_eur, fact_hkd):
        if sub is not None and not sub.empty and "Année" in sub.columns:
            years.extend(pd.to_numeric(sub["Année"], errors="coerce").dropna().astype(int).tolist())
    years = sorted(set(y for y in years if 2000 <= y <= 2199))
    if not years:
        df["Facturation (convertie) N"] = 0.0
        df["Facturation (convertie) N-1"] = 0.0
        return df

    yearN = years[-1]
    yearN1 = years[-2] if len(years) >= 2 else None

    def agg(sub: pd.DataFrame, year: Optional[int]) -> pd.Series:
        if sub is None or sub.empty or year is None:
            return pd.Series(dtype=float)
        tmp = sub[pd.to_numeric(sub["Année"], errors="coerce") == year]
        if tmp.empty:
            return pd.Series(dtype=float)
        return tmp.groupby("Code analytique", dropna=False)["Montant"].sum()

    eur_N,  eur_N1  = agg(fact_eur, yearN),  agg(fact_eur, yearN1)
    hkd_N,  hkd_N1  = agg(fact_hkd, yearN),  agg(fact_hkd, yearN1)

    # Map totals (per code) onto both 'Code analytique (cf session)' and '(pdt)'
    df = _map_series_on_two_codes(df, eur_N,  "EUR_N")
    df = _map_series_on_two_codes(df, eur_N1, "EUR_N1")
    df = _map_series_on_two_codes(df, hkd_N,  "HKD_N")
    df = _map_series_on_two_codes(df, hkd_N1, "HKD_N1")

    # Apply HKD->EUR conversion rate to ALL HKD totals if provided
    rate = _parse_rate(hkd_to_eur_rate)
    if rate is not None:
        for col in ["HKD_N", "HKD_N1"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) * rate

    # Final converted totals
    def c0(x): return pd.to_numeric(x, errors="coerce").fillna(0.0)
    df["Facturation (convertie) N"]   = c0(df.get("EUR_N", 0))   + c0(df.get("HKD_N", 0))
    df["Facturation (convertie) N-1"] = c0(df.get("EUR_N1", 0)) + c0(df.get("HKD_N1", 0)) if yearN1 else 0.0

    # Cleanup temp cols
    for c in ["EUR_N", "EUR_N1", "HKD_N", "HKD_N1"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    return df

def compute_ca_attendu(df: pd.DataFrame) -> pd.DataFrame:
    def pick(row):
        orig = str(row.get("Origine rapport", "")).strip().lower()
        if orig == "sans_session":
            return row.get("Prix total")
        if orig == "intra":
            return row.get("Prix Intra 1 standard (converti)")
        if orig == "inter":
            return row.get("CA session (converti)", row.get("Prix de vente (converti)"))
        return pd.NA
    df["CA attendu"] = df.apply(pick, axis=1)
    return df

def normalize_to_target(df: pd.DataFrame) -> pd.DataFrame:
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    for c in INTERMEDIATE_MONETARY_COLS:
        if c in df.columns and c not in TARGET_COLUMNS:
            df.drop(columns=[c], inplace=True)
    return df[TARGET_COLUMNS]
