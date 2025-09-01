# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
from typing import List, Optional, Tuple
from .settings import TARGET_COLUMNS, INTERMEDIATE_MONETARY_COLS
from .helpers import export_excel

# ---------------------------- helpers ----------------------------

def pipeline_concat(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dfs, ignore_index=True, sort=False) if dfs else pd.DataFrame()

def _c0(series):
    return pd.to_numeric(series, errors="coerce").fillna(0.0)

def _parse_rate(val) -> Optional[float]:
    """Accepts '0.11', '0,11'. Returns float or None."""
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

def _mdy_to_datetime(s: pd.Series) -> pd.Series:
    """
    Convert our standard m/d/yy strings to datetime. Falls back to coercion.
    """
    try:
        return pd.to_datetime(s, format="%m/%d/%y", errors="coerce")
    except Exception:
        return pd.to_datetime(s, errors="coerce", dayfirst=False, yearfirst=False)

def _overlap_fraction(start: pd.Series, end: pd.Series,
                      win_start: pd.Timestamp, win_end: pd.Timestamp) -> pd.Series:
    """
    Fraction of [start, end] that overlaps [win_start, win_end].
    If duration <= 0 or invalid dates -> fallback 1 (as in Excel SI.ERREUR(...;1)).
    Values outside window -> 0.
    """
    s = start.copy()
    e = end.copy()

    dur = (e - s).dt.days
    dur = dur.where(dur > 0, other=pd.NA)

    inter_start = pd.concat([s, pd.Series([win_start] * len(s), index=s.index)], axis=1).max(axis=1)
    inter_end   = pd.concat([e, pd.Series([win_end] * len(e), index=e.index)], axis=1).min(axis=1)
    inter_days  = (inter_end - inter_start).dt.days
    inter_days  = inter_days.where(inter_days > 0, other=0)

    frac = inter_days / dur
    # Fallback to 1 when duration invalid (mimic SI.ERREUR(...;1))
    frac = frac.fillna(1.0)
    # Clip [0,1]
    frac = frac.clip(lower=0.0, upper=1.0)
    return frac

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
    df[out_col] = _c0(df.get(f"__{out_col}_cf")) + _c0(df.get(f"__{out_col}_pdt"))
    for c in [f"__{out_col}_cf", f"__{out_col}_pdt"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    return df

# ---------------------------- facturation ----------------------------

def _agg_by_year(sub: pd.DataFrame, year: Optional[int]) -> pd.Series:
    if sub is None or sub.empty or year is None:
        return pd.Series(dtype=float)
    tmp = sub[pd.to_numeric(sub["Année"], errors="coerce") == year]
    if tmp.empty:
        return pd.Series(dtype=float)
    return tmp.groupby("Code analytique", dropna=False)["Montant"].sum()

def compute_facturation_from_external(df_main: pd.DataFrame,
                                      fact_eur: pd.DataFrame,
                                      fact_hkd: pd.DataFrame,
                                      hkd_to_eur_rate) -> Tuple[pd.DataFrame, Optional[int]]:
    """
    - fact_eur / fact_hkd: ['Code analytique','Année','Montant']
    - Returns (df_with_fact, closure_year_detected_from_facts_or_None)
    Now computes Y, Y-1, Y-2 and 'Facutration total' (sum of the three).
    Applies HKD->EUR rate to ALL HKD totals if provided.
    """
    df = df_main.copy()

    # Determine available fact years
    years = []
    for sub in (fact_eur, fact_hkd):
        if sub is not None and not sub.empty and "Année" in sub.columns:
            years.extend(pd.to_numeric(sub["Année"], errors="coerce").dropna().astype(int).tolist())
    years = sorted(set(y for y in years if 2000 <= y <= 2199))
    if not years:
        for col in ["Facturation Y-2", "Facturation Y-1", "Facturation Y", "Facutration totale"]:
            df[col] = 0.0
        return df, None

    yearY  = years[-1]               # most recent
    yearY1 = years[-2] if len(years) >= 2 else None
    yearY2 = years[-3] if len(years) >= 3 else None

    # aggregate per code/year
    eur_Y,  eur_Y1,  eur_Y2  = _agg_by_year(fact_eur, yearY),  _agg_by_year(fact_eur, yearY1),  _agg_by_year(fact_eur, yearY2)
    hkd_Y,  hkd_Y1,  hkd_Y2  = _agg_by_year(fact_hkd, yearY),  _agg_by_year(fact_hkd, yearY1),  _agg_by_year(fact_hkd, yearY2)

    # map onto both code columns
    df = _map_series_on_two_codes(df, eur_Y,  "EUR_Y")
    df = _map_series_on_two_codes(df, eur_Y1, "EUR_Y1")
    df = _map_series_on_two_codes(df, eur_Y2, "EUR_Y2")
    df = _map_series_on_two_codes(df, hkd_Y,  "HKD_Y")
    df = _map_series_on_two_codes(df, hkd_Y1, "HKD_Y1")
    df = _map_series_on_two_codes(df, hkd_Y2, "HKD_Y2")

    # HKD conversion
    rate = _parse_rate(hkd_to_eur_rate)
    if rate is not None:
        for col in ["HKD_Y", "HKD_Y1", "HKD_Y2"]:
            if col in df.columns:
                df[col] = _c0(df[col]) * rate

    # outputs
    df["Facturation Y"]   = _c0(df.get("EUR_Y", 0))   + _c0(df.get("HKD_Y", 0))
    df["Facturation Y-1"] = _c0(df.get("EUR_Y1", 0))  + _c0(df.get("HKD_Y1", 0)) if yearY1 else 0.0
    df["Facturation Y-2"] = _c0(df.get("EUR_Y2", 0))  + _c0(df.get("HKD_Y2", 0)) if yearY2 else 0.0
    df["Facutration totale"] = _c0(df["Facturation Y"]) + _c0(df["Facturation Y-1"]) + _c0(df["Facturation Y-2"])

    # cleanup temps
    for c in ["EUR_Y","EUR_Y1","EUR_Y2","HKD_Y","HKD_Y1","HKD_Y2"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    return df, yearY

# ---------------------------- CA attendu ----------------------------

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

# ---------------------------- Repère dates ----------------------------

def compute_dates_repere(df: pd.DataFrame) -> pd.DataFrame:
    """
    Date de début/fin repère:
      - si 'Date d'éxécution (produit sans session)' non vide -> début = fin = exécution
      - sinon début = 'Date de début', fin = 'Date de fin'
    """
    exec_s = df.get("Date d'éxécution (produit sans session)", pd.Series([""] * len(df)))
    ddeb_s = df.get("Date de début", "")
    dfin_s = df.get("Date de fin", "")

    exec_dt = _mdy_to_datetime(exec_s)
    ddeb_dt = _mdy_to_datetime(ddeb_s)
    dfin_dt = _mdy_to_datetime(dfin_s)

    use_exec = exec_s.astype(str).str.strip().ne("")

    debut_repere_dt = ddeb_dt.where(~use_exec, exec_dt)
    fin_repere_dt   = dfin_dt.where(~use_exec, exec_dt)

    df["Date de début repère"] = debut_repere_dt.apply(lambda d: f"{d.month}/{d.day}/{str(d.year)[-2:]}" if pd.notna(d) else "")
    df["Date de fin repère"]   = fin_repere_dt.apply(  lambda d: f"{d.month}/{d.day}/{str(d.year)[-2:]}" if pd.notna(d) else "")
    return df

# ---------------------------- Avancement / FAE / PCA ----------------------------

def compute_progress_and_backlog(df: pd.DataFrame,
                                 date_cloture: str,
                                 debut_exercice: str,
                                 fin_exercice: str,
                                 closure_year_hint: Optional[int]) -> pd.DataFrame:
    """
      - Avancement global: overlap([Début,Fin], [DébutEx, DateClôture]) with fallback 1 if zero-length
      - Avancement EOY:    overlap([Début,Fin], [DateClôture, FinEx]) with same fallback
      - Status "Annulée" forces both to 0
      - CA avancement = Avancement global * CA attendu
      - CA YTD = (not Annulée) * Avancement global * CA attendu
      - CA EOY (backlog) = (year(Fin)>=Année_clôture) * Avancement EOY * CA attendu
      - FAE = (year(Fin)>=Année_clôture) * max(CA avancement - Facutration totale, 0)
      - PCA = (year(Fin)>=Année_clôture) * max(Facutration totale - CA avancement, 0)
    Works for INTRA / SANS_SESSION / INTER uniformly.
    """
    df = df.copy()

    # Dates
    ddeb_dt = _mdy_to_datetime(df.get("Date de début", ""))
    dfin_dt = _mdy_to_datetime(df.get("Date de fin", ""))

    try:
        dc = pd.to_datetime(date_cloture, errors="coerce")
    except Exception:
        dc = pd.NaT
    try:
        de = pd.to_datetime(debut_exercice, errors="coerce")
    except Exception:
        de = pd.NaT
    try:
        fe = pd.to_datetime(fin_exercice, errors="coerce")
    except Exception:
        fe = pd.NaT

    # Fractions
    if pd.notna(de) and pd.notna(dc):
        frac_global = _overlap_fraction(ddeb_dt, dfin_dt, de, dc)
    else:
        frac_global = pd.Series([1.0] * len(df), index=df.index)

    if pd.notna(dc) and pd.notna(fe):
        frac_eoy = _overlap_fraction(ddeb_dt, dfin_dt, dc, fe)
    else:
        frac_eoy = pd.Series([0.0] * len(df), index=df.index)

    # Status annulée -> 0
    statut = df.get("Statut", "").astype(str).str.lower().str.strip()
    is_annulee = statut.eq("annulée") | statut.eq("annulee")
    frac_global = frac_global.where(~is_annulee, other=0.0)
    frac_eoy    = frac_eoy.where(~is_annulee,    other=0.0)

    df["Avancement global"] = frac_global
    df["Avancement EOY"]    = frac_eoy

    # CA attendu
    ca_att = _c0(df.get("CA attendu", 0))

    # Derived CAs
    df["CA avancement"]      = ca_att * df["Avancement global"]
    df["CA YTD"]             = (1 - is_annulee.astype(int)) * ca_att * df["Avancement global"]
    # year gating
    fin_year = dfin_dt.dt.year
    clo_year = None
    if closure_year_hint is not None:
        clo_year = closure_year_hint
    else:
        clo_year = pd.to_datetime(date_cloture, errors="coerce").year if pd.notna(dc) else None

    if clo_year is None:
        after_gate = pd.Series([1] * len(df), index=df.index)
    else:
        after_gate = (fin_year >= clo_year).fillna(False).astype(int)

    df["CA EOY (backlog)"]   = after_gate * ca_att * df["Avancement EOY"]

    # Facturation totale (already computed)
    fact_tot = _c0(df.get("Facutration totale", 0))

    df["FAE"] = after_gate * (df["CA avancement"] - fact_tot).clip(lower=0.0)
    df["PCA"] = after_gate * (fact_tot - df["CA avancement"]).clip(lower=0.0)

    return df

# ---------------------------- Normalize to target ----------------------------

def normalize_to_target(df: pd.DataFrame) -> pd.DataFrame:
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    for c in INTERMEDIATE_MONETARY_COLS:
        if c in df.columns and c not in TARGET_COLUMNS:
            df.drop(columns=[c], inplace=True)
    return df[TARGET_COLUMNS]
