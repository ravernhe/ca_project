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

def _mdy_to_datetime(s: pd.Series) -> pd.Series:
    """Convert standard m/d/yy strings to datetime. Fallback = coercion."""
    try:
        return pd.to_datetime(s, format="%m/%d/%y", errors="coerce")
    except Exception:
        return pd.to_datetime(s, errors="coerce", dayfirst=False, yearfirst=False)

def _overlap_fraction(start: pd.Series, end: pd.Series,
                      win_start: pd.Timestamp, win_end: pd.Timestamp) -> pd.Series:
    """
    Fraction of [start, end] overlapping [win_start, win_end].
    Duration <= 0 → fallback 1. Out of window → 0.
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
    frac = frac.fillna(1.0).clip(lower=0.0, upper=1.0)
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
    Computes Y, Y-1, Y-2 and 'Facutration totale'.
    Applies HKD->EUR ONLY for rows where 'Filiale First Finance' contains 'FFI'.
    """
    df = df_main.copy()

    # Available years
    years = []
    for sub in (fact_eur, fact_hkd):
        if sub is not None and not sub.empty and "Année" in sub.columns:
            years.extend(pd.to_numeric(sub["Année"], errors="coerce").dropna().astype(int).tolist())
    years = sorted(set(y for y in years if 2000 <= y <= 2199))
    if not years:
        for col in ["Facturation Y-2", "Facturation Y-1", "Facturation Y", "Facutration totale"]:
            df[col] = 0.0
        return df, None

    yearY  = years[-1]
    yearY1 = years[-2] if len(years) >= 2 else None
    yearY2 = years[-3] if len(years) >= 3 else None

    eur_Y, eur_Y1, eur_Y2 = _agg_by_year(fact_eur, yearY), _agg_by_year(fact_eur, yearY1), _agg_by_year(fact_eur, yearY2)
    hkd_Y, hkd_Y1, hkd_Y2 = _agg_by_year(fact_hkd, yearY), _agg_by_year(fact_hkd, yearY1), _agg_by_year(fact_hkd, yearY2)

    # Map per code
    df = _map_series_on_two_codes(df, eur_Y,  "EUR_Y")
    df = _map_series_on_two_codes(df, eur_Y1, "EUR_Y1")
    df = _map_series_on_two_codes(df, eur_Y2, "EUR_Y2")
    df = _map_series_on_two_codes(df, hkd_Y,  "HKD_Y")
    df = _map_series_on_two_codes(df, hkd_Y1, "HKD_Y1")
    df = _map_series_on_two_codes(df, hkd_Y2, "HKD_Y2")

    # HKD conversion only for FFI
    rate = _parse_rate(hkd_to_eur_rate)
    if rate is not None:
        filiale = df.get("Filiale First Finance", "").astype(str).str.lower()
        is_ffi = filiale.str.contains("ffi", na=False)
        for col in ["HKD_Y", "HKD_Y1", "HKD_Y2"]:
            if col in df.columns:
                df[col] = df[col].where(~is_ffi, other=_c0(df[col]) * rate)

    df["Facturation Y"]   = _c0(df.get("EUR_Y", 0))  + _c0(df.get("HKD_Y", 0))
    df["Facturation Y-1"] = (_c0(df.get("EUR_Y1", 0)) + _c0(df.get("HKD_Y1", 0))) if yearY1 else 0.0
    df["Facturation Y-2"] = (_c0(df.get("EUR_Y2", 0)) + _c0(df.get("HKD_Y2", 0))) if yearY2 else 0.0
    df["Facutration totale"] = _c0(df["Facturation Y"]) + _c0(df["Facturation Y-1"]) + _c0(df["Facturation Y-2"])

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


# ---------------------------- Dates repère ----------------------------

def compute_dates_repere(df: pd.DataFrame) -> pd.DataFrame:
    """
    Date de début repère  = Date prévisionnelle de début de projet (sinon Date de début)
    Date de fin repère    = Date de fin de projet (sinon Date de fin)
    """
    def _parse_mdy(series):
        try:
            return pd.to_datetime(series, format="%m/%d/%y", errors="coerce")
        except Exception:
            return pd.to_datetime(series, errors="coerce", dayfirst=False, yearfirst=False)

    prev_deb = df.get("Date prévisionnelle de début de projet", "")
    prev_fin = df.get("Date de fin de projet", "")
    deb      = df.get("Date de début", "")
    fin      = df.get("Date de fin", "")

    prev_deb_dt = _parse_mdy(prev_deb)
    prev_fin_dt = _parse_mdy(prev_fin)
    deb_dt      = _parse_mdy(deb)
    fin_dt      = _parse_mdy(fin)

    use_prev_deb = prev_deb.astype(str).str.strip().ne("")
    use_prev_fin = prev_fin.astype(str).str.strip().ne("")

    deb_repere_dt = deb_dt.where(~use_prev_deb, prev_deb_dt)
    fin_repere_dt = fin_dt.where(~use_prev_fin, prev_fin_dt)

    df["Date de début repère"] = deb_repere_dt.apply(lambda d: f"{d.month}/{d.day}/{str(d.year)[-2:]}" if pd.notna(d) else "")
    df["Date de fin repère"]   = fin_repere_dt.apply(  lambda d: f"{d.month}/{d.day}/{str(d.year)[-2:]}" if pd.notna(d) else "")
    return df


# ---------------------------- Avancement / Backlog / FAE / PCA ----------------------------

def compute_progress_and_backlog(df: pd.DataFrame,
                                 date_cloture: str,
                                 debut_exercice: str,
                                 fin_exercice: str,
                                 closure_year_hint: Optional[int]) -> pd.DataFrame:
    """
    - Avancement global = overlap([Date début, Date fin], [Début d'exercice, Date de clôture]), fallback 1 if zero-length
    - Avancement EOY:
        * INTRA / SANS_SESSION  ⇒ Excel rule on REPÈRE dates:
            if Annulée → 0
            elif year(Fin_repère) < Année_clôture → 0
            elif Fin_repère < Date_clôture → 1
            elif Début_repère > Date_clôture → 0
            else IFERROR( (Date_clôture - Début_repère) / (Fin_repère - Début_repère) ; 1 )
        * INTER (others)        ⇒ overlap([Date_clôture, Fin d'exercice]) on REAL dates
    - CA avancement  = Avancement global * CA attendu
    - CA YTD         = (not Annulée) * Avancement global * CA attendu
    - CA EOY (backlog) uses REPÈRE-year gate: year(Date de fin repère) ≥ Année_clôture
    - FAE = gate * max(CA avancement - Facutration totale, 0)
    - PCA = gate * max(Facutration totale - CA avancement, 0)
    """
    df = df.copy()

    # Parse data dates (stored as m/d/yy strings)
    def _mdy(series):
        try:
            return pd.to_datetime(series, format="%m/%d/%y", errors="coerce")
        except Exception:
            return pd.to_datetime(series, errors="coerce", dayfirst=False, yearfirst=False)

    ddeb_dt      = _mdy(df.get("Date de début", ""))
    dfin_dt      = _mdy(df.get("Date de fin", ""))
    ddeb_rep_dt  = _mdy(df.get("Date de début repère", ""))
    dfin_rep_dt  = _mdy(df.get("Date de fin repère", ""))

    # Parse control dates from GUI (ISO expected)
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

    # Avancement global: overlap on REAL dates with [Début Ex, Clôture]
    if pd.notna(de) and pd.notna(dc):
        frac_global = _overlap_fraction(ddeb_dt, dfin_dt, de, dc)
    else:
        frac_global = pd.Series([1.0] * len(df), index=df.index)

    # Status 'Annulée' handling
    statut = df.get("Statut", "").astype(str).str.lower().str.strip()
    is_annulee = statut.eq("annulée") | statut.eq("annulee")
    frac_global = frac_global.where(~is_annulee, other=0.0)
    df["Avancement global"] = frac_global

    # Origin
    origin = df.get("Origine rapport", "").astype(str).str.lower().str.strip()
    is_intra_or_ss = origin.isin(["intra", "sans_session"])

    # Closure year
    if closure_year_hint is not None:
        cloture_year = closure_year_hint
    else:
        cloture_year = pd.to_datetime(date_cloture, errors="coerce").year if pd.notna(dc) else None

    # Default EOY (for INTER): overlap on REAL dates with [Clôture, Fin Ex]
    if pd.notna(dc) and pd.notna(fe):
        frac_eoy_default = _overlap_fraction(ddeb_dt, dfin_dt, dc, fe)
    else:
        frac_eoy_default = pd.Series([0.0] * len(df), index=df.index)

    # INTRA / SANS_SESSION EOY per Excel using REPÈRE dates
    frac_eoy_intra_ss = pd.Series([0.0] * len(df), index=df.index, dtype=float)
    if pd.notna(dc):
        fin_year_rep = dfin_rep_dt.dt.year
        cond_annulee   = is_annulee
        cond_year_lt   = (fin_year_rep < cloture_year) if cloture_year is not None else pd.Series([False] * len(df), index=df.index)
        cond_fin_lt_dc = (dfin_rep_dt < dc)
        cond_deb_gt_dc = (ddeb_rep_dt > dc)

        # Start from zeros
        frac = pd.Series([0.0] * len(df), index=df.index, dtype=float)

        # AE(repère) < Clôture ⇒ 1
        frac = frac.where(~cond_fin_lt_dc, other=1.0)

        # Else ratio on repère dates, fallback 1 if invalid duration
        dur = (dfin_rep_dt - ddeb_rep_dt).dt.days
        with pd.option_context('mode.use_inf_as_na', True):
            ratio = ((dc - ddeb_rep_dt).dt.days / dur.replace(0, pd.NA)).fillna(1.0).clip(lower=0.0, upper=1.0)
        need_ratio = ~(cond_fin_lt_dc | cond_deb_gt_dc)
        frac = frac.where(~need_ratio, other=ratio)

        # Apply gates & annulée
        frac = frac.where(~cond_year_lt, other=0.0)
        frac = frac.where(~cond_deb_gt_dc, other=0.0)
        frac = frac.where(~cond_annulee,  other=0.0)

        frac_eoy_intra_ss = frac

    # Choose per row
    frac_eoy = frac_eoy_default.where(~is_intra_or_ss, frac_eoy_intra_ss)
    frac_eoy = frac_eoy.where(~is_annulee, other=0.0)
    df["Avancement EOY"] = frac_eoy

    # CA derivations
    ca_att = _c0(df.get("CA attendu", 0))
    df["CA avancement"] = ca_att * df["Avancement global"]
    df["CA YTD"]        = (1 - is_annulee.astype(int)) * ca_att * df["Avancement global"]

    # Backlog gate uses REPÈRE end year
    fin_year_repere = dfin_rep_dt.dt.year
    if cloture_year is None:
        after_gate = pd.Series([1] * len(df), index=df.index)
    else:
        after_gate = (fin_year_repere >= cloture_year).fillna(False).astype(int)

    df["CA EOY (backlog)"] = after_gate * ca_att * df["Avancement EOY"]

    # FAE / PCA versus 'Facutration totale'
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

