# -*- coding: utf-8 -*-

from __future__ import annotations
import re
import pandas as pd
from typing import List, Optional, Tuple
from .settings import TARGET_COLUMNS, INTERMEDIATE_MONETARY_COLS

# -------------------------------------------------------
# Utils
# -------------------------------------------------------

def pipeline_concat(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dfs, ignore_index=True, sort=False) if dfs else pd.DataFrame()

def _c0(series):
    return pd.to_numeric(series, errors="coerce").fillna(0.0)

def _parse_rate(val) -> Optional[float]:
    """
    Taux saisi = HKD par 1 EUR (as-is).
    - "0,116" -> 0.116
    - "0.116" -> 0.116
    - "9"     -> 9.0
    """
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9eE\+\-\.]", "", s)
    try:
        return float(s)
    except Exception:
        return None

def _mdy_to_datetime(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, format="%m/%d/%y", errors="coerce")
    except Exception:
        return pd.to_datetime(s, errors="coerce", dayfirst=False, yearfirst=False)

def _overlap_fraction(start: pd.Series, end: pd.Series,
                      win_start: pd.Timestamp, win_end: pd.Timestamp) -> pd.Series:
    dur = (end - start).dt.days
    dur = dur.where(dur > 0, other=pd.NA)
    inter_start = pd.concat([start, pd.Series([win_start]*len(start), index=start.index)], axis=1).max(axis=1)
    inter_end   = pd.concat([end,   pd.Series([win_end]*len(end),   index=end.index)],   axis=1).min(axis=1)
    inter_days  = (inter_end - inter_start).dt.days
    inter_days  = inter_days.where(inter_days > 0, other=0)
    frac = (inter_days / dur).fillna(1.0).clip(lower=0.0, upper=1.0)
    return frac

# -------------------------------------------------------
# Facturation (Y, Y-1, Y-2 + N, N-1)
# -------------------------------------------------------

def _agg_by_year(sub: pd.DataFrame, year: Optional[int]) -> pd.Series:
    if sub is None or sub.empty or year is None:
        return pd.Series(dtype=float)
    tmp = sub[pd.to_numeric(sub["Année"], errors="coerce") == year]
    if tmp.empty:
        return pd.Series(dtype=float)
    return tmp.groupby("Code analytique", dropna=False)["Montant"].sum()

def _majority_pair(sub: pd.DataFrame) -> tuple[int|None, int|None]:
    """Retourne (N-1, N) à partir des colonnes __Y_HINT_*, si présentes."""
    if sub is None or sub.empty:
        return (None, None)
    cN  = sub.get("__Y_HINT_N")
    cN1 = sub.get("__Y_HINT_N1")
    if cN is not None and cN1 is not None:
        try:
            yN  = pd.to_numeric(cN,  errors="coerce").dropna().astype(int).mode()
            yN1 = pd.to_numeric(cN1, errors="coerce").dropna().astype(int).mode()
            yN  = int(yN.iloc[0])  if len(yN)  else None
            yN1 = int(yN1.iloc[0]) if len(yN1) else None
            return (yN1, yN)
        except Exception:
            return (None, None)
    return (None, None)

def compute_facturation_from_external(df_main: pd.DataFrame,
                                      fact_eur: pd.DataFrame,
                                      fact_hkd: pd.DataFrame,
                                      hkd_to_eur_rate) -> Tuple[pd.DataFrame, Optional[int]]:
    """
    - Construit 'Facturation Y', 'Y-1', 'Y-2', 'Facutration totale'.
    - Alimente 'Facturation (convertie) N' = 'Facturation Y', 'N-1' = 'Facturation Y-1'.
    - Conversion HKD->EUR : DIVISION par le taux (HKD par 1 EUR) et UNIQUEMENT pour Filiale FFI.
    - Les années N/N-1 sont obligatoirement celles de l'en-tête des fichiers de facturation si disponibles.
    Retourne (df, N) où N=année la plus récente utilisée.
    """
    df = df_main.copy()

    # 1) Déterminer N et N-1 depuis les hints (prioritaires)
    y1_eur, yN_eur = _majority_pair(fact_eur)
    y1_hkd, yN_hkd = _majority_pair(fact_hkd)

    if yN_eur and y1_eur:
        yearY, yearY1 = yN_eur, y1_eur
    elif yN_hkd and y1_hkd:
        yearY, yearY1 = yN_hkd, y1_hkd
    else:
        # Fallback sur les valeurs présentes
        years = []
        for sub in (fact_eur, fact_hkd):
            if sub is not None and not sub.empty and "Année" in sub.columns:
                years.extend(pd.to_numeric(sub["Année"], errors="coerce").dropna().astype(int).tolist())
        years = sorted(set(y for y in years if 2000 <= y <= 2199))
        if not years:
            for col in ["Facturation Y-2", "Facturation Y-1", "Facturation Y",
                        "Facutration totale", "Facturation (convertie) N", "Facturation (convertie) N-1"]:
                df[col] = 0.0
            return df, None
        yearY  = years[-1]
        yearY1 = years[-2] if len(years) >= 2 else None

    # On peut déduire Y-2
    yearY2 = yearY1 - 1 if yearY1 is not None else None

    # 2) Agrégation par année / code
    eur_Y   = _agg_by_year(fact_eur, yearY)
    eur_Y1  = _agg_by_year(fact_eur, yearY1)
    eur_Y2  = _agg_by_year(fact_eur, yearY2)
    hkd_Y   = _agg_by_year(fact_hkd, yearY)
    hkd_Y1  = _agg_by_year(fact_hkd, yearY1)
    hkd_Y2  = _agg_by_year(fact_hkd, yearY2)

    # 3) Projection sur cf/pdt puis somme
    def _map2(df0, ser, name):
        if ser is None or ser.empty:
            df0[name] = 0.0
            return df0
        d = df0.merge(ser.rename(f"__{name}_cf"), how="left",
                      left_on="Code analytique (cf session)", right_index=True)
        d = d.merge(ser.rename(f"__{name}_pdt"), how="left",
                    left_on="Code analytique (pdt)", right_index=True)
        d[name] = _c0(d.get(f"__{name}_cf")) + _c0(d.get(f"__{name}_pdt"))
        d.drop(columns=[c for c in [f"__{name}_cf", f"__{name}_pdt"] if c in d.columns], inplace=True)
        return d

    df = _map2(df, eur_Y,  "EUR_Y")
    df = _map2(df, eur_Y1, "EUR_Y1")
    df = _map2(df, eur_Y2, "EUR_Y2")
    df = _map2(df, hkd_Y,  "HKD_Y")
    df = _map2(df, hkd_Y1, "HKD_Y1")
    df = _map2(df, hkd_Y2, "HKD_Y2")

    # 4) Conversion HKD->EUR (DIVISION par le taux), seulement pour Filiale FFI
    rate = _parse_rate(hkd_to_eur_rate)
    if rate is not None and rate != 0:
        fil = df.get("Filiale First Finance", "").astype(str).str.lower()

        def _is_ffi(name: str) -> bool:
            n = name
            if "ffi" in n:
                return True
            return ("first" in n and "finance" in n and ("institute" in n or "institue" in n))

        is_ffi = fil.apply(_is_ffi)

        for col in ["HKD_Y", "HKD_Y1", "HKD_Y2"]:
            if col in df.columns:
                df[col] = df[col].where(~is_ffi, other=_c0(df[col]) / rate)

    # 5) Sorties
    df["Facturation Y"]   = _c0(df.get("EUR_Y", 0))  + _c0(df.get("HKD_Y", 0))
    df["Facturation Y-1"] = (_c0(df.get("EUR_Y1", 0)) + _c0(df.get("HKD_Y1", 0))) if yearY1 else 0.0
    df["Facturation Y-2"] = (_c0(df.get("EUR_Y2", 0)) + _c0(df.get("HKD_Y2", 0))) if yearY2 else 0.0
    df["Facutration totale"] = _c0(df["Facturation Y"]) + _c0(df["Facturation Y-1"]) + _c0(df["Facturation Y-2"])

    # Colonnes “historiques” demandées
    df["Facturation (convertie) N"]   = df["Facturation Y"]
    df["Facturation (convertie) N-1"] = df["Facturation Y-1"]

    # nettoyage colonnes temporaires
    for c in ["EUR_Y","EUR_Y1","EUR_Y2","HKD_Y","HKD_Y1","HKD_Y2"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    return df, yearY

# -------------------------------------------------------
# CA attendu
# -------------------------------------------------------

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

# -------------------------------------------------------
# Dates repère (prévisionnel)
# -------------------------------------------------------

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

# -------------------------------------------------------
# Avancement / Backlog / FAE / PCA
# -------------------------------------------------------

def compute_progress_and_backlog(df: pd.DataFrame,
                                 date_cloture: str,
                                 debut_exercice: str,
                                 fin_exercice: str,
                                 closure_year_hint: Optional[int]) -> pd.DataFrame:
    """
    - Avancement global = overlap([Date début, Date fin], [Début d'exercice, Date de clôture]), fallback 1 si durée invalide.
    - Avancement EOY :
        * INTRA / SANS_SESSION => règle Excel sur dates REPÈRE (voir commentaire)
        * INTER (autres) => overlap([Clôture, Fin d'exercice]) sur dates RÉELLES
    - CA avancement, CA YTD
    - CA EOY (backlog) avec gating sur année de fin REPÈRE >= Année de clôture
    - FAE / PCA vs Facutration totale
    """
    df = df.copy()

    # Dates réelles et repère
    ddeb_dt     = _mdy_to_datetime(df.get("Date de début", ""))
    dfin_dt     = _mdy_to_datetime(df.get("Date de fin", ""))
    ddeb_rep_dt = _mdy_to_datetime(df.get("Date de début repère", ""))
    dfin_rep_dt = _mdy_to_datetime(df.get("Date de fin repère", ""))

    # Dates de contrôle
    dc = pd.to_datetime(date_cloture, errors="coerce")
    de = pd.to_datetime(debut_exercice, errors="coerce")
    fe = pd.to_datetime(fin_exercice, errors="coerce")

    # Avancement global (réelles) sur [DébutEx, Clôture]
    if pd.notna(de) and pd.notna(dc):
        frac_global = _overlap_fraction(ddeb_dt, dfin_dt, de, dc)
    else:
        frac_global = pd.Series([1.0] * len(df), index=df.index)

    statut = df.get("Statut", "").astype(str).str.lower().str.strip()
    is_annulee = statut.eq("annulée") | statut.eq("annulee")
    frac_global = frac_global.where(~is_annulee, other=0.0)
    df["Avancement global"] = frac_global

    # Origine
    origin = df.get("Origine rapport", "").astype(str).str.lower().str.strip()
    is_intra_or_ss = origin.isin(["intra", "sans_session"])

    # Année de clôture
    if closure_year_hint is not None:
        cloture_year = closure_year_hint
    else:
        cloture_year = pd.to_datetime(date_cloture, errors="coerce").year if pd.notna(dc) else None

    # EOY par défaut (INTER) sur réelles : [Clôture, FinEx]
    if pd.notna(dc) and pd.notna(fe):
        frac_eoy_default = _overlap_fraction(ddeb_dt, dfin_dt, dc, fe)
    else:
        frac_eoy_default = pd.Series([0.0] * len(df), index=df.index)

    # EOY pour INTRA/SANS_SESSION sur REPÈRE (formule Excel)
    frac_eoy_intra_ss = pd.Series([0.0] * len(df), index=df.index, dtype=float)
    if pd.notna(dc):
        fin_year_rep = dfin_rep_dt.dt.year
        cond_annulee   = is_annulee
        cond_year_lt   = (fin_year_rep < cloture_year) if cloture_year is not None else pd.Series([False] * len(df), index=df.index)
        cond_fin_lt_dc = (dfin_rep_dt < dc)
        cond_deb_gt_dc = (ddeb_rep_dt > dc)

        frac = pd.Series([0.0] * len(df), index=df.index, dtype=float)
        # AE(repère) < Clôture => 1
        frac = frac.where(~cond_fin_lt_dc, other=1.0)
        # sinon ratio (Clôture - Début_repère) / (Fin_repère - Début_repère), fallback 1 si durée invalide
        dur = (dfin_rep_dt - ddeb_rep_dt).dt.days
        with pd.option_context('mode.use_inf_as_na', True):
            ratio = ((dc - ddeb_rep_dt).dt.days / dur.replace(0, pd.NA)).fillna(1.0).clip(lower=0.0, upper=1.0)
        need_ratio = ~(cond_fin_lt_dc | cond_deb_gt_dc)
        frac = frac.where(~need_ratio, other=ratio)

        # gates
        frac = frac.where(~cond_year_lt, other=0.0)
        frac = frac.where(~cond_deb_gt_dc, other=0.0)
        frac = frac.where(~cond_annulee,  other=0.0)

        frac_eoy_intra_ss = frac

    # Choix final
    frac_eoy = frac_eoy_default.where(~is_intra_or_ss, frac_eoy_intra_ss)
    frac_eoy = frac_eoy.where(~is_annulee, other=0.0)
    df["Avancement EOY"] = frac_eoy

    # CA dérivés
    ca_att = _c0(df.get("CA attendu", 0))
    df["CA avancement"] = ca_att * df["Avancement global"]
    df["CA YTD"]        = (1 - is_annulee.astype(int)) * ca_att * df["Avancement global"]

    # Backlog gate : année de fin REPÈRE >= année de clôture
    fin_year_repere = dfin_rep_dt.dt.year
    if cloture_year is None:
        after_gate = pd.Series([1] * len(df), index=df.index)
    else:
        after_gate = (fin_year_repere >= cloture_year).fillna(False).astype(int)

    df["CA EOY (backlog)"] = after_gate * ca_att * df["Avancement EOY"]

    # FAE / PCA
    fact_tot = _c0(df.get("Facutration totale", 0))
    df["FAE"] = after_gate * (df["CA avancement"] - fact_tot).clip(lower=0.0)
    df["PCA"] = after_gate * (fact_tot - df["CA avancement"]).clip(lower=0.0)

    return df

# -------------------------------------------------------
# Normalisation
# -------------------------------------------------------

def normalize_to_target(df: pd.DataFrame) -> pd.DataFrame:
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    for c in INTERMEDIATE_MONETARY_COLS:
        if c in df.columns and c not in TARGET_COLUMNS:
            df.drop(columns=[c], inplace=True)
    return df[TARGET_COLUMNS]
