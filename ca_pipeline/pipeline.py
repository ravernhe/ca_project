# -*- coding: utf-8 -*-

from __future__ import annotations
import re
import pandas as pd
from typing import List, Optional, Tuple
from .settings import TARGET_COLUMNS, INTERMEDIATE_MONETARY_COLS
from .helpers import _clean_str_series

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

def _agg_by_year_with_cutoff(df: pd.DataFrame, year: int, cutoff: pd.Timestamp | None) -> pd.Series:
    """
    Agrège 'Montant' par 'Code analytique' sur l'année demandée.
    Si cutoff est fourni et appartient à 'year', borne à DateDT <= cutoff.
    Retourne une Series indexée par Code analytique (somme des montants).
    """
    if df is None or df.empty or year is None:
        return pd.Series(dtype="float64")
    d = df.copy()
    if "Année" not in d.columns:
        return pd.Series(dtype="float64")
    mask = d["Année"].astype("Int64") == year
    if cutoff is not None and cutoff.year == year and "DateDT" in d.columns:
        # si DateDT absent/NaT, on garde uniquement les lignes datées <= cutoff
        dd = pd.to_datetime(d["DateDT"], errors="coerce")
        mask = mask & dd.notna() & (dd <= cutoff)
    d = d.loc[mask, ["Code analytique", "Montant"]]
    if d.empty:
        return pd.Series(dtype="float64")
    return d.groupby("Code analytique")["Montant"].sum()

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
                                      hkd_to_eur_rate,
                                      date_cloture: str | None = None) -> Tuple[pd.DataFrame, Optional[int]]:
    """
    - Construit 'Facturation Y', 'Y-1', 'Y-2', 'Facturation totale'.
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
                        "Facturation totale", "Facturation (convertie) N", "Facturation (convertie) N-1"]:
                df[col] = 0.0
            return df, None
        yearY  = years[-1]
        yearY1 = years[-2] if len(years) >= 2 else None

    # On peut déduire Y-2
    yearY2 = yearY1 - 1 if yearY1 is not None else None

    # Parse souple de la date de clôture (FR accepté)
    cutoff = None
    if date_cloture:
        cut1 = pd.to_datetime(date_cloture, errors="coerce", dayfirst=True,  yearfirst=False)
        cut2 = pd.to_datetime(date_cloture, errors="coerce", dayfirst=False, yearfirst=False)
        cutoff = cut1 if pd.notna(cut1) else (cut2 if pd.notna(cut2) else None)
    
    # Année N (Y) bornée à la date de clôture si fournie
    eur_Y   = _agg_by_year_with_cutoff(fact_eur, yearY, cutoff)
    hkd_Y   = _agg_by_year_with_cutoff(fact_hkd, yearY, cutoff)
    # N-1 et N-2 sur l'année complète (pas de cut-off)
    eur_Y1  = _agg_by_year(fact_eur, yearY1)
    eur_Y2  = _agg_by_year(fact_eur, yearY2)
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
    df["Facturation totale"] = _c0(df["Facturation Y"]) + _c0(df["Facturation Y-1"]) + _c0(df["Facturation Y-2"])

    # Colonnes “historiques” demandées
    df["Facturation (convertie) N"]   = df["Facturation Y"]
    df["Facturation (convertie) N-1"] = df["Facturation Y-1"]
    df["Facturation (convertie) N-2"] = df["Facturation Y-2"]

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

def _to_dt(s: pd.Series) -> pd.Series:
    # Try MDY (e.g., 6/14/23), then DMY (14/06/23), then a permissive fallback with dayfirst
    s1 = pd.to_datetime(s, format="%m/%d/%y", errors="coerce")
    # fill the NaT with DMY parse
    need = s1.isna()
    if need.any():
        s2 = pd.to_datetime(s[need], format="%d/%m/%y", errors="coerce")
        s1.loc[need] = s2
        need = s1.isna()
    if need.any():
        s3 = pd.to_datetime(s[need], errors="coerce", dayfirst=True)
        s1.loc[need] = s3
    return s1


def _clean_str_series(s: pd.Series) -> pd.Series:
    if not isinstance(s, pd.Series):
        return pd.Series([], dtype="object")
    return s.astype(str).str.replace("\u00A0", " ", regex=False).str.strip()

def _origin_norm_series(df: pd.DataFrame) -> pd.Series:
    if "Origine rapport" not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    s = _clean_str_series(df["Origine rapport"]).str.lower()
    return s.replace({
        "sans session": "sans_session",
        "pdt ss session": "sans_session",
        "ss": "sans_session",
        "inter": "inter",
        "intra": "intra",
    })

def parse_param_date(s: str) -> pd.Timestamp:
    s = str(s or "").strip()
    if not s:
        return pd.NaT
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        y, a, b = map(int, m.groups())
        # normal YYYY-MM-DD
        if 1 <= a <= 12 and 1 <= b <= 31:
            return pd.to_datetime(f"{y:04d}-{a:02d}-{b:02d}", errors="coerce")
        # swapped YYYY-DD-MM
        if 1 <= b <= 12 and 1 <= a <= 31:
            return pd.to_datetime(f"{y:04d}-{b:02d}-{a:02d}", errors="coerce")
    # common fallbacks
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass
    return pd.to_datetime(s, errors="coerce")

def compute_avancement_eoy_v2(df: pd.DataFrame, date_cloture_str: str, fin_exercice_str: str) -> pd.DataFrame:
    df = df.copy()

    col_exec = "Date d'éxécution (produit sans session)"
    col_deb  = "Date de début"
    col_fin  = "Date de fin"

    # Parse params (tolerant)
    dc = parse_param_date(date_cloture_str)
    fe = parse_param_date(fin_exercice_str)

    # Parse data dates
    deb = _to_dt(df[col_deb]) if col_deb in df.columns else pd.Series([pd.NaT]*len(df), index=df.index)
    fin = _to_dt(df[col_fin]) if col_fin in df.columns else pd.Series([pd.NaT]*len(df), index=df.index)
    exe = _to_dt(df[col_exec]) if col_exec in df.columns else pd.Series([pd.NaT]*len(df), index=df.index)

    origin = _origin_norm_series(df)
    col_eoy = "Avancement EOY"
    if col_eoy not in df.columns:
        df[col_eoy] = 0.0

    # If params invalid -> EOY = 0
    if pd.isna(dc) or pd.isna(fe):
        df[col_eoy] = 0.0
        return df

    # ---- SANS_SESSION ----
    mask_ss = origin.eq("sans_session")
    if col_exec in df.columns:
        no_exec = mask_ss & (_clean_str_series(df[col_exec]).fillna("") == "")
        df.loc[no_exec, col_eoy] = "=NA()"
    has_exec = mask_ss & exe.notna()
    df.loc[has_exec, col_eoy] = ((exe > dc) & (exe <= fe)).astype(float)

    # ---- INTRA / INTER ----
    mask_ii = origin.isin(["intra", "inter"])

    # Consider backlog only after closure and up to FE → overlap of [dc, fe] with [deb, fin]
    # Fully before closure: fin <= dc -> 0
    fully_before = mask_ii & fin.notna() & (fin <= dc)
    # Fully after FE: deb >= fe -> 0
    fully_after  = mask_ii & deb.notna() & (deb >= fe)
    df.loc[fully_before | fully_after, col_eoy] = 0.0

    overlap_mask = mask_ii & deb.notna() & fin.notna() & ~(fully_before | fully_after)

    # Duration (avoid div by zero)
    dur = (fin - deb).dt.days

    # Overlap days inside (dc, fe] — we use (dc, fe] to exclude exact-closure day as "after"
    start = pd.concat([deb, pd.Series([dc]*len(df), index=df.index)], axis=1).max(axis=1)
    end   = pd.concat([fin, pd.Series([fe]*len(df), index=df.index)], axis=1).min(axis=1)

    # Exclude exact equality at start: if end <= dc -> 0 by earlier rule; here treat strictly after dc
    overlap_days = (end - start).dt.days.clip(lower=0)

    ratio = (overlap_days / dur.replace(0, pd.NA)).fillna(0.0).clip(lower=0.0, upper=1.0)
    df.loc[overlap_mask, col_eoy] = ratio

    # Zero-duration: project on a single day
    zero_dur = overlap_mask & (dur == 0)
    # If that single day lies in (dc, fe] → 1 else 0
    single_in_window = (deb > dc) & (deb <= fe)
    df.loc[zero_dur & single_in_window, col_eoy] = 1.0
    df.loc[zero_dur & ~single_in_window, col_eoy] = 0.0

    return df


def compute_taux_avancement_global_v2(df: pd.DataFrame, date_cloture_str: str) -> pd.DataFrame:
    df = df.copy()

    # Colonnes utilisées
    col_exec = "Date d'éxécution (produit sans session)"
    col_deb  = "Date de début"
    col_fin  = "Date de fin"

    # Parse dates
    dc  = pd.to_datetime(date_cloture_str, errors="coerce")
    deb = _to_dt(df[col_deb]) if col_deb in df.columns else pd.Series([pd.NaT] * len(df), index=df.index)
    fin = _to_dt(df[col_fin]) if col_fin in df.columns else pd.Series([pd.NaT] * len(df), index=df.index)
    exe = _to_dt(df[col_exec]) if col_exec in df.columns else pd.Series([pd.NaT] * len(df), index=df.index)

    # Origine (toujours en Series)
    if "Origine rapport" in df.columns:
        origin = df["Origine rapport"].astype(str).str.lower().str.strip()
    else:
        origin = pd.Series([""] * len(df), index=df.index)

    out_col = "Taux d'avancement global"
    df[out_col] = pd.NA  # init vide

    # Si pas de date de clôture → garder le fallback actuel
    if pd.isna(dc):
        # sans session : exécuté => 1 ; pas de date => erreur
        df.loc[origin.eq("sans_session") & exe.notna(), out_col] = 1.0
        df.loc[origin.eq("sans_session") & exe.isna(),  out_col] = "=NA()"
        # inter/intra : 0 par défaut (pas d'échelle temporelle)
        df.loc[origin.isin(["intra", "inter"]), out_col] = 0.0
        return df

    # --- SANS_SESSION (NOUVELLE RÈGLE) ---
    mask_ss = origin.eq("sans_session")
    if col_exec in df.columns:
        no_exec = mask_ss & (df[col_exec].astype(str).str.strip() == "")
        df.loc[no_exec, out_col] = "=NA()"      # vraie erreur Excel si pas de date

    # Si exe > clôture -> 0, sinon 1
    m0 = mask_ss & exe.notna() & (exe > dc)     # <<< changé : '>' (avant : '<')
    df.loc[m0, out_col] = 0.0
    m1 = mask_ss & exe.notna() & ~m0
    df.loc[m1, out_col] = 1.0

    # --- INTRA / INTER (inchangé) ---
    mask_ii = origin.isin(["intra", "inter"])
    # fin < clôture -> 1
    m_full = mask_ii & fin.notna() & (fin < dc)
    df.loc[m_full, out_col] = 1.0

    # début > clôture -> 0
    m_zero = mask_ii & deb.notna() & (deb > dc)
    df.loc[m_zero, out_col] = 0.0

    # cas fractionnel
    m_frac = mask_ii & deb.notna() & fin.notna() & ~(m_full | m_zero)
    dur_days = (fin - deb).dt.days
    ratio = ((dc - deb).dt.days / dur_days.replace(0, pd.NA)).fillna(1.0).clip(lower=0.0, upper=1.0)

    # durée nulle : si clôture >= fin -> 1, sinon 0
    zero_dur = m_frac & (dur_days == 0)
    df.loc[m_frac, out_col] = ratio
    df.loc[zero_dur & (dc >= fin), out_col] = 1.0
    df.loc[zero_dur & (dc <  fin), out_col] = 0.0

    return df



def compute_progress_and_backlog(df: pd.DataFrame,
                                 date_cloture: str,
                                 debut_exercice: str,
                                 fin_exercice: str,
                                 closure_year_hint: Optional[int]) -> pd.DataFrame:
    df = df.copy()

    # 1) Taux d'avancement global (déjà V2)
    df = compute_taux_avancement_global_v2(df, date_cloture_str=date_cloture)

    # 2) CA attendu
    ca_att = pd.to_numeric(df.get("CA attendu", 0), errors="coerce").fillna(0)

    
    # 3) CA YTD
    taux = df.get("Taux d'avancement global", 0)
    taux_num = pd.to_numeric(taux, errors="coerce").fillna(0)
    df["CA YTD"] = ca_att * taux_num
    df["CA avancement"] = df["CA YTD"]
    # df = _enforce_ytd_only_current_year(df, closure_year_hint)

    # 4) Facturation totale (robuste)
    col_fact_total = "Facturation totale"
    if col_fact_total not in df.columns:
        df[col_fact_total] = 0.0
    df[col_fact_total] = pd.to_numeric(df[col_fact_total], errors="coerce").fillna(0)
    fact_tot = df[col_fact_total]

    # 5) FAE / PCA (V2)
    diff = df["CA YTD"] - fact_tot
    df["FAE"] = diff.clip(lower=0)
    df["PCA"] = diff.clip(upper=0)

    # 6) Avancement EOY (V2) + CA EOY (backlog)
    df = compute_avancement_eoy_v2(df, date_cloture_str=date_cloture, fin_exercice_str=fin_exercice)

    # Numeric copy for multiplication (keep any =NA() text in display column)
    av_eoy_num = pd.to_numeric(df.get("Avancement EOY", 0), errors="coerce").fillna(0)
    ca_att = pd.to_numeric(df.get("CA attendu", 0), errors="coerce").fillna(0)
    df["CA EOY (backlog)"] = ca_att * av_eoy_num

    return df



def inject_excel_error_for_ss_missing_exec(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col_av = "Taux d'avancement global"
    col_exec = "Date d'éxécution (produit sans session)"

    # masque: sans_session ET date d'exécution vide / manquante
    mask = (
        df.get("Origine rapport", "").astype(str).str.lower().eq("sans_session")
        & (df.get(col_exec, "").astype(str).str.strip() == "")
    )

    # Injecter une erreur Excel vraie : =NA()
    # Important: ne pas mettre des quotes; commencer par "=" pour que ce soit une formule.
    df.loc[mask, col_av] = "=(1/0)"
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

# def _enforce_ytd_only_current_year(df: pd.DataFrame, annee_N: int | None, date_cloture: str | None = None) -> pd.DataFrame:
#     """
#     Ensure CA YTD only applies to the current year N.
#     If per-line 'Année' exists, use it; otherwise approximate with Facturation Y > 0.
#     date_cloture may be 'DD/MM/YYYY' or 'YYYY-MM-DD'; it's not strictly required for the guard.
#     """
#     out = df.copy()
#     if "CA YTD" in out.columns:
#         if "Année" in out.columns and annee_N is not None:
#             is_N = pd.to_numeric(out["Année"], errors="coerce").astype("Int64") == annee_N
#             out.loc[~is_N, "CA YTD"] = 0.0
#         elif "Facturation Y" in out.columns:
#             is_N = out["Facturation Y"].fillna(0) > 0
#             out.loc[~is_N, "CA YTD"] = 0.0
#     return out
