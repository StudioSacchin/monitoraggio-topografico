# monitoraggio.py
# Web app per monitoraggio topografico
# import file txt generato da file di monitoraggio Studio Sacchin
# applicazione creata da Luis Saggiomo con l'ausilio di ChatGPT5
# Studio Sacchin

import io
import os
import math
import json
import datetime as dt
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:
    alt = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

# ---- Plotly theming helper ----
def _auto_date_ticks(dates):
    """Ritorna impostazioni asse X con ~8-12 etichette regolari.
    - Sceglie cadenza: settimanale (D7/14/21), mensile (M1/2/3), semestrale (M6), annuale (M12/24)
    - Restituisce tickvals/ticktext in ITA per compatibilità anche con export PDF (Kaleido)
    """
    import math
    import pandas as pd
    from datetime import datetime, timedelta
    if len(dates) == 0:
        return {'type':'date', 'tickangle': -30}
    s = pd.to_datetime(pd.Series(dates))
    s = s.dropna()
    if s.empty:
        return {'type':'date', 'tickangle': -30}
    dmin, dmax = s.min(), s.max()
    # Estensione
    months = (dmax.year - dmin.year) * 12 + (dmax.month - dmin.month) + 1
    days = (dmax.normalize() - dmin.normalize()).days + 1

    # Candidati mensili e superiori
    cand = [
        ('M1',  '%b %Y'),
        ('M2',  '%b %Y'),
        ('M3',  '%b %Y'),
        ('M6',  '%b %Y'),
        ('M12', '%Y'),
        ('M24', '%Y'),
    ]
    best = None; best_diff = 1e9
    for step, fmt in cand:
        m = int(step[1:])
        count = math.ceil(months / m)
        diff = abs(count - 10)
        if 8 <= count <= 12:
            best = (step, fmt, count)
            break
        if diff < best_diff:
            best = (step, fmt, count)
            best_diff = diff
    step, fmt, _ = best

    # Periodi brevi -> prova settimanale
    if months <= 6:
        day_cand = [('D7','%d/%m/%Y'), ('D14','%d/%m/%Y'), ('D21','%d/%m/%Y')]
        best_d = None; best_diff = 1e9
        for dstep, dfmt in day_cand:
            d = int(dstep[1:])
            count = math.ceil(days / d)
            diff = abs(count - 10)
            if 8 <= count <= 12:
                best_d = (dstep, dfmt)
                break
            if diff < best_diff:
                best_d = (dstep, dfmt)
                best_diff = diff
        step, fmt = best_d

    # Helper per somma mesi
    def add_months(dt, m):
        y = dt.year + (dt.month - 1 + m) // 12
        mo = (dt.month - 1 + m) % 12 + 1
        day = min(dt.day, [31,29 if y%4==0 and (y%100!=0 or y%400==0) else 28,31,30,31,30,31,31,30,31,30,31][mo-1])
        return dt.replace(year=y, month=mo, day=day, hour=0, minute=0, second=0, microsecond=0)

    # Allineamento tick0
    if step.startswith('M'):
        mm = int(step[1:])
        tick0 = dmin.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if mm in (2,3,6,12,24):
            base = ((tick0.month-1)//mm)*mm + 1
            tick0 = tick0.replace(month=base)
    else:
        tick0 = dmin.replace(hour=0, minute=0, second=0, microsecond=0)

    # Genera tickvals regolari
    tickvals = []
    cur = tick0
    while cur <= dmax + pd.Timedelta(days=1):
        tickvals.append(cur.to_pydatetime() if hasattr(cur, 'to_pydatetime') else cur)
        if step.startswith('M'):
            cur = add_months(cur, int(step[1:]))
        else:
            cur = cur + pd.Timedelta(days=int(step[1:]))

    # Etichette italiane (abbreviate per mesi)
    mesi_it = ['gen','feb','mar','apr','mag','giu','lug','ago','set','ott','nov','dic']
    def label_it(dt):
        if step.startswith('M'):
            mm = int(step[1:])
            if mm >= 12:
                return f"{dt.year}"
            # Mese abbreviato + anno
            return f"{mesi_it[dt.month-1]} {dt.year}"
        else:
            # formato giorno/mese/anno
            return dt.strftime("%d/%m/%Y")

    ticktext = [label_it(pd.to_datetime(v)) for v in tickvals]

    return {'type': 'date', 'tickmode': 'array', 'tickvals': tickvals, 'ticktext': ticktext, 'tickangle': -30}

def _apply_plotly_theme(fig, *, for_pdf: bool = False, n_points: int | None = None):
    # Applica uno stile professionale, chiaro e leggibile ai grafici Plotly,
    # con font più grandi, griglie leggere e margini adeguati anche in PDF.
    import plotly.io as pio
    # Base template pulito
    fig.update_layout(template="plotly_white")

    # Dimensionamento caratteri (raddoppiati circa)
    base = 22 if not for_pdf else 26
    title_sz = 28 if not for_pdf else 32
    ax_title_sz = base + 4
    tick_sz = base

    # Margini e legende
    fig.update_layout(
        font=dict(size=tick_sz),
        title=dict(font=dict(size=title_sz), x=0.02, xanchor="left"),
        margin=dict(l=60, r=30, t=80, b=60),
        legend=dict(font=dict(size=base), bgcolor="rgba(255,255,255,0.5)"),
    )

    # Assi con standoff per evitare sovrapposizioni con le etichette
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)",
                     title_standoff=22, tickfont=dict(size=tick_sz),
                     title_font=dict(size=ax_title_sz),
                     automargin=True, ticks="outside", ticklen=7)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)",
                     title_standoff=22, tickfont=dict(size=tick_sz),
                     title_font=dict(size=ax_title_sz),
                     automargin=True, ticks="outside", ticklen=7)

    # Se noto il numero di punti sul tempo, ruota leggermente le date quando sono molte
    if n_points is not None and n_points >= 8:
        fig.update_xaxes(tickangle=-35)

try:
    import folium
    from streamlit_folium import st_folium
except Exception:
    folium = None
    st_folium = None

try:
    from pyproj import Transformer
except Exception:
    Transformer = None

# -----------------------------
# Parser utilities
# -----------------------------

def _detect_point_meta_cols(header_row: pd.Series) -> Dict[str, int]:
    meta = {}
    for idx, name in enumerate(header_row):
        n = str(name).strip().lower()
        if n in {"prima_misurata","misurata_inizio","misurata_prima","first_epoch","first_misurata"}:
            meta['prima_misurata'] = idx
        if n in {"pos0","posizione 0","pos. 0","posizione0"}:
            meta['pos0'] = idx
        if n in {"descrizione","descr","desc"}:
            meta['descrizione'] = idx
        if n in {"codice","id","punto","nome"}:
            meta['codice'] = idx
    return meta


def _parse_esempio_matrix_style(xl: pd.ExcelFile, sheet: str = None) -> Tuple[pd.DataFrame, Dict]:
    if sheet is None:
        sheet = xl.sheet_names[0]
    raw = xl.parse(sheet_name=sheet, header=None)

    header_row_idx = None
    for r in range(0, min(20, len(raw))):
        row_vals = raw.iloc[r].astype(str).str.lower().tolist()
        if any(v.startswith('data') for v in row_vals):
            header_row_idx = r
            break
    if header_row_idx is None:
        raise ValueError("Formato matrix: riga header con 'data_*' non trovata")

    header = raw.iloc[header_row_idx]
    point_meta_idx = _detect_point_meta_cols(header)

    # trova blocchi per ogni epoca (data, X, Y, Z)
    groups = []
    rest = [i for i in range(len(header)) if i not in point_meta_idx.values()]
    for i in range(0, len(rest), 4):
        block = rest[i:i+4]
        if len(block) < 4:
            break
        groups.append(block)

    # trova righe dati e righe meta opzionali
    data_start = header_row_idx + 1
    data_end = len(raw)

    # mappa misurata → data
    date_to_mis = {}
    records = []

    for block_cols in groups:
        dcol, xcol, ycol, zcol = block_cols
        dttm = pd.to_datetime(raw.iloc[data_start:data_end, dcol], errors='coerce')
        if dttm.isna().all():
            continue
        # prima data valida
        dttm = dttm.iloc[0]
        date_to_mis[pd.to_datetime(dttm)] = len(date_to_mis)

    if not date_to_mis:
        raise ValueError("Nessuna epoca valida trovata nel formato matrix")

    meta_rows = {'temperatura': None, 'pressione': None}

    for block_cols in groups:
        dcol, xcol, ycol, zcol = block_cols
        dttm = raw.iloc[data_start:data_end, dcol].dropna()
        if dttm.empty:
            continue
        dttm = pd.to_datetime(dttm.iloc[0], errors='coerce')
        mis = date_to_mis[pd.to_datetime(dttm)]

        # range righe punto
        start_r, end_r = data_start, data_end

        block = pd.DataFrame({
            'codice': raw.iloc[start_r:end_r, point_meta_idx.get('codice', 0)].values,
            'descrizione': raw.iloc[start_r:end_r, point_meta_idx.get('descrizione', 1)].values,
            'X': raw.iloc[start_r:end_r, xcol].values,
            'Y': raw.iloc[start_r:end_r, ycol].values,
            'Z': raw.iloc[start_r:end_r, zcol].values,
        })
        if 'prima_misurata' in point_meta_idx:
            block['prima_misurata'] = pd.to_numeric(raw.iloc[start_r:end_r, point_meta_idx['prima_misurata']].values, errors='coerce')
        else:
            block['prima_misurata'] = np.nan
        block['data'] = pd.to_datetime(dttm)
        block['misurata'] = mis
        def _get_meta_val_row(ridx, col_idx):
            try:
                return raw.iloc[ridx, col_idx]
            except Exception:
                return None
        block['temperatura'] = pd.to_numeric(_get_meta_val_row(meta_rows['temperatura'], xcol), errors='coerce')
        block['pressione'] = pd.to_numeric(_get_meta_val_row(meta_rows['pressione'], xcol), errors='coerce')
        records.append(block)

    df_long = pd.concat(records, ignore_index=True)
    df_long['codice'] = df_long['codice'].astype(str).str.strip()
    df_long['descrizione'] = df_long['descrizione'].astype(str).str.strip()
    for c in ['X','Y','Z']:
        df_long[c] = pd.to_numeric(df_long[c], errors='coerce')

    meta = {
        'n_punti': df_long['codice'].nunique(),
        'n_epoche': df_long['data'].nunique(),
        'date': sorted(df_long['data'].dropna().unique()),
        'sheet': sheet,
        'format': 'matrix',
        'title': None,
    }
    return df_long, meta


def _parse_long_format(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    rename_map = {
        'code': 'codice', 'id': 'codice', 'punto': 'codice', 'nome': 'codice',
        'desc': 'descrizione', 'descr': 'descrizione', 'descrizione': 'descrizione',
        'date': 'data', 'epoch': 'data', 'misurata': 'misurata',
        'x':'X', 'y':'Y', 'z':'Z'
    }
    df1 = df.rename(columns={c: rename_map.get(str(c).strip().lower(), c) for c in df.columns})
    needed = {'codice','data','X','Y'}
    if not needed.issubset(set(df1.columns)):
        raise ValueError("Formato 'long' non valido: mancano colonne necessarie")

    df1['codice'] = df1['codice'].astype(str).str.strip()
    if 'descrizione' not in df1:
        df1['descrizione'] = ''
    if 'misurata' not in df1:
        # mappa data → indice ordine
        ord_map = {d: i for i, d in enumerate(sorted(pd.to_datetime(df1['data'].dropna().unique())))}
        df1['misurata'] = pd.to_datetime(df1['data']).map(ord_map)

    for c in ['X','Y','Z']:
        if c in df1:
            df1[c] = pd.to_numeric(df1[c], errors='coerce')

    meta = {
        'n_punti': df1['codice'].nunique(),
        'n_epoche': df1['misurata'].nunique(),
        'date': sorted(pd.to_datetime(df1['data'].dropna().unique())),
        'sheet': None,
        'format': 'long',
        'title': None,
    }
    return df1, meta


def _parse_semicolon_wide_from_bytes(data: bytes) -> Tuple[pd.DataFrame, Dict]:
    text = data.decode('utf-8', errors='ignore')
    # rimuovi righe completamente vuote
    lines = [ln for ln in (l.strip() for l in text.splitlines()) if ln != '']
    if len(lines) < 2:
        raise ValueError("File troppo corto per il formato ';' con titolo + header")

    title_line = lines[0].strip()
    # titolo può includere anche il sistema di coordinate es.: "BAITA;UTM32" oppure "LAVORO;Sistema Locale"
    parts = [p.strip() for p in title_line.split(';') if p.strip() != '']
    title = parts[0] if parts else None
    coord_raw = parts[1].lower() if len(parts) > 1 else None
    if coord_raw in {'utm32', 'utm 32'}:
        coord_sys = 'UTM32'
    elif coord_raw in {'utm33', 'utm 33'}:
        coord_sys = 'UTM33'
    elif coord_raw in {'sistema locale', 'locale'}:
        coord_sys = 'Sistema Locale'
    else:
        coord_sys = None  # manterremo default più avanti

    csv_text = '\n'.join(lines[1:])

    df0 = pd.read_csv(io.StringIO(csv_text), sep=';', engine='python')
    # elimina colonne vuote create da ';' finali
    df0 = df0.loc[:, [c for c in df0.columns if not str(c).strip().lower().startswith('unnamed')]]

    # trova colonne meta (case-insensitive)
    cols_lower = {str(c).strip().lower(): c for c in df0.columns}
    if 'codice' not in cols_lower or 'descrizione' not in cols_lower:
        raise ValueError("Header non valido: attese colonne 'Codice' e 'Descrizione' nella seconda riga")
    c_code = cols_lower['codice']
    c_desc = cols_lower['descrizione']
    c_tipo = cols_lower.get('tipologia', None)

    # colonne successive: gruppi da 4 (data, X, Y, Z) per epoca
    rest = [c for c in df0.columns if c not in (c_code, c_desc) and (c_tipo is None or c != c_tipo)]
    groups = []
    for i in range(0, len(rest), 4):
        block = rest[i:i+4]
        if len(block) < 4:
            break
        groups.append(tuple(block))  # (data, X, Y, Z)

    records = []
    for _, row in df0.iterrows():
        code = str(row[c_code]).strip()
        desc = str(row[c_desc]).strip()
        tipo = str(row[c_tipo]).strip() if c_tipo is not None else None
        for gi, (dcol, xcol, ycol, zcol) in enumerate(groups, start=0):
            dval = row[dcol]
            x = pd.to_numeric(row[xcol], errors='coerce')
            y = pd.to_numeric(row[ycol], errors='coerce')
            z = pd.to_numeric(row[zcol], errors='coerce')
            if pd.isna(dval) and pd.isna(x) and pd.isna(y) and pd.isna(z):
                continue
            # data: può essere numero stile Excel (anche float) o stringa
            date = pd.NaT
            if not pd.isna(dval):
                try:
                    if isinstance(dval, (int, float)) and not pd.isna(dval):
                        # Excel serial date (origin 1899-12-30) – conserva anche la frazione di giorno
                        days = float(dval)
                        date = pd.to_datetime('1899-12-30') + pd.to_timedelta(days, unit='D')
                    else:
                        date = pd.to_datetime(dval, dayfirst=True, errors='coerce')
                except Exception:
                    date = pd.to_datetime(dval, dayfirst=True, errors='coerce')
            records.append({'codice': code, 'descrizione': desc, 'tipologia': tipo, 'data': date, 'misurata': gi, 'X': x, 'Y': y, 'Z': z})

    df_long = pd.DataFrame.from_records(records)
    if df_long.empty:
        raise ValueError("Nessun dato interpretato dal formato ';'")

    meta = {
        'n_punti': df_long['codice'].nunique(),
        'n_epoche': df_long['misurata'].nunique(),
        'date': sorted(df_long['data'].dropna().unique()),
        'sheet': None,
        'format': 'semicolon_wide',
        'title': title,
        'coord_sys': coord_sys,
    }
    return df_long, meta


# -----------------------------
# Ingest
# -----------------------------

def parse_upload(file: io.BytesIO) -> Tuple[pd.DataFrame, Dict]:
    name = getattr(file, 'name', 'upload')
    suffix = os.path.splitext(name)[-1].lower()

    if suffix in ['.xlsx', '.xls']:
        xl = pd.ExcelFile(file)
        try:
            df_long, meta = _parse_esempio_matrix_style(xl)
            return df_long, meta
        except Exception:
            pass
        df0 = xl.parse(xl.sheet_names[0])
        return _parse_long_format(df0)

    elif suffix in ['.csv', '.txt']:
        # leggi bytes una volta sola, prova prima nuovo formato ';' con titolo
        data = file.read() if hasattr(file, 'read') else file
        try:
            return _parse_semicolon_wide_from_bytes(data)
        except Exception:
            # fallback: tenta come formato lungo generico
            try:
                df0 = pd.read_csv(io.BytesIO(data))
                return _parse_long_format(df0)
            except Exception:
                # ultimo tentativo: separatore ';' senza titolo
                df0 = pd.read_csv(io.BytesIO(data), sep=';')
                return _parse_long_format(df0)

    else:
        try:
            df0 = pd.read_excel(file)
            return _parse_long_format(df0)
        except Exception as e:
            raise ValueError(f"Formato file non supportato: {suffix}. Errore: {e}")


# -----------------------------
# Calcoli spostamenti
# -----------------------------

def compute_displacements(df_long: pd.DataFrame, ref_epoch: int) -> pd.DataFrame:
    base = df_long[df_long['misurata'] == ref_epoch][['codice','X','Y','Z']].rename(columns={'X':'X0','Y':'Y0','Z':'Z0'})
    merged = df_long.merge(base, on='codice', how='left')
    merged['dX'] = merged['X'] - merged['X0']
    merged['dY'] = merged['Y'] - merged['Y0']
    merged['dZ'] = merged['Z'] - merged['Z0']
    merged['d2D'] = np.sqrt(merged['dX']**2 + merged['dY']**2)
    merged['d3D'] = np.sqrt(merged['d2D']**2 + (merged['dZ']**2))
    return merged

# -----------------------------
# Calcolo degli spostamenti dalla prima misurata disponibile
# -----------------------------

@st.cache_data
def _point_bases(df_long: pd.DataFrame) -> dict:
    """
    Per ogni 'codice' ritorna il primo campione (misurata più vecchia)
    con X,Y,Z tutti valorizzati: { codice: {'mis_base': int, 'date_base': ts,
                                            'X0': float, 'Y0': float, 'Z0': float} }
    """
    valid = df_long[df_long[['X','Y','Z']].notna().all(axis=1)].copy()
    valid = valid.sort_values(['codice','misurata'])
    first = (valid.groupby('codice', as_index=False)
                  .first()[['codice','misurata','data','X','Y','Z']]
                  .rename(columns={'misurata':'mis_base','data':'date_base','X':'X0','Y':'Y0','Z':'Z0'}))
    return {str(r['codice']): {'mis_base': int(r['mis_base']),
                               'date_base': pd.to_datetime(r['date_base']),
                               'X0': float(r['X0']), 'Y0': float(r['Y0']), 'Z0': float(r['Z0'])}
            for _, r in first.iterrows()}
            
# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Monitoraggio Topografico", layout="wide")

with st.sidebar:
    up = st.file_uploader("Carica file (.txt)", type=["txt"], accept_multiple_files=False)


if up is None:
    st.title("📐 STUDIO SACCHIN - MONITORAGGIO TOPOGRAFICO")
    st.info("🔼 Carica un file per iniziare.")
    st.stop()

# Parsing
try:
    df_long, meta = parse_upload(up)
    bases_map = _point_bases(df_long)
except Exception as e:
    st.error(f"Errore nel parsing: {e}")
    st.stop()

#Titolo dinamico della paginain base al file
titolo_importato = meta.get("title", "Monitoraggio senza nome")

# Variante 1: tutto su una riga
# st.title(f"📐 STUDIO SACCHIN - MONITORAGGIO TOPOGRAFICO – {titolo_importato}")

# Variante 2: titolo fisso sopra e sotto quello importato
st.title("📐 STUDIO SACCHIN - MONITORAGGIO TOPOGRAFICO")
st.subheader(titolo_importato)

# garantisci presenza colonna tipologia
if 'tipologia' not in df_long.columns:
    df_long['tipologia'] = np.nan

# Info base
min_d, max_d = pd.to_datetime(df_long['data']).min(), pd.to_datetime(df_long['data']).max()
mis_sorted = sorted({int(m): d for m,d in df_long[['misurata','data']].dropna().drop_duplicates().values}.items())
mis_to_date = {int(m): pd.to_datetime(d) for m, d in mis_sorted}
all_mis = sorted(mis_to_date.keys())

# State iniziale
if 'selected_code' not in st.session_state:
    st.session_state.selected_code = None

# Calcoli base: Riferimento default: prima misurata disponibile
ref_m = all_mis[0] if all_mis else None
if ref_m is None:
    st.error("Nessuna misurata disponibile.")
    st.stop()

# Calcola tutti gli spostamenti rispetto alla prima misurata
res = compute_displacements(df_long, int(ref_m))
res['misurata_ref'] = ref_m

# Calcola i massimi reali da usare come limiti per gli slider
max_d2d = float((res['d2D'].abs().max(skipna=True)) or 0.0)
max_d3d = float((res['d3D'].abs().max(skipna=True)) or 0.0)

# Sidebar: solo Punti + soglie d2D/d3D (con slider sincronizzati)
with st.sidebar:
    st.header("Filtri")

    # Inizializza session_state per le soglie
    if "d2d_num" not in st.session_state:
        st.session_state["d2d_num"] = 0.0
    if "d2d_sld" not in st.session_state:
        st.session_state["d2d_sld"] = st.session_state["d2d_num"]

    if "d3d_num" not in st.session_state:
        st.session_state["d3d_num"] = 0.0
    if "d3d_sld" not in st.session_state:
        st.session_state["d3d_sld"] = st.session_state["d3d_num"]

    # Callbacks per sincronizzazione...
    def _sync_d2d_from_num():
        v = float(st.session_state["d2d_num"])
        v = max(0.0, min(v, max_d2d))
        st.session_state["d2d_num"] = v
        st.session_state["d2d_sld"] = v

    def _sync_d2d_from_sld():
        st.session_state["d2d_num"] = float(st.session_state["d2d_sld"])

    def _sync_d3d_from_num():
        v = float(st.session_state["d3d_num"])
        v = max(0.0, min(v, max_d3d))
        st.session_state["d3d_num"] = v
        st.session_state["d3d_sld"] = v

    def _sync_d3d_from_sld():
        st.session_state["d3d_num"] = float(st.session_state["d3d_sld"])

    # Soglia d2D
    st.number_input(
        "Soglia 2D (m)",
        min_value=0.0, max_value=max_d2d,
        step=0.001, format="%.3f",
        key="d2d_num", on_change=_sync_d2d_from_num,
    )
    st.slider(
        " ", min_value=0.0, max_value=max_d2d,
        step=0.01, key="d2d_sld",
        on_change=_sync_d2d_from_sld
    )

    st.markdown("---")

    # Soglia d3D
    st.number_input(
        "Soglia 3D (m)",
        min_value=0.0, max_value=max_d3d,
        step=0.001, format="%.3f",
        key="d3d_num", on_change=_sync_d3d_from_num,
    )
    st.slider(
        "  ", min_value=0.0, max_value=max_d3d,
        step=0.01, key="d3d_sld",
        on_change=_sync_d3d_from_sld
    )

    if st.button("Reset soglie"):
        for k in ("d2d_num","d2d_sld","d3d_num","d3d_sld"):
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

# Nessun filtro su misurate o punti: usiamo tutto il periodo e tutti i codici
res_f = res.copy()

# Applica soglie d2D / d3D (sul valore assoluto)
d2d_thresh = float(st.session_state.get("d2d_num", 0.0) or 0.0)
d3d_thresh = float(st.session_state.get("d3d_num", 0.0) or 0.0)

if d2d_thresh > 0:
    res_f = res_f[res_f['d2D'].abs() >= d2d_thresh]
if d3d_thresh > 0:
    res_f = res_f[res_f['d3D'].abs() >= d3d_thresh]

# KPI veloci
def _fmt_max(series: pd.Series) -> str:
    val = pd.to_numeric(series, errors="coerce").abs().max(skipna=True)
    return "-" if pd.isna(val) else f"{val:.3f}"
    
col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
col_k1.metric("N. punti", int(res_f['codice'].nunique()))
col_k2.metric("N. misurate", int(res_f['misurata'].nunique()))
col_k3.metric("Max d2D (m)", _fmt_max(res_f['d2D']) if not res_f.empty else "-")
col_k4.metric("Max |dZ| (m)", _fmt_max(res_f['dZ']) if not res_f.empty else "-")
col_k5.metric("Max d3D (m)", _fmt_max(res_f['d3D']) if not res_f.empty else "-")

st.divider()

st.subheader("Tabella punti – confronto tra misurate")
col_ref, col_cmp = st.columns(2)
with col_ref:
    ref_tab = st.selectbox(
        "Misurata di riferimento",
        options=all_mis, index=0, key='tbl_ref',
        format_func=lambda m: f"Misurata {m} – {mis_to_date[m].date()}"
    )
with col_cmp:
    cmp_options = [m for m in all_mis if m >= ref_tab]
    default_idx = len(cmp_options)-1 if len(cmp_options) > 0 else 0
    cmp_tab = st.selectbox(
        "Misurata di confronto",
        options=cmp_options, index=default_idx, key='tbl_cmp',
        format_func=lambda m: f"Misurata {m} – {mis_to_date[m].date()}"
    )

ref_df = res[res['misurata']==ref_tab][['codice','descrizione','X','Y','Z']].rename(
    columns={'X':'X_ref','Y':'Y_ref','Z':'Z_ref'}
)
cmp_df = res[res['misurata']==cmp_tab][['codice','X','Y','Z']].rename(
    columns={'X':'X_cmp','Y':'Y_cmp','Z':'Z_cmp'}
)
comp = ref_df.merge(cmp_df, on='codice', how='inner')
comp['dX'] = comp['X_cmp'] - comp['X_ref']
comp['dY'] = comp['Y_cmp'] - comp['Y_ref']
comp['dZ'] = comp['Z_cmp'] - comp['Z_ref']
comp['vettore_2D'] = np.sqrt(comp['dX']**2 + comp['dY']**2)
comp['vettore_3D'] = np.sqrt(comp['vettore_2D']**2 + (comp['dZ']**2))

# === UTIL per PDF e tabella riepilogativa ===
@st.cache_data
def _build_summary_for_pdf(df_long: pd.DataFrame, cmp_mis: int, mis_to_date: dict | None = None) -> pd.DataFrame:
    """
    Riepilogo per PDF:
    - include SOLO i punti con X,Y,Z presenti nella misurata di confronto 'cmp_mis'
    - per ogni codice prende come riferimento la PRIMA misurata dove X,Y,Z sono tutti validi
    - calcola dX, dY, dZ, dist.2D, dist.3D
    - compila 'misurata di riferimento' come 'mis. N del dd/mm/yyyy'
    """
    # Punti presenti nella mis. di confronto con coordinate valide
    cmp_df = (
        df_long[(df_long['misurata'] == cmp_mis) & df_long[['X','Y','Z']].notna().all(axis=1)]
        [['codice','descrizione','X','Y','Z','misurata','data']]
        .rename(columns={'X':'X_cmp','Y':'Y_cmp','Z':'Z_cmp','data':'data_cmp'})
    )

    # Prima misurata valida per ciascun codice
    valid_all = df_long[df_long[['X','Y','Z']].notna().all(axis=1)]
    first_rows = (
        valid_all.sort_values(['codice','misurata'])
                 .groupby('codice', as_index=False)
                 .first()[['codice','X','Y','Z','misurata','data']]
                 .rename(columns={'X':'X_ref','Y':'Y_ref','Z':'Z_ref','misurata':'mis_ref','data':'data_ref'})
    )

    # Tieni solo codici presenti nella mis. di confronto
    merged = cmp_df.merge(first_rows, on='codice', how='inner')

    # Delta e distanze
    merged['dX'] = merged['X_cmp'] - merged['X_ref']
    merged['dY'] = merged['Y_cmp'] - merged['Y_ref']
    merged['dZ'] = merged['Z_cmp'] - merged['Z_ref']
    merged['dist.2D'] = np.sqrt(merged['dX']**2 + merged['dY']**2)
    merged['dist.3D'] = np.sqrt(merged['dist.2D']**2 + (merged['dZ']**2))

    # "misurata di riferimento"
    def _fmt_ref(row):
        misn = int(row['mis_ref'])
        if mis_to_date and misn in mis_to_date:
            d = pd.to_datetime(mis_to_date[misn])
            return f"mis. {misn} del {d.strftime('%d/%m/%Y')}"
        if pd.notna(row.get('data_ref', None)):
            try:
                d = pd.to_datetime(row['data_ref'], unit='D', origin='1899-12-30')
            except Exception:
                d = pd.to_datetime(row['data_ref'])
            return f"mis. {misn} del {d.strftime('%d/%m/%Y')}"
        return f"mis. {misn}"

    merged['misurata di riferimento'] = merged.apply(_fmt_ref, axis=1)

    out = merged[['codice','descrizione','misurata di riferimento','dX','dY','dZ','dist.2D','dist.3D']].copy()
    for c in ['dX','dY','dZ','dist.2D','dist.3D']:
        out[c] = pd.to_numeric(out[c], errors='coerce').round(3)

    return out.sort_values('codice').reset_index(drop=True)

# @st.cache_data
def _pdf_from_summary_table(project_title: str,
                            df_summary: pd.DataFrame,
                            cmp_mis: int,
                            cmp_date=None) -> bytes:
    """
    PDF con:
    - Titolo grande centrato
    - Sottotitolo (titolo progetto) centrato
    - Riga 'Misurata N del ...' in grassetto, prima della tabella
    - Tabella riepilogativa
    - Piè di pagina con contatti + link mailto/URL
    """
    from io import BytesIO
    buffer = BytesIO()
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.pdfbase.pdfmetrics import stringWidth

        # Documento PDF
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            leftMargin=24, rightMargin=24, topMargin=28, bottomMargin=36
        )

        # Stili base
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('TitleCenter', parent=styles['Title'], alignment=1, fontSize=20, leading=24)
        subtitle_style = ParagraphStyle('H2Center', parent=styles['Heading2'], alignment=1, fontSize=14, leading=18)
        bold_center = ParagraphStyle('BoldCenter', parent=styles['Heading4'], alignment=1, fontSize=10, leading=12)

        # Stile celle (wrapping)
        cell_style = ParagraphStyle('cell', parent=styles['BodyText'],
                                    fontName='Helvetica', fontSize=8, leading=10,
                                    spaceAfter=0, spaceBefore=0)

        # Data estesa italiana
        mesi = ['gennaio','febbraio','marzo','aprile','maggio','giugno',
                'luglio','agosto','settembre','ottobre','novembre','dicembre']
        if cmp_date is not None:
            d = pd.to_datetime(cmp_date)
            cmp_date_ext = f"{d.day} {mesi[d.month-1]} {d.year}"
        else:
            cmp_date_ext = ""

        # Footer
        FOOTER_TEXT = ("Studio Sacchin - Studio Tecnico Associato - via Postgranz Strasse 16 - "
                       "39012 Meran/o (BZ) - Tel. 0473 445058 - "
                       "info@studiosacchin.it - www.studiosacchin.it")

        def _footer(canvas, doc):
            canvas.saveState()
            w, h = doc.pagesize
            y = 18
            txt_width = stringWidth(FOOTER_TEXT, 'Helvetica', 8)
            x = (w - txt_width) / 2.0
            canvas.setFont('Helvetica', 8)
            canvas.drawString(x, y, FOOTER_TEXT)

            # link area calcolata
            prefix = "Studio Sacchin - Studio Tecnico Associato - via Postgranz Strasse 16 - 39012 Meran/o (BZ) - Tel. 0473 445058 - "
            mail = "info@studiosacchin.it"
            mid = " - "
            url = "www.studiosacchin.it"
            x0 = (w - stringWidth(FOOTER_TEXT, 'Helvetica', 8)) / 2.0
            x_mail = x0 + stringWidth(prefix, 'Helvetica', 8)
            x_url  = x_mail + stringWidth(mail + mid, 'Helvetica', 8)

            canvas.linkURL("mailto:info@studiosacchin.it",
                           (x_mail, y-1, x_mail + stringWidth(mail, 'Helvetica', 8), y+9),
                           relative=0, thickness=0, color=colors.blue)
            canvas.linkURL("https://www.studiosacchin.it",
                           (x_url, y-1, x_url + stringWidth(url, 'Helvetica', 8), y+9),
                           relative=0, thickness=0, color=colors.blue)
            canvas.restoreState()

        # Story
        story = []
        story.append(Paragraph("STUDIO SACCHIN - MONITORAGGIO TOPOGRAFICO", title_style))
        if project_title:
            story.append(Paragraph(str(project_title), subtitle_style))
        story.append(Spacer(1, 8))

        if cmp_date_ext:
            story.append(Paragraph(f"<b>Misurata {cmp_mis} del {cmp_date_ext}</b>", bold_center))
            story.append(Spacer(1, 6))

        # Tabella
        col_widths = [55, 120, 125, 50, 50, 50, 45, 45]
        headers = ["codice","descrizione","misurata di riferimento","dX","dY","dZ","dist.2D","dist.3D"]

        rows = []
        for _, r in df_summary.iterrows():
            c0 = Paragraph(str(r["codice"]), cell_style)
            c1 = Paragraph(str(r["descrizione"]), cell_style)
            c2 = Paragraph(str(r["misurata di riferimento"]), cell_style)
            c3 = f'{r["dX"]}'
            c4 = f'{r["dY"]}'
            c5 = f'{r["dZ"]}'
            c6 = f'{r["dist.2D"]}'
            c7 = f'{r["dist.3D"]}'
            rows.append([c0,c1,c2,c3,c4,c5,c6,c7])

        data = [headers] + rows

        tbl = Table(data, colWidths=col_widths, repeatRows=1, hAlign='LEFT')
        tbl.setStyle(TableStyle([
            ('FONT', (0,0), (-1,0), 'Helvetica-Bold', 9),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f0f0f0")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#333333")),
            ('ALIGN', (3,1), (-1,-1), 'RIGHT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 2),
            ('RIGHTPADDING', (0,0), (-1,-1), 2),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('FONTSIZE', (0,1), (-1,-1), 8),
            ('INNERGRID', (0,0), (-1,-1), 0.25, colors.HexColor("#cccccc")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#999999")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#fbfbfb")]),
        ]))
        story.append(tbl)

        # Build con footer
        doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
        buffer.seek(0)
        return buffer.getvalue()

    except Exception as e:
        st.error(f"Per la stampa PDF serve il pacchetto 'reportlab'. Errore: {e}")
        return b""

def _build_point_figures(res: pd.DataFrame, meta: dict, point_code: str, misurate_sel: list, bases_map: dict):
    import plotly.express as px
    import plotly.graph_objects as go

    pt = res[res['codice'].astype(str) == str(point_code)].sort_values('misurata')
    if misurate_sel:
        pt = pt[pt['misurata'].isin(misurate_sel)].copy()
    if pt.empty:
        return None, None

    # Base = prima misurata valida del punto
    b = bases_map.get(str(point_code))
    if b is None:
        base_x = pt.iloc[0]['X']; base_y = pt.iloc[0]['Y']; base_z = pt.iloc[0].get('Z', np.nan)
    else:
        base_x, base_y, base_z = b['X0'], b['Y0'], b['Z0']

    pt['cum_dX'] = pt['X'] - base_x
    pt['cum_dY'] = pt['Y'] - base_y
    pt['cum_dZ'] = pt['Z'] - base_z

    # Grafico 1: Spostamenti nel piano
    fig_vec = go.Figure()
    palette = px.colors.qualitative.Plotly
    pt_r = pt.reset_index(drop=True)
    for i, row in pt_r.iterrows():
        label = f"Misurata {int(row['misurata'])} – {pd.to_datetime(row['data']).date()}"
        if i > 0:
            prev = pt_r.loc[i-1]
            fig_vec.add_trace(go.Scatter(
                x=[prev['cum_dX'], row['cum_dX']],
                y=[prev['cum_dY'], row['cum_dY']],
                mode='lines', line=dict(width=2),
                hoverinfo='skip', showlegend=False
            ))
        fig_vec.add_trace(go.Scatter(
            x=[row['cum_dX']], y=[row['cum_dY']],
            mode='markers',
            marker=dict(size=12),
            name=label,
            hovertemplate=("ΔX: %{x:.3f} m<br>ΔY: %{y:.3f} m<br>"+f"{label}<extra></extra>")
        ))
    fig_vec.update_layout(
        title=f"Spostamenti nel piano – punto {point_code}",
        xaxis_title="ΔX (m)", yaxis_title="ΔY (m)",
        legend=dict(orientation='v', x=0.01, y=0.99)
    )
    fig_vec.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_vec.update_xaxes(constrain='domain')
    _apply_plotly_theme(fig_vec, for_pdf=False, n_points=len(pt_r))

    # Grafico 2: Velocità/Componenti cumulate nel tempo
    # Garantiamo dtype datetime per l'asse X per evitare numeri tipo 1.64e18
    pt = pt.copy()
    try:
        pt['data'] = pd.to_datetime(pt['data'], errors='coerce')
        # rimuove eventuale timezone per compatibilità export
        if hasattr(pt['data'].dtype, 'tz') and pt['data'].dtype.tz is not None:
            pt['data'] = pt['data'].dt.tz_convert(None)
        # array di datetime Python puro (niente ns) per una resa corretta in PDF
        xdates = np.array(np.array(pt['data'].dt.to_pydatetime()))
    except Exception:
        xdates = np.array(pd.to_datetime(np.array(pt['data'], errors='coerce').dt.to_pydatetime()))

    coord_sys = meta.get('coord_sys', None)
    if coord_sys in ('UTM32','UTM33','UTM'):
        lblX, lblY, lblZ = "Est","Nord","Quota"
    else:
        lblX, lblY, lblZ = "X","Y","Z"
    fig_vel = go.Figure()
    fig_vel.add_trace(go.Scatter(x=xdates, y=pt['cum_dX'], mode='lines+markers', name=f"Δ{lblX} (m)", line=dict(width=3), marker=dict(size=12)))
    fig_vel.add_trace(go.Scatter(x=xdates, y=pt['cum_dY'], mode='lines+markers', name=f"Δ{lblY} (m)", line=dict(width=3), marker=dict(size=12)))
    if pt['cum_dZ'].notna().any():
        fig_vel.add_trace(go.Scatter(x=xdates, y=pt['cum_dZ'], mode='lines+markers', name=f"Δ{lblZ} (m)", line=dict(width=3), marker=dict(size=12)))
    fig_vel.update_layout(
        title=f"Componenti degli spostamenti – punto {point_code}",
        xaxis_title="Data", yaxis_title="Variazione (m)",
        legend=dict(orientation='h', yanchor='bottom', y=0, xanchor='center', x=0.5)
    )
    _tick = _auto_date_ticks(xdates)
    fig_vel.update_xaxes(**_tick)
    _apply_plotly_theme(fig_vel, for_pdf=False, n_points=len(pt))
    return fig_vec, fig_vel


def _point_table_dataframe(res: pd.DataFrame, meta: dict, point_code: str, misurate_sel: list, bases_map: dict) -> pd.DataFrame:
    pt_tab = res[res['codice'].astype(str) == str(point_code)].copy()
    if misurate_sel:
        pt_tab = pt_tab[pt_tab['misurata'].isin(misurate_sel)]
    pt_tab['data'] = pd.to_datetime(pt_tab['data'])
    pt_tab = pt_tab.sort_values('data')
    if pt_tab.empty:
        return pd.DataFrame()

    # Riferimento per-punto
    b = bases_map.get(str(point_code))
    if b is None:
        base_x = pt_tab.iloc[0]['X']; base_y = pt_tab.iloc[0]['Y']; base_z = pt_tab.iloc[0].get('Z', np.nan)
    else:
        base_x, base_y, base_z = b['X0'], b['Y0'], b['Z0']

    has_z = ('Z' in pt_tab.columns) and pd.notna(base_z)

    pt_tab['ΔX'] = pt_tab['X'] - base_x
    pt_tab['ΔY'] = pt_tab['Y'] - base_y
    pt_tab['ΔZ'] = (pt_tab['Z'] - base_z) if has_z else np.nan
    pt_tab['dist.2D'] = np.sqrt(pt_tab['ΔX']**2 + pt_tab['ΔY']**2)
    pt_tab['dist.3D'] = np.sqrt(pt_tab['ΔX']**2 + pt_tab['ΔY']**2 + (pt_tab['ΔZ']**2))

    coord_sys = meta.get('coord_sys', None)
    is_utm = coord_sys in ('UTM32','UTM33','UTM')
    x_lbl, y_lbl, z_lbl = ('Est','Nord','Quota') if is_utm else ('X','Y','Z')

    df_show = pd.DataFrame({
        'Punto'   : str(point_code),
        'Misurata': pt_tab['misurata'],
        'Data'    : pt_tab['data'].dt.date,
        x_lbl     : pt_tab['X'],
        y_lbl     : pt_tab['Y'],
        z_lbl     : pt_tab['Z'] if has_z else np.nan,
        'ΔX'      : pt_tab['ΔX'],
        'ΔY'      : pt_tab['ΔY'],
        'ΔZ'      : pt_tab['ΔZ'],
        'dist.2D' : pt_tab['dist.2D'],
        'dist.3D' : pt_tab['dist.3D'],
    })
    num_cols = [c for c in [x_lbl,y_lbl,z_lbl,'ΔX','ΔY','ΔZ','dist.2D','dist.3D'] if c in df_show.columns]
    for c in num_cols:
        df_show[c] = pd.to_numeric(df_show[c], errors='coerce')
    df_show[num_cols] = df_show[num_cols].round(3)
    return df_show

def _pdf_full_report(project_title: str,
                     summary_df: pd.DataFrame,
                     cmp_mis: int,
                     cmp_date=None,
                     res: pd.DataFrame = None,
                     meta: dict = None,
                     misurate_sel: list = None,
                     bases_map: dict = None) -> bytes:
    """
    Crea un PDF multi-pagina:
    - Prima pagina: riepilogo (come _pdf_from_summary_table)
    - Segue: per ciascun 'codice' in summary_df, pagina con:
        * 'Analisi del Punto XX' (centrato) + descrizione (centrata)
        * 2 grafici affiancati
        * tabella 'Dati misurate' del punto
    - Footer con 'pag. X / Y' e riga contatti con link
    """
    from io import BytesIO
    buf = BytesIO()
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.pdfbase.pdfmetrics import stringWidth
        from reportlab.pdfgen import canvas
        import plotly.io as pio

        # ---------- Canvas numerato ----------
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase.pdfmetrics import stringWidth

        class NumberedCanvas(canvas.Canvas):
            """Canvas che disegna SOLO 'pag. X / Y' in fase di save() (niente link qui)."""
            def __init__(self, *args, **kwargs):
                canvas.Canvas.__init__(self, *args, **kwargs)
                self._saved_page_states = []

            def showPage(self):
                self._saved_page_states.append(dict(self.__dict__))
                self._startPage()

            def save(self):
                page_count = len(self._saved_page_states)
                for state in self._saved_page_states:
                    self.__dict__.update(state)
                    self._draw_page_number(page_count)
                    canvas.Canvas.showPage(self)
                canvas.Canvas.save(self)

            def _draw_page_number(self, page_count):
                w, h = self._pagesize
                self.setFont("Helvetica", 8)
                self.drawCentredString(w/2.0, 28, f"pag. {self._pageNumber} / {page_count}")

        def _footer_contacts(canv, doc_):
            """Riga contatti + link (disegnata nel passaggio normale, evita collisioni di Annot)."""
            from reportlab.lib import colors
            FOOTER_TEXT = ("Studio Sacchin - Studio Tecnico Associato - via Postgranz Strasse 16 - "
                           "39012 Meran/o (BZ) - Tel. 0473 445058 - "
                           "info@studiosacchin.it - www.studiosacchin.it")
            canv.saveState()
            w, h = doc_.pagesize
            y = 16
            canv.setFont('Helvetica', 8)
            txt_w = stringWidth(FOOTER_TEXT, 'Helvetica', 8)
            x = (w - txt_w) / 2.0
            canv.drawString(x, y, FOOTER_TEXT)

            # link cliccabili (OK farli qui: niente canvas.save “secondo passaggio”)
            prefix = "Studio Sacchin - Studio Tecnico Associato - via Postgranz Strasse 16 - 39012 Meran/o (BZ) - Tel. 0473 445058 - "
            mail = "info@studiosacchin.it"
            mid  = " - "
            url  = "www.studiosacchin.it"
            x_mail = x + stringWidth(prefix, 'Helvetica', 8)
            x_url  = x_mail + stringWidth(mail + mid, 'Helvetica', 8)

            canv.linkURL("mailto:info@studiosacchin.it",
                         (x_mail, y-1, x_mail + stringWidth(mail, 'Helvetica', 8), y+9),
                         relative=0, thickness=0, color=colors.blue)
            canv.linkURL("https://www.studiosacchin.it",
                         (x_url, y-1, x_url + stringWidth(url, 'Helvetica', 8), y+9),
                         relative=0, thickness=0, color=colors.blue)
            canv.restoreState()

        # ---------- Documento/stili ----------
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=24, rightMargin=24, topMargin=28, bottomMargin=36)
        styles = getSampleStyleSheet()
        title_style    = ParagraphStyle('TitleCenter', parent=styles['Title'],    alignment=1, fontSize=20, leading=24)
        subtitle_style = ParagraphStyle('H2Center',    parent=styles['Heading2'], alignment=1, fontSize=14, leading=18)
        bold_center    = ParagraphStyle('BoldCenter',  parent=styles['Heading4'], alignment=1, fontSize=10, leading=12)
        cell_style     = ParagraphStyle('cell', parent=styles['BodyText'], fontName='Helvetica', fontSize=8, leading=10)

        # Data estesa (per intestazione "Misurata N del ...")
        mesi = ['gennaio','febbraio','marzo','aprile','maggio','giugno','luglio','agosto','settembre','ottobre','novembre','dicembre']
        if cmp_date is not None:
            d = pd.to_datetime(cmp_date)
            cmp_date_ext = f"{d.day} {mesi[d.month-1]} {d.year}"
        else:
            cmp_date_ext = ""

        story = []

        # ---------- Pagina 1: riepilogo ("Stampa Tabella") ----------
        story.append(Paragraph("STUDIO SACCHIN - MONITORAGGIO TOPOGRAFICO", title_style))
        if project_title:
            story.append(Paragraph(str(project_title), subtitle_style))
        story.append(Spacer(1, 8))
        if cmp_date_ext:
            story.append(Paragraph(f"<b>Misurata {cmp_mis} del {cmp_date_ext}</b>", bold_center))
            story.append(Spacer(1, 6))

        # tabella riepilogo
        headers = ["codice","descrizione","misurata di riferimento","dX","dY","dZ","dist.2D","dist.3D"]
        col_widths = [55, 120, 125, 50, 50, 50, 45, 45]
        rows = []
        for _, r in summary_df.iterrows():
            c0 = Paragraph(str(r["codice"]), cell_style)
            c1 = Paragraph(str(r["descrizione"]), cell_style)
            c2 = Paragraph(str(r["misurata di riferimento"]), cell_style)
            c3 = f'{r["dX"]}'; c4 = f'{r["dY"]}'; c5 = f'{r["dZ"]}'; c6 = f'{r["dist.2D"]}'; c7 = f'{r["dist.3D"]}'
            rows.append([c0,c1,c2,c3,c4,c5,c6,c7])
        data = [headers] + rows
        tbl = Table(data, colWidths=col_widths, repeatRows=1, hAlign='LEFT')
        tbl.setStyle(TableStyle([
            ('FONT', (0,0), (-1,0), 'Helvetica-Bold', 9),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f0f0f0")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#333333")),
            ('ALIGN', (3,1), (-1,-1), 'RIGHT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 2),
            ('RIGHTPADDING', (0,0), (-1,-1), 2),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('FONTSIZE', (0,1), (-1,-1), 8),
            ('INNERGRID', (0,0), (-1,-1), 0.25, colors.HexColor("#cccccc")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#999999")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#fbfbfb")]),
        ]))
        story.append(tbl)

        # ---------- Pagine successive: un punto per pagina ----------
        # Lista di codici in ordine (solo quelli presenti nella misurata di confronto)
        codes = [str(c) for c in summary_df['codice'].tolist()]
        for i, code in enumerate(codes):
            story.append(PageBreak())

            # descrizione punto (prima disponibile)
            try:
                point_desc = (res[res['codice'].astype(str)==code]
                              .sort_values('misurata').iloc[0]['descrizione'])
            except Exception:
                point_desc = ""

            # Titoli del blocco punto
            story.append(Paragraph("STUDIO SACCHIN - MONITORAGGIO TOPOGRAFICO", title_style))
            if project_title:
                story.append(Paragraph(str(project_title), subtitle_style))
            story.append(Spacer(1, 8))
            story.append(Paragraph(f"<b>Analisi del Punto {code}</b>", bold_center))
            if point_desc:
                story.append(Paragraph(str(point_desc), ParagraphStyle('desc', parent=styles['BodyText'], alignment=1, fontSize=10)))
            story.append(Spacer(1, 10))

            # Grafici (PNG via kaleido)
            fig_vec, fig_vel = _build_point_figures(res, meta, code, misurate_sel or [], bases_map)
            if fig_vec is None or fig_vel is None:
                # niente dati → salta ai successivi
                continue

            # Applica tema per PDF con caratteri più grandi
            _apply_plotly_theme(fig_vec, for_pdf=True, n_points=None)
            _apply_plotly_theme(fig_vel, for_pdf=True, n_points=None)

            img_vec = pio.to_image(fig_vec, format="png", width=1100, height=825, scale=2)
            img_vel = pio.to_image(fig_vel, format="png", width=1100, height=825, scale=2)

            w_img = 260
            pic1 = RLImage(BytesIO(img_vec), width=w_img, height=w_img*0.75)
            pic2 = RLImage(BytesIO(img_vel), width=w_img, height=w_img*0.75)

            tbl_imgs = Table([[pic1, pic2]], colWidths=[w_img, w_img])
            tbl_imgs.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('LEFTPADDING', (0,0), (-1,-1), 2),
                ('RIGHTPADDING', (0,0), (-1,-1), 2),
                ('TOPPADDING', (0,0), (-1,-1), 2),
                ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ]))
            story.append(tbl_imgs)
            story.append(Spacer(1, 10))

            # Tabella dati del punto
            df_show = _point_table_dataframe(res, meta, code, misurate_sel or [], bases_map)
            if not df_show.empty:
                headers_p = [str(c) for c in df_show.columns.tolist()]
                rows_p = []
                num_cols = {c for c in df_show.columns if pd.api.types.is_numeric_dtype(df_show[c])}
                for _, rr in df_show.iterrows():
                    row_cells = []
                    for c in df_show.columns:
                        val = rr[c]
                        if c in num_cols:
                            row_cells.append("" if pd.isna(val) else f"{val}")
                        else:
                            row_cells.append(Paragraph("" if pd.isna(val) else str(val), cell_style))
                    rows_p.append(row_cells)
                data_p = [headers_p] + rows_p

                tbl_p = Table(data_p, repeatRows=1, hAlign='LEFT')
                tbl_p.setStyle(TableStyle([
                    ('FONT', (0,0), (-1,0), 'Helvetica-Bold', 9),
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f0f0f0")),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#333333")),
                    ('ALIGN', (0,1), (-1,-1), 'LEFT'),
                    *[('ALIGN', (i,1), (i,-1), 'RIGHT') for i,c in enumerate(df_show.columns) if c in num_cols],
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('LEFTPADDING', (0,0), (-1,-1), 2),
                    ('RIGHTPADDING', (0,0), (-1,-1), 2),
                    ('TOPPADDING', (0,0), (-1,-1), 2),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
                    ('FONTSIZE', (0,1), (-1,-1), 8),
                    ('INNERGRID', (0,0), (-1,-1), 0.25, colors.HexColor("#cccccc")),
                    ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#999999")),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#fbfbfb")]),
                ]))
                story.append(tbl_p)

        # Build con canvas numerato
        doc.build(story, onFirstPage=_footer_contacts, onLaterPages=_footer_contacts, canvasmaker=NumberedCanvas )
        buf.seek(0)
        return buf.getvalue()

    except Exception as e:
        st.error(f"Errore nella creazione del report completo: {e}")
        return b""

# Soglie (d2D e d3D) sulla tabella di confronto
d2d_thresh = float(st.session_state.get("d2d_num", 0.0) or 0.0)
d3d_thresh = float(st.session_state.get("d3d_num", 0.0) or 0.0)
if d2d_thresh > 0:
    comp = comp[comp['vettore_2D'].abs() >= d2d_thresh]
if d3d_thresh > 0:
    comp = comp[comp['vettore_3D'].abs() >= d3d_thresh]

cols_show = [
    'codice','descrizione',
    'X_ref','Y_ref','Z_ref','X_cmp','Y_cmp','Z_cmp',
    'dX','dY','dZ','vettore_2D','vettore_3D'
]

# ordina per codice crescente e resetta l'indice (così non appare la colonna numerica a sinistra)
comp_visible = comp[cols_show].sort_values('codice', ascending=True).reset_index(drop=True)

# mostra tabella SENZA indice
st.dataframe(comp_visible, use_container_width=True, hide_index=True)

# Download XLSX subito sotto la tabella, con i dati VISIBILI
@st.cache_data
def _to_xlsx(df: pd.DataFrame) -> bytes:
    from io import BytesIO
    buffer = BytesIO()
    # usa openpyxl (di solito già presente in Streamlit Cloud)
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Confronto")
    buffer.seek(0)
    return buffer.getvalue()

# --- utility per stampa pdf analisi punto
@st.cache_data
def _pdf_point_report(project_title: str,
                      point_code: str,
                      point_desc: str,
                      fig_vec,   # figura Plotly "Spostamenti nel piano"
                      fig_vel,   # figura Plotly "Velocità degli spostamenti"
                      df_show: pd.DataFrame) -> bytes:
    """
    Crea un PDF con:
    - Titolo e sottotitolo centrati (come Stampa Tabella)
    - 'Analisi del Punto XX' centrato + descrizione centrata
    - I due grafici affiancati
    - Tabella 'Dati misurate' del punto (stesse colonne di df_show)
    """
    from io import BytesIO
    buffer = BytesIO()
    try:
        # --- Dipendenze ---
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.pdfbase.pdfmetrics import stringWidth
        import plotly.io as pio

        # --- Documento ---
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            leftMargin=24, rightMargin=24, topMargin=28, bottomMargin=36
        )

        # --- Stili ---
        styles = getSampleStyleSheet()
        title_style    = ParagraphStyle('TitleCenter', parent=styles['Title'],    alignment=1, fontSize=20, leading=24)
        subtitle_style = ParagraphStyle('H2Center',    parent=styles['Heading2'], alignment=1, fontSize=14, leading=18)
        h4_center_bold = ParagraphStyle('H4CenterB',   parent=styles['Heading4'], alignment=1, fontSize=12, leading=14)
        h5_center      = ParagraphStyle('H5Center',    parent=styles['BodyText'], alignment=1, fontSize=10, leading=12)
        cell_style     = ParagraphStyle('cell',        parent=styles['BodyText'], fontName='Helvetica', fontSize=8, leading=10, spaceAfter=0, spaceBefore=0)

        # --- Footer (uguale a Stampa Tabella) ---
        FOOTER_TEXT = ("Studio Sacchin - Studio Tecnico Associato - via Postgranz Strasse 16 - "
                       "39012 Meran/o (BZ) - Tel. 0473 445058 - "
                       "info@studiosacchin.it - www.studiosacchin.it")

        def _footer(canvas, doc_):
            canvas.saveState()
            w, h = doc_.pagesize
            y = 18
            txt_w = stringWidth(FOOTER_TEXT, 'Helvetica', 8)
            x = (w - txt_w) / 2.0
            canvas.setFont('Helvetica', 8)
            canvas.drawString(x, y, FOOTER_TEXT)
            prefix = "Studio Sacchin - Studio Tecnico Associato - via Postgranz Strasse 16 - 39012 Meran/o (BZ) - Tel. 0473 445058 - "
            mail = "info@studiosacchin.it"
            mid  = " - "
            url  = "www.studiosacchin.it"
            x0   = (w - txt_w) / 2.0
            x_mail = x0 + stringWidth(prefix, 'Helvetica', 8)
            x_url  = x_mail + stringWidth(mail + mid, 'Helvetica', 8)
            canvas.linkURL("mailto:info@studiosacchin.it",
                           (x_mail, y-1, x_mail + stringWidth(mail, 'Helvetica', 8), y+9),
                           relative=0, thickness=0, color=colors.blue)
            canvas.linkURL("https://www.studiosacchin.it",
                           (x_url, y-1, x_url + stringWidth(url, 'Helvetica', 8), y+9),
                           relative=0, thickness=0, color=colors.blue)
            canvas.restoreState()

        story = []

        # --- Titoli ---
        story.append(Paragraph("STUDIO SACCHIN - MONITORAGGIO TOPOGRAFICO", title_style))
        if project_title:
            story.append(Paragraph(str(project_title), subtitle_style))
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"<b>Analisi del Punto {point_code}</b>", h4_center_bold))
        if point_desc:
            story.append(Paragraph(str(point_desc), h5_center))
        story.append(Spacer(1, 10))

        # --- Grafici: esporta a PNG con kaleido ---
        try:
            # Applica tema per PDF con caratteri più grandi
            _apply_plotly_theme(fig_vec, for_pdf=True, n_points=None)
            _apply_plotly_theme(fig_vel, for_pdf=True, n_points=None)
            img_vec = pio.to_image(fig_vec, format="png", width=1300, height=975, scale=2)
            img_vel = pio.to_image(fig_vel, format="png", width=1300, height=975, scale=2)
        except Exception as ex:
            st.error("Per esportare i grafici in PDF serve il pacchetto 'kaleido'. Installa con: pip install -U kaleido")
            return b""

        # dimensioni per affiancarli su A4 (2 colonne)
        # area utile ≈ 555 pt → due colonne da ~260 pt (con piccoli margini)
        w_img = 260
        pic1 = RLImage(BytesIO(img_vec), width=w_img, height=w_img*0.75)  # rapporto circa 4:3
        pic2 = RLImage(BytesIO(img_vel), width=w_img, height=w_img*0.75)

        tbl_imgs = Table([[pic1, pic2]], colWidths=[w_img, w_img])
        tbl_imgs.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 2),
            ('RIGHTPADDING', (0,0), (-1,-1), 2),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(tbl_imgs)
        story.append(Spacer(1, 10))

        # --- Tabella "Dati misurate" del punto ---
        headers = [str(c) for c in df_show.columns.tolist()]

        # costruzione righe con wrapping per testo
        rows = []
        # colonne numeriche per allineamento
        num_cols = {c for c in df_show.columns if pd.api.types.is_numeric_dtype(df_show[c])}
        for _, rr in df_show.iterrows():
            row_cells = []
            for c in df_show.columns:
                val = rr[c]
                if c in num_cols:
                    # numeri come stringhe, allineati a dx via TableStyle
                    row_cells.append("" if pd.isna(val) else f"{val}")
                else:
                    row_cells.append(Paragraph("" if pd.isna(val) else str(val), cell_style))
            rows.append(row_cells)

        data_tbl = [headers] + rows

        # colWidths automatici ma con limiti: forziamo alcune larghezze
        # Se hai molte colonne, puoi ridurre un po' ogni colonna numerica
        # Qui usiamo 'None' per lasciare adattare, e limitiamo padding
        tbl_data = Table(data_tbl, repeatRows=1, hAlign='LEFT')
        tbl_data.setStyle(TableStyle([
            ('FONT', (0,0), (-1,0), 'Helvetica-Bold', 9),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f0f0f0")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#333333")),
            ('ALIGN', (0,1), (-1,-1), 'LEFT'),
            # allinea a dx le colonne numeriche
            *[('ALIGN', (i,1), (i,-1), 'RIGHT') for i,c in enumerate(df_show.columns) if c in num_cols],
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 2),
            ('RIGHTPADDING', (0,0), (-1,-1), 2),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('FONTSIZE', (0,1), (-1,-1), 8),
            ('INNERGRID', (0,0), (-1,-1), 0.25, colors.HexColor("#cccccc")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#999999")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#fbfbfb")]),
        ]))
        story.append(tbl_data)

        # --- Build ---
        doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
        buffer.seek(0)
        return buffer.getvalue()

    except Exception as e:
        st.error(f"Errore nella creazione del PDF grafici: {e}")
        return b""

# --- Pulsanti azione (XLSX + PDF) affiancati ---
col_xlsx, col_pdf = st.columns([1, 1])

with col_xlsx:
    st.download_button(
        "⬇️ Scarica Tabella",  # rinominato
        data=_to_xlsx(comp_visible),
        file_name=f"confronto_misurate_{int(ref_tab)}_vs_{int(cmp_tab)}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_xlsx_main",
        use_container_width=True
    )

with col_pdf:
    summary_df = _build_summary_for_pdf(df_long, int(cmp_tab), mis_to_date)

    # Data confronto per intestazione PDF
    cmp_dt = mis_to_date.get(int(cmp_tab), None)
    try:
        cmp_dt = pd.to_datetime(cmp_dt) if cmp_dt is not None else None
    except Exception:
        cmp_dt = None

    pdf_bytes = _pdf_from_summary_table(titolo_importato, summary_df, int(cmp_tab), cmp_dt)

    st.download_button(
        "🖨️ Stampa Tabella",
        data=pdf_bytes if pdf_bytes else b"",
        file_name=f"report_{int(cmp_tab)}_{pd.to_datetime(mis_to_date[int(cmp_tab)]).strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
        disabled=(pdf_bytes == b""),
        key="dl_pdf_main",
        use_container_width=True
    )

# --- Stile colore pulsanti (verde Excel / rosso PDF) ---
st.markdown("""
<style>
/* Verde per XLSX */
div[data-testid="stDownloadButton"] > button[aria-label="⬇️ Scarica Tabella"]{
  background-color:#107C41 !important;
  color:#ffffff !important;
  border-color:#0e6b38 !important;
}
/* Rosso per PDF */
div[data-testid="stDownloadButton"] > button[aria-label="🖨️ Stampa Tabella"]{
  background-color:#D93025 !important;
  color:#ffffff !important;
  border-color:#b3241b !important;
}
</style>
""", unsafe_allow_html=True)


st.divider()

# =========================
# Mappa + Analisi punto (mappa sopra, analisi sotto) – aggiornati
# =========================

all_codes_sorted = sorted(res['codice'].unique())
if not all_codes_sorted:
    st.info("Nessun punto nei filtri correnti.")
else:
    # --- Mappa (scelta sfondo) a larghezza piena ---
    st.markdown("### Mappa punti")
    # Determina sistema di coordinate dal meta (nuovo formato) o default UTM32
    coord_sys = meta.get('coord_sys', None)
    # Opzioni sfondo in base al sistema
    if coord_sys == 'Sistema Locale':
        map_options = ["Mappa Locale"]
        default_idx = 0
    else:
        map_options = ["Mappa Locale", "OpenStreetMap", "Google Satellite", "Esri Satellite"]
        default_idx = 1
    map_bg = st.selectbox("Sfondo mappa", map_options, index=default_idx)

    # primo rilievo per ogni punto + tipologia (per colore)
    first_pts = df_long.sort_values('misurata').groupby('codice').first().reset_index()[['codice','descrizione','X','Y']]
    if 'tipologia' in df_long.columns:
        tipo_map = df_long.sort_values('misurata').groupby('codice').first().reset_index()[['codice','tipologia']]
        first_pts = first_pts.merge(tipo_map, on='codice', how='left')
    else:
        first_pts['tipologia'] = None

    # se non locale e pacchetti mancanti, fallback a mappa locale
    if map_bg != "Mappa Locale" and (folium is None or st_folium is None or Transformer is None):
        st.warning("Per la mappa interattiva servono i pacchetti: `folium`, `streamlit-folium`, `pyproj`.")
        map_bg = "Mappa Locale"

    if map_bg == "Mappa Locale":
        # scatter XY in sistema locale/UTM senza sfondo web – colori per tipologia
        if px is None:
            st.info("Plotly non disponibile")
        else:
            fig_local = px.scatter(first_pts, x='X', y='Y', color='tipologia', hover_data=['codice','descrizione','tipologia'])
            fig_local.update_yaxes(scaleanchor="x", scaleratio=1, showticklabels=False, showgrid=False, zeroline=False, title_text=None)
            fig_local.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, title_text=None)
            fig_local.update_layout(margin=dict(l=0,r=0,t=10,b=0), showlegend=True)
            st.plotly_chart(fig_local, use_container_width=True)
    else:
        # Trasformazione UTM32/33 → WGS84
        epsg_in = 32633 if meta.get("coord_sys")=="UTM33" else 32632
        transformer = Transformer.from_crs(epsg_in, 4326, always_xy=True)
        
        # filtra righe con X/Y valide ed evita crash su NaN
        first_pts = first_pts.copy()
        first_pts[['X','Y']] = first_pts[['X','Y']].apply(pd.to_numeric, errors='coerce')
        valid_idx = first_pts[['X','Y']].dropna().index
        fp_valid = first_pts.loc[valid_idx].copy()

        lons, lats = [], []
        for _, row in fp_valid.iterrows():
            lon, lat = transformer.transform(float(row['X']), float(row['Y']))
            lons.append(lon); lats.append(lat)

        fp_valid = fp_valid.assign(lon=lons, lat=lats)
        # mantieni solo i punti trasformati
        first_pts = fp_valid

        features = []
        for _, row in first_pts.iterrows():
            features.append({
                'type':'Feature',
                'properties':{'codice': str(row['codice']), 'descrizione': str(row['descrizione']), 'tipologia': (None if pd.isna(row.get('tipologia')) else str(row.get('tipologia')))},
                'geometry':{'type':'Point','coordinates':[row['lon'], row['lat']]}
            })
        geojson = {'type':'FeatureCollection','features':features}

        if first_pts[['lat','lon']].dropna().empty:
            # fallback (es. Merano): cambia se preferisci un centro diverso
            center = [46.669, 11.159]
        else:
            center = [float(np.nanmean(first_pts['lat'])), float(np.nanmean(first_pts['lon']))]
        
        
        m = folium.Map(location=center, zoom_start=18, tiles=None)
        if map_bg == "OpenStreetMap":
            folium.TileLayer(tiles='OpenStreetMap', name='OSM Standard', control=False).add_to(m)
        elif map_bg == "Google Satellite":
            folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', name='Google Satellite', control=False, max_zoom=20, attr='Google').add_to(m)
        else:
            folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', name='Esri Satellite', control=False, max_zoom=20, attr='Esri').add_to(m)

        # palette per tipologia
        tipos = sorted([t for t in first_pts['tipologia'].dropna().unique().tolist()]) if 'tipologia' in first_pts.columns else []
        base_colors = [
            "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
            "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
        ]
        color_map = {t: base_colors[i % len(base_colors)] for i, t in enumerate(tipos)}

        # >>> NUOVO: scegli i campi del tooltip in base al sistema di coordinate
        coord_sys = meta.get('coord_sys', None)
        is_utm = coord_sys in ('UTM32', 'UTM33', 'UTM')
        tooltip_fields = ['codice', 'descrizione'] if is_utm else ['codice', 'descrizione', 'tipologia']

        def _style_fun(feat):
            t = feat['properties'].get('tipologia')
            col = color_map.get(t, "#3388ff")
            return {'color': col, 'fillColor': col, 'fillOpacity': 0.8, 'radius': 6}

        gj = folium.GeoJson(
            geojson,
            name='Punti',
            tooltip=folium.GeoJsonTooltip(fields=tooltip_fields),  # <<< cambiato
            marker=folium.CircleMarker(),
            style_function=_style_fun
        )
        gj.add_to(m)

        # >>> NUOVO: in UTM aggiungo label/marker personalizzati con codice
        if is_utm:
            # funzione per creare l'HTML del marker/etichetta con codice
            def _label_html(code_str, bg_color):
                code_str = str(code_str)
                H = 28  # altezza/diametro coerente col cerchio breve

                if len(code_str) <= 4:
                    # testo dentro al pallino (come ora)
                    return f"""
                    <div style="
                        pointer-events:none;
                        display:flex; align-items:center; justify-content:center;
                        width:{H}px; height:{H}px; border-radius:50%;
                        background:{bg_color}; color:white; font-weight:700; font-size:12px;
                        border:2px solid rgba(0,0,0,0.25); box-shadow:0 0 4px rgba(0,0,0,0.2);
                    ">{code_str}</div>"""
                else:
                    # CAPSULA: rettangolo con angoli arrotondati (semicerchi alle estremità)
                    return f"""
                    <div style="
                        pointer-events:none;
                        display:inline-flex; align-items:center; justify-content:center;
                        --h:{H}px;
                        height:var(--h); padding:0 10px;
                        background:{bg_color}; color:white; font-weight:700; font-size:12px;
                        border:2px solid rgba(0,0,0,0.25); box-shadow:0 0 4px rgba(0,0,0,0.2);
                        border-radius:calc(var(--h)/2);
                        white-space:nowrap; line-height:var(--h);
                    ">{code_str}</div>"""

            # aggiungo un Marker con DivIcon per ogni feature (sovrapposto al GeoJson)
            for feat in geojson.get('features', []):
                props = feat.get('properties', {})
                geom = feat.get('geometry', {})
                if not geom or geom.get('type') != 'Point':
                    continue
                # attenzione all'ordine GeoJSON: [lon, lat]
                lon, lat = geom.get('coordinates', [None, None])[:2]
                if lat is None or lon is None:
                    continue
                tip = props.get('tipologia')
                col = color_map.get(tip, "#3388ff")
                code_str = props.get('codice', '')
                html = _label_html(code_str, col)

                folium.Marker(
                    location=[lat, lon],
                    icon=folium.DivIcon(
                        html=html,
                        class_name='empty',  # nessun CSS extra
                        icon_size=(0, 0),    # lasciamo al contenuto il suo size
                        icon_anchor=(14, 14) # ancora approssimativa al centro
                    )
                ).add_to(m)

        out = st_folium(m, height=520, width=None, returned_objects=['last_object_clicked'])

        # click mappa: aggiorna solo se cambia e poi rerun
        if out and out.get('last_object_clicked') and out['last_object_clicked'].get('properties'):
            props = out['last_object_clicked']['properties']
            if 'codice' in props:
                new_code = str(props['codice'])
                if st.session_state.get('selected_code') != new_code:
                    st.session_state.selected_code = new_code
                    st.rerun()


    # --- Selezione punto sotto la mappa ---

    try:
        all_codes_sorted = sorted(map(str, first_pts['codice'].dropna().unique().tolist()))
    except Exception:
        all_codes_sorted = []

    if not all_codes_sorted:
        st.warning("Nessun punto disponibile con i filtri correnti.")
        st.stop()

    # inizializza se mancante o non valido
    if 'selected_code' not in st.session_state or \
       st.session_state.selected_code not in all_codes_sorted:
        st.session_state.selected_code = all_codes_sorted[0]

    st.subheader("Analisi punti")

    # 2) Selectbox: usa SOLO la key, niente index
    st.selectbox(
        "Seleziona punto",
        options=all_codes_sorted,
        key='selected_code'
    )    

    # =========================
    # Dati punto selezionato – GRAFICI
    # =========================
    
    pt = res[res['codice'] == st.session_state.selected_code].sort_values('misurata')
    # Usiamo tutto il periodo (niente filtro sulle misurate)
    # pt = pt.copy() è implicito in molte operazioni successive; se serve:
    pt = pt.copy()

    if len(pt) == 0:
        st.info("Il punto non ha misurate nella selezione corrente.")
    else:
        # Base = prima misurata valida del punto (globale, non dipende dai filtri)
        b = bases_map.get(str(st.session_state.selected_code))
        if b is None:
            # fallback: primo campione visibile (non dovrebbe accadere)
            base_x = pt.iloc[0]['X']; base_y = pt.iloc[0]['Y']; base_z = pt.iloc[0].get('Z', np.nan)
        else:
            base_x, base_y, base_z = b['X0'], b['Y0'], b['Z0']

        # CUMULATE rispetto alla base per-punto
        pt['cum_dX'] = pt['X'] - base_x
        pt['cum_dY'] = pt['Y'] - base_y
        pt['cum_dZ'] = pt['Z'] - base_z

        # ---------- Spostamenti nel piano (ΔX, ΔY) ----------
        cc1, cc2 = st.columns(2)
        with cc1:
            fig_vec = go.Figure()

            # PALETTE per misurate (usiamo una palette qualitativa)
            palette = px.colors.qualitative.Plotly if px else [
                "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
            ]

            # segmenti colorati + marker e legenda per OGNI misurata                 # >>> MOD
            pt_r = pt.reset_index(drop=True)
            for i, row in pt_r.iterrows():
                color = palette[i % len(palette)]
                label = f"Misurata {int(row['misurata'])} – {pd.to_datetime(row['data']).date()}"

                # segmento dal punto precedente a quello corrente (se esiste)
                if i > 0:
                    prev = pt_r.loc[i-1]
                    fig_vec.add_trace(go.Scatter(
                        x=[prev['cum_dX'], row['cum_dX']],
                        y=[prev['cum_dY'], row['cum_dY']],
                        mode='lines',
                        line=dict(width=2, color=color),
                        hoverinfo='skip',
                        showlegend=False
                    ))

                # marker con legenda
                fig_vec.add_trace(go.Scatter(
                    x=[row['cum_dX']], y=[row['cum_dY']],
                    mode='markers',
                    marker=dict(size=9, color=color),
                    name=label,
                    hovertemplate=("ΔX: %{x:.3f} m<br>ΔY: %{y:.3f} m<br>" +
                                   f"{label}<extra></extra>")
                ))

            fig_vec.update_layout(
                title=f"Spostamenti nel piano – punto {st.session_state.selected_code}",
                xaxis_title="ΔX (m)", yaxis_title="ΔY (m)",
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation='v', x=0.01, y=0.99)
            )
            fig_vec.update_yaxes(scaleanchor="x", scaleratio=1)
            fig_vec.update_xaxes(constrain='domain')
            st.plotly_chart(fig_vec, use_container_width=True, config={'locale': 'it'})

        # ---------- Velocità / Componenti degli spostamenti ----------
        with cc2:
            coord_sys = meta.get('coord_sys', None)
            if coord_sys in ('UTM32', 'UTM33', 'UTM'):                                   # >>> MOD
                lblX, lblY, lblZ = "Est", "Nord", "Quota"
            else:
                lblX, lblY, lblZ = "X", "Y", "Z"

            fig_vel = go.Figure()
            # Asse X come datetime Python (evita epoch ns in PDF)
            _pt_data = pd.to_datetime(pt['data'], errors='coerce')
            try:
                _pt_data = _pt_data.dt.tz_convert(None)
            except Exception:
                try:
                    _pt_data = _pt_data.dt.tz_localize(None)
                except Exception:
                    pass
            xdates = np.array(np.array(_pt_data.dt.to_pydatetime()))
            _tick = _auto_date_ticks(xdates)
            fig_vel.update_xaxes(**_tick)
            # COLORI richiesti: X/Est=Rosso, Y/Nord=Verde, Z/Quota=Blu                 # >>> MOD
            color_x = "red"
            color_y = "green"
            color_z = "blue"

            # Rinomina grafico: "Velocità degli spostamenti"                            # >>> MOD
            fig_vel.add_trace(go.Scatter(
                x=xdates, y=pt['cum_dX'], mode='lines+markers',
                name=f"Δ{lblX} (m)",
                line=dict(color=color_x), marker=dict(color=color_x)
            ))
            fig_vel.add_trace(go.Scatter(
                x=xdates, y=pt['cum_dY'], mode='lines+markers',
                name=f"Δ{lblY} (m)",
                line=dict(color=color_y), marker=dict(color=color_y)
            ))
            if pt['cum_dZ'].notna().any():
                fig_vel.add_trace(go.Scatter(
                    x=xdates, y=pt['cum_dZ'], mode='lines+markers',
                    name=f"Δ{lblZ} (m)",
                    line=dict(color=color_z), marker=dict(color=color_z)
                ))

            fig_vel.update_layout(
                title=f"Velocità degli spostamenti – punto {st.session_state.selected_code}",  # >>> MOD
                xaxis_title="Data", yaxis_title="Variazione (m)",
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation='h', yanchor='bottom', y=0, xanchor='center', x=0.5)
            )
            st.plotly_chart(fig_vel, use_container_width=True, config={'locale': 'it'})

        # ---------- Tabella dettaglio misurate del punto ----------
        sel_code = st.session_state.selected_code
        st.markdown(f"#### Dati misurate del punto {sel_code}")

        # ricostruisco i dati del punto selezionato e ordino per data crescente
        sel_code = st.session_state.selected_code
        pt_tab = res[res['codice'].astype(str) == str(sel_code)].copy()

        pt_tab['data'] = pd.to_datetime(pt_tab['data'])
        pt_tab = pt_tab.sort_values('data')

        if len(pt_tab) == 0:
            st.info("Nessuna misurata disponibile per il punto selezionato nella selezione corrente.")
        else:
            # Riferimento per-punto: prima misurata valida globale
            b = bases_map.get(str(sel_code))
            if b is None:
                base_x = pt_tab.iloc[0]['X']; base_y = pt_tab.iloc[0]['Y']
                base_z = pt_tab.iloc[0].get('Z', np.nan)
            else:
                base_x, base_y, base_z = b['X0'], b['Y0'], b['Z0']

            has_z = 'Z' in pt_tab.columns and pd.notna(base_z)

            # Δ rispetto alla base per-punto
            pt_tab['ΔX'] = pt_tab['X'] - base_x
            pt_tab['ΔY'] = pt_tab['Y'] - base_y
            pt_tab['ΔZ'] = (pt_tab['Z'] - base_z) if has_z else np.nan

            # distanze
            pt_tab['dist.2D'] = np.sqrt(pt_tab['ΔX']**2 + pt_tab['ΔY']**2)
            pt_tab['dist.3D'] = np.sqrt(pt_tab['ΔX']**2 + pt_tab['ΔY']**2 + (pt_tab['ΔZ']**2))

            # etichette X/Y/Z ↔ Est/Nord/Quota in base al sistema di coordinate
            coord_sys = meta.get('coord_sys', None)
            is_utm = coord_sys in ('UTM32', 'UTM33', 'UTM')
            x_lbl, y_lbl, z_lbl = ('Est','Nord','Quota') if is_utm else ('X','Y','Z')

            # costruiamo la tabella nell’ordine richiesto
            df_show = pd.DataFrame({
                'Punto'   : str(sel_code),
                'Misurata': pt_tab['misurata'],
                'Data'    : pt_tab['data'].dt.date,
                x_lbl     : pt_tab['X'],
                y_lbl     : pt_tab['Y'],
                z_lbl     : pt_tab['Z'] if has_z else np.nan,
                'ΔX'      : pt_tab['ΔX'],
                'ΔY'      : pt_tab['ΔY'],
                'ΔZ'      : pt_tab['ΔZ'],
                'dist.2D' : pt_tab['dist.2D'],
                'dist.3D' : pt_tab['dist.3D'],
            })

            # arrotondamenti (se vuoi 3 decimali su numerici)
            num_cols = [c for c in [x_lbl, y_lbl, z_lbl, 'ΔX', 'ΔY', 'ΔZ', 'dist.2D', 'dist.3D'] if c in df_show.columns]
            for c in num_cols:
                df_show[c] = pd.to_numeric(df_show[c], errors='coerce')
            df_show[num_cols] = df_show[num_cols].round(3)

            # visualizzazione
            st.dataframe(df_show, use_container_width=True, hide_index=True)

            # download Excel
            @st.cache_data
            def _to_xlsx_point(df: pd.DataFrame) -> bytes:
                from io import BytesIO
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name=f"Punto_{sel_code}")
                buffer.seek(0)
                return buffer.getvalue()

            # pulsanti affiancati: Scarica Tabella (XLSX) + Esporta Grafici (PDF)
            cA, cB = st.columns(2)

            with cA:
                st.download_button(
                    "⬇️ Scarica Tabella",
                    data=_to_xlsx_point(df_show),
                    file_name=f"dati_punto_{sel_code}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            with cB:
                # descrizione punto (prima disponibile)
                try:
                    point_desc = (
                        df_long[df_long['codice'].astype(str)==str(sel_code)]
                        .sort_values('misurata')
                        .iloc[0]['descrizione']
                    )
                except Exception:
                    point_desc = ""

                # Crea il PDF con i due grafici e la tabella del punto
                point_pdf = _pdf_point_report(
                    project_title=titolo_importato,
                    point_code=str(sel_code),
                    point_desc=str(point_desc) if point_desc is not None else "",
                    fig_vec=fig_vec,
                    fig_vel=fig_vel,
                    df_show=df_show
                )

                st.download_button(
                    "🖨️ Esporta Grafici",
                    data=point_pdf if point_pdf else b"",
                    file_name=f"analisi_punto_{sel_code}.pdf",
                    mime="application/pdf",
                    disabled=(point_pdf == b""),
                    use_container_width=True
                )
            
st.divider()
st.markdown("### Report completo")

# placeholder per messaggi e per il pulsante di download finale
msg_box = st.empty()
dl_box = st.empty()

# stato in sessione
if 'is_generating_full' not in st.session_state:
    st.session_state.is_generating_full = False
if 'full_pdf_bytes' not in st.session_state:
    st.session_state.full_pdf_bytes = None

# bottone di azione (sempre visibile)
#clicked = st.download_button(
#    "ESPORTA REPORT",
#    data=b"",  # per Streamlit dev'essere bytes; qui mettiamo vuoto
#    file_name="report_completo.pdf",
#    mime="application/pdf",
#    disabled=True if st.session_state.is_generating_full else (st.session_state.full_pdf_bytes is None),
#    key="btn_full_report",
#    use_container_width=True
#)

# se abbiamo già un PDF pronto (da un run precedente), mostra il vero download
if st.session_state.full_pdf_bytes:
    dl_box.download_button(
        "⬇️ Scarica Report Completo (PDF)",
        data=st.session_state.full_pdf_bytes,
        file_name=f"report_completo_mis_{int(cmp_tab)}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

# bottone “Crea/rigenera” che avvia la redazione
gen_clicked = st.button("Genera/aggiorna il report completo", use_container_width=True)

if gen_clicked:
    st.session_state.is_generating_full = True
    st.session_state.full_pdf_bytes = None
    msg_box.info("🛠️ Redazione del report in corso. Il processo potrebbe richiedere diversi minuti.")

    # — ricostruisci dati come per Stampa Tabella —
    summary_df = _build_summary_for_pdf(df_long, int(cmp_tab), mis_to_date)
    cmp_dt = mis_to_date.get(int(cmp_tab), None)
    try:
        cmp_dt = pd.to_datetime(cmp_dt) if cmp_dt is not None else None
    except Exception:
        cmp_dt = None

    # — genera PDF —
    full_pdf = _pdf_full_report(
        project_title=titolo_importato,
        summary_df=summary_df,
        cmp_mis=int(cmp_tab),
        cmp_date=cmp_dt,
        res=res,
        meta=meta,
        misurate_sel=None,  # niente filtro misurate
        bases_map=bases_map
    )

    # salva in sessione e aggiorna UI
    st.session_state.full_pdf_bytes = full_pdf if full_pdf else None
    st.session_state.is_generating_full = False

    if st.session_state.full_pdf_bytes:
        msg_box.success("✅ Report completo pronto al download.")
        dl_box.download_button(
            "⬇️ Scarica Report Completo (PDF)",
            data=st.session_state.full_pdf_bytes,
            file_name=f"report_completo_mis_{int(cmp_tab)}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    else:
        msg_box.error("❌ Errore nella redazione del report. Riprova.")

st.divider()
