# monitoraggio.py
# Web app per monitoraggio topografico
# import file csv generato monitoraggio Studio Sacchin
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
import plotly.express as px
import plotly.graph_objects as go

# opzionali (per mappa e highlight avanzato)
try:
    import altair as alt
except Exception:
    alt = None

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
        if any(v.strip() == 'codice' for v in row_vals):
            header_row_idx = r
            break
    if header_row_idx is None:
        header_row_idx = 4

    header_row = raw.iloc[header_row_idx]

    date_cols = []
    for c in raw.columns:
        val = raw.iloc[0, c]
        if isinstance(val, (str, dt.datetime, pd.Timestamp)):
            try:
                d = pd.to_datetime(val)
                if d.year > 1990:
                    date_cols.append((c, d))
            except Exception:
                pass
    if not date_cols:
        raise ValueError("Formato 'matrice' non riconosciuto: nessuna data nella prima riga.")

    meta_rows = {'temperatura': 1, 'pressione': 2, 'misurata_flag': 3}
    point_meta_idx = _detect_point_meta_cols(header_row)

    records = []
    sorted_dates = [d for _, d in sorted(date_cols, key=lambda x: x[0])]
    date_to_mis = {d: i+1 for i, d in enumerate(sorted_dates)}

    for (c_idx, dttm) in date_cols:
        start_r = header_row_idx + 1
        end_r = len(raw)
        block = pd.DataFrame({
            'codice': raw.iloc[start_r:end_r, point_meta_idx.get('codice', 0)].values,
            'descrizione': raw.iloc[start_r:end_r, point_meta_idx.get('descrizione', 1)].values,
            'X': raw.iloc[start_r:end_r, c_idx].values,
            'Y': raw.iloc[start_r:end_r, c_idx+1].values,
            'Z': raw.iloc[start_r:end_r, c_idx+2].values,
        })
        if 'prima_misurata' in point_meta_idx:
            block['prima_misurata'] = pd.to_numeric(raw.iloc[start_r:end_r, point_meta_idx['prima_misurata']].values, errors='coerce')
        else:
            block['prima_misurata'] = np.nan
        block['data'] = pd.to_datetime(dttm)
        block['misurata'] = date_to_mis[pd.to_datetime(dttm)]
        def _get_meta_val_row(ridx, col_idx):
            try:
                return raw.iloc[ridx, col_idx]
            except Exception:
                return None
        block['temperatura'] = pd.to_numeric(_get_meta_val_row(meta_rows['temperatura'], c_idx), errors='coerce')
        block['pressione'] = pd.to_numeric(_get_meta_val_row(meta_rows['pressione'], c_idx), errors='coerce')
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
        'date': 'data', 'epoch': 'data', 'epoca': 'data',
        'e': 'X', 'east': 'X', 'easting': 'X', 'x': 'X',
        'n': 'Y', 'north': 'Y', 'northing': 'Y', 'y': 'Y',
        'z': 'Z', 'quota': 'Z', 'alt': 'Z', 'h': 'Z',
        'prima_misurata':'prima_misurata','misurata_inizio':'prima_misurata', 'first_misurata':'prima_misurata'
    }
    lowercols = {c.lower(): c for c in df.columns}
    final_cols = {}
    for k, target in rename_map.items():
        if k in lowercols:
            final_cols[lowercols[k]] = target
    df = df.rename(columns=final_cols)

    required = {'codice', 'data', 'X', 'Y'}
    if not required.issubset(set(df.columns)):
        raise ValueError("Formato lungo non riconosciuto: servono colonne almeno 'codice', 'data', 'X', 'Y'")

    df['data'] = pd.to_datetime(df['data'], errors='coerce')
    for c in ['X','Y','Z','prima_misurata']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan

    order = {d:i+1 for i, d in enumerate(sorted(df['data'].dropna().unique()))}
    df['misurata'] = df['data'].map(order)

    meta = {
        'n_punti': df['codice'].nunique(),
        'n_epoche': df['data'].nunique(),
        'date': sorted(df['data'].dropna().unique()),
        'sheet': None,
        'format': 'long',
        'title': None,
    }
    return df[['codice','descrizione','prima_misurata','data','misurata','X','Y','Z']], meta


# --- Nuovo parser: formato ";" con prima riga titolo e seconda riga header ---

def _parse_semicolon_wide_from_bytes(data: bytes) -> Tuple[pd.DataFrame, Dict]:
    text = data.decode('utf-8', errors='ignore')
    # rimuovi righe completamente vuote in coda/testa
    lines = [ln for ln in (l.strip() for l in text.splitlines()) if ln != '']
    if len(lines) < 2:
        raise ValueError("File troppo corto per il formato ';' con titolo + header")

    title_line = lines[0].strip()
    csv_text = '\n'.join(lines[1:])

    df0 = pd.read_csv(io.StringIO(csv_text), sep=';', engine='python')
    # elimina colonne vuote create da ';' finali
    df0 = df0.loc[:, [c for c in df0.columns if str(c).strip().lower() != 'unnamed: 0']]

    # trova colonne codice/descrizione (case-insensitive)
    cols_lower = {str(c).strip().lower(): c for c in df0.columns}
    if 'codice' not in cols_lower or 'descrizione' not in cols_lower:
        raise ValueError("Header non valido: attese colonne 'Codice' e 'Descrizione' nella seconda riga")
    c_code = cols_lower['codice']
    c_desc = cols_lower['descrizione']

    # colonne successive: gruppi da 4 (data, X, Y, Z) per epoca
    rest = [c for c in df0.columns if c not in (c_code, c_desc)]
    groups = []
    for i in range(0, len(rest), 4):
        block = rest[i:i+4]
        if len(block) < 4:
            break
        groups.append(tuple(block))  # (data, X, Y, Z)

    records = []
    for _, row in df0.iterrows():
        code = str(row[c_code]).strip()
        desc = str(row[c_desc]).strip() if not pd.isna(row[c_desc]) else ''
        for gi, (dcol, xcol, ycol, zcol) in enumerate(groups, start=1):
            dval = row[dcol]
            x = pd.to_numeric(row[xcol], errors='coerce')
            y = pd.to_numeric(row[ycol], errors='coerce')
            z = pd.to_numeric(row[zcol], errors='coerce')
            if pd.isna(dval) and pd.isna(x) and pd.isna(y) and pd.isna(z):
                continue
            # data: può essere numero stile Excel (44430) o stringa
            date = pd.NaT
            if not pd.isna(dval):
                try:
                    if isinstance(dval, (int, float)) and not pd.isna(dval):
                        # Excel serial date (origin 1899-12-30)
                        date = pd.to_datetime('1899-12-30') + pd.to_timedelta(int(dval), unit='D')
                    else:
                        date = pd.to_datetime(dval, dayfirst=True, errors='coerce')
                except Exception:
                    date = pd.to_datetime(dval, dayfirst=True, errors='coerce')
            records.append({'codice': code, 'descrizione': desc, 'data': date, 'misurata': gi, 'X': x, 'Y': y, 'Z': z})

    df_long = pd.DataFrame.from_records(records)
    if df_long.empty:
        raise ValueError("Nessun dato interpretato dal formato ';'")

    meta = {
        'n_punti': df_long['codice'].nunique(),
        'n_epoche': df_long['misurata'].nunique(),
        'date': sorted(df_long['data'].dropna().unique()),
        'sheet': None,
        'format': 'semicolon_wide',
        'title': title_line,
    }
    return df_long, meta


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
        data = file.read()
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
# UI
# -----------------------------

st.set_page_config(page_title="Monitoraggio Topografico", layout="wide")

# Caricamento
with st.sidebar:
    st.header("Caricamento dati")
    up = st.file_uploader("Carica file (.xlsx, .xls, .csv, .txt)", type=["xlsx","xls","csv","txt"], accept_multiple_files=False)
    st.caption("Nuovo formato supportato: prima riga titolo; seconda riga header con 'Codice;Descrizione;data_1;X_1;Y_1;Z_1;...' (separatore ';').")

    st.divider()
    st.header("Impostazioni")
    st.caption("Seleziona le misurate da analizzare. Modalità predefinita: Intero periodo.")

if up is None:
    st.title("📐 MONITORAGGIO TOPOGRAFICO")
    st.info("🔼 Carica un file per iniziare. Puoi usare direttamente il file di esempio.")
    st.stop()

# Parsing
try:
    df_long, meta = parse_upload(up)
except Exception as e:
    st.error(f"Errore di parsing: {e}")
    st.stop()

# Titolo pagina dinamico dal file
page_title_suffix = str(meta.get('title') or '').strip()
if page_title_suffix:
    st.title(f"📐 MONITORAGGIO TOPOGRAFICO – {page_title_suffix.upper()}")
else:
    st.title("📐 MONITORAGGIO TOPOGRAFICO – Dashboard Interattiva")

# Pulizia
df_long = df_long.dropna(subset=['codice','data'])
df_long['data'] = pd.to_datetime(df_long['data'])

# Misurate
mis_sorted = sorted(
    df_long[['misurata','data']].dropna().drop_duplicates().sort_values('misurata').itertuples(index=False),
    key=lambda x: x[0]
)
mis_to_date = {int(m): pd.to_datetime(d) for m, d in mis_sorted}
all_mis = sorted(mis_to_date.keys())

# State iniziale
if 'selection_mode' not in st.session_state:
    st.session_state.selection_mode = 'Intero periodo'
if 'misurate_sel' not in st.session_state:
    st.session_state.misurate_sel = all_mis
if 'selected_code' not in st.session_state:
    st.session_state.selected_code = None

# Sidebar: selezione misurate
with st.sidebar:
    st.header("Selezione misurate")
    mode_options = ["Intero periodo","Ultime N","Personalizzata","Intervallo"]
    mode = st.radio("Modalità", mode_options, index=mode_options.index(st.session_state.selection_mode), horizontal=True, key='mode_radio')

    if mode == "Intero periodo":
        misurate_sel = all_mis
    elif mode == "Ultime N":
        N = st.number_input("N ultime misurate", min_value=1, max_value=max(1,len(all_mis)), value=min(3,len(all_mis)))
        misurate_sel = all_mis[-int(N):]
    elif mode == "Personalizzata":
        default_sel = st.session_state.misurate_sel if st.session_state.misurate_sel else all_mis
        misurate_sel = st.multiselect("Scegli misurate", options=all_mis, default=default_sel)
    else:
        start_m = st.selectbox("Prima misurata", options=all_mis, index=0)
        end_m = st.selectbox("Ultima misurata", options=all_mis, index=len(all_mis)-1)
        if start_m > end_m:
            start_m, end_m = end_m, start_m
        misurate_sel = [m for m in all_mis if start_m <= m <= end_m]

    st.session_state.misurate_sel = misurate_sel
    st.session_state.selection_mode = mode

    ref_m = st.selectbox("Misurata di riferimento (globale)", options=all_mis, index=0, format_func=lambda m: f"Misurata {m} – {mis_to_date[m].date()}")

    codes = sorted(df_long['codice'].unique())
    selected_codes = st.multiselect("Punti", options=codes, default=[])
    disp_thresh = st.number_input("Soglia d2D (m)", min_value=0.0, value=0.0, step=0.001, format="%.3f")

# Calcoli base
res = compute_displacements(df_long, int(ref_m))
res['misurata_ref'] = ref_m
mask = res['misurata'].isin(st.session_state.misurate_sel if st.session_state.misurate_sel else all_mis)
if selected_codes:
    mask &= res['codice'].isin(selected_codes)
res_f = res[mask].copy()
if disp_thresh and disp_thresh > 0:
    res_f = res_f[(res_f['d2D'].abs() >= disp_thresh)]

# KPI
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Punti", int(res_f['codice'].nunique()))
with c2:
    st.metric("Misurate selezionate", len(sorted(set(res_f['misurata']))))
with c3:
    m2d = res_f.groupby('codice')['d2D'].max().max()
    st.metric("Spost. max 2D (m)", f"{(m2d if pd.notna(m2d) else 0):.3f}")
with c4:
    mz = res_f['dZ'].abs().max()
    st.metric("Spost. max Z (m)", f"{(mz if pd.notna(mz) else 0):.3f}")

st.divider()

# =========================
# Vista planare – spostamenti (coordinate locali, colore per misurata)
# =========================

st.subheader("Vista planare – spostamenti")

# Pulsanti misurate sotto il titolo
st.caption("Attiva/Disattiva misurate")
btn_selected = []
cols_per_row = 10
rows = (len(all_mis) + cols_per_row - 1) // cols_per_row
for r in range(rows):
    cols = st.columns(min(cols_per_row, len(all_mis) - r*cols_per_row))
    for i, m in enumerate(all_mis[r*cols_per_row : (r+1)*cols_per_row]):
        default_on = m in st.session_state.misurate_sel
        with cols[i]:
            state = st.checkbox(str(m), value=default_on, key=f"mis_btn_{m}")
            if state:
                btn_selected.append(m)

btn_selected_sorted = sorted(btn_selected)
if set(btn_selected_sorted) != set(st.session_state.misurate_sel):
    st.session_state.misurate_sel = btn_selected_sorted
    st.session_state.selection_mode = 'Personalizzata'

misurate_sel = st.session_state.misurate_sel if st.session_state.misurate_sel else all_mis
mask_buttons = res['misurata'].isin(misurate_sel)
if selected_codes:
    mask_buttons &= res['codice'].isin(selected_codes)
res_btn = res[mask_buttons].copy()
if disp_thresh and disp_thresh > 0:
    res_btn = res_btn[(res_btn['d2D'].abs() >= disp_thresh)]

# Coordinate locali (baseline: prima misurata nel subset per punto)
local_df = res_btn.copy()
firsts = local_df.sort_values('misurata').groupby('codice')[['X','Y']].first().rename(columns={'X':'Xb','Y':'Yb'})
local_df = local_df.merge(firsts, left_on='codice', right_index=True, how='left')
local_df['lX'] = local_df['X'] - local_df['Xb']
local_df['lY'] = local_df['Y'] - local_df['Yb']
local_df['ld2D'] = np.sqrt(local_df['lX']**2 + local_df['lY']**2)

# Chart: Altair (hover → evidenzia stesso codice), fallback Plotly
if alt is not None and not local_df[['lX','lY']].dropna().empty:
    x_min, x_max = local_df['lX'].min(), local_df['lX'].max()
    y_min, y_max = local_df['lY'].min(), local_df['lY'].max()
    span = float(max(x_max - x_min, y_max - y_min)) if np.isfinite([x_min, x_max, y_min, y_max]).all() else 1.0
    cx = float((x_max + x_min)/2.0) if np.isfinite([x_min, x_max]).all() else 0.0
    cy = float((y_max + y_min)/2.0) if np.isfinite([y_min, y_max]).all() else 0.0
    dom_x = [cx - span/2.0, cx + span/2.0]
    dom_y = [cy - span/2.0, cy + span/2.0]

    hover = alt.selection_single(fields=['codice'], on='mouseover', nearest=True, empty='none')
    base = alt.Chart(local_df).mark_circle(size=60).encode(
        x=alt.X('lX:Q', scale=alt.Scale(domain=dom_x), title='ΔX locale (m)'),
        y=alt.Y('lY:Q', scale=alt.Scale(domain=dom_y), title='ΔY locale (m)'),
        color=alt.Color('misurata:N', legend=alt.Legend(title='Misurata')),
        tooltip=['codice','descrizione','misurata','data:T','ld2D:Q']
    ).properties(width=600, height=600)

    chart = base.encode(opacity=alt.condition(hover, alt.value(1), alt.value(0.2))).add_selection(hover)
    st.altair_chart(chart, use_container_width=False)
else:
    fig_scatter = px.scatter(local_df, x='lX', y='lY', color='misurata',
                             hover_data={'codice':True,'descrizione':True,'misurata':True,'data':True,'ld2D':':.3f'},
                             labels={'lX':'ΔX locale (m)', 'lY':'ΔY locale (m)', 'misurata':'Misurata'},
                             title="Spostamenti locali per misurata")
    fig_scatter.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_scatter.update_layout(margin=dict(l=10,r=10,t=40,b=10), legend_title_text="Misurata")
    st.plotly_chart(fig_scatter, use_container_width=True)
    if alt is None:
        st.info("Per evidenziare automaticamente il punto al passaggio del mouse, installa Altair: `pip install altair`.")

st.divider()

# =========================
# Tabella confronto tra misurate (ref vs confronto)
# =========================

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

# Applica i filtri globali: punti selezionati e soglia d2D (se impostata)
if selected_codes:
    comp = comp[comp['codice'].isin(selected_codes)]
if disp_thresh and disp_thresh > 0:
    comp = comp[comp['vettore_2D'].abs() >= disp_thresh]

cols_show = [
    'codice','descrizione',
    'X_ref','Y_ref','Z_ref','X_cmp','Y_cmp','Z_cmp',
    'dX','dY','dZ','vettore_2D','vettore_3D'
]
comp_visible = comp[cols_show].sort_values('vettore_2D', ascending=False)

st.dataframe(comp_visible, use_container_width=True)

# Download CSV subito sotto la tabella, con i dati VISIBILI
@st.cache_data
def _to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

st.download_button(
    "⬇️ Scarica tabella (CSV)",
    data=_to_csv(comp_visible),
    file_name=f"confronto_misurate_{int(ref_tab)}_vs_{int(cmp_tab)}.csv",
    mime="text/csv"
)

st.divider()

# =========================
# Inspector punto – mappa a larghezza piena, spostamenti e velocità
# =========================

st.subheader("Inspector punto")

all_codes_sorted = sorted(res['codice'].unique())
if not all_codes_sorted:
    st.info("Nessun punto nei filtri correnti.")
else:
    # --- Mappa (scelta sfondo) a larghezza piena ---
    st.markdown("### Mappa punti")
    map_bg = st.selectbox(
        "Sfondo mappa",
        ["Nessuna mappa (locale)", "OpenStreetMap Standard", "Google Satellite", "Esri Satellite"],
        index=1
    )

    first_pts = df_long.sort_values('misurata').groupby('codice').first().reset_index()[['codice','descrizione','X','Y']]

    if map_bg != "Nessuna mappa (locale)" and (folium is None or st_folium is None or Transformer is None):
        st.warning("Per la mappa interattiva servono i pacchetti: `folium`, `streamlit-folium`, `pyproj`.")
        map_bg = "Nessuna mappa (locale)"

    if map_bg == "Nessuna mappa (locale)":
        fig_local = px.scatter(first_pts, x='X', y='Y', hover_data=['codice','descrizione'])
        fig_local.update_yaxes(scaleanchor="x", scaleratio=1, showticklabels=False, showgrid=False, zeroline=False, title_text=None)
        fig_local.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, title_text=None, constrain='domain')
        fig_local.update_layout(height=500, margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
        st.plotly_chart(fig_local, use_container_width=True)
    else:
        epsg_in = 32632
        transformer = Transformer.from_crs(epsg_in, 4326, always_xy=True)
        lons, lats = [], []
        for _, row in first_pts.iterrows():
            lon, lat = transformer.transform(float(row['X']), float(row['Y']))
            lons.append(lon); lats.append(lat)
        first_pts = first_pts.assign(lon=lons, lat=lats)

        features = []
        for _, row in first_pts.iterrows():
            features.append({
                'type':'Feature',
                'properties':{'codice': str(row['codice']), 'descrizione': str(row['descrizione'])},
                'geometry':{'type':'Point','coordinates':[row['lon'], row['lat']]}
            })
        geojson = {'type':'FeatureCollection','features':features}

        center = [float(np.nanmean(first_pts['lat'])), float(np.nanmean(first_pts['lon']))]
        m = folium.Map(location=center, zoom_start=18, tiles=None)
        if map_bg == "OpenStreetMap Standard":
            folium.TileLayer(tiles='OpenStreetMap', name='OSM Standard', control=False).add_to(m)
        elif map_bg == "Google Satellite":
            folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google', name='Google Satellite', control=False, max_zoom=20).add_to(m)
        else:
            folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri', name='Esri Satellite', control=False, max_zoom=20).add_to(m)

        gj = folium.GeoJson(geojson, name='Punti', tooltip=folium.GeoJsonTooltip(fields=['codice','descrizione']))
        gj.add_to(m)

        out = st_folium(m, height=520, width=None, returned_objects=['last_object_clicked'])
        if out and out.get('last_object_clicked') and out['last_object_clicked'].get('properties'):
            props = out['last_object_clicked']['properties']
            if 'codice' in props:
                st.session_state.selected_code = str(props['codice'])

    # --- Selezione punto sotto la mappa ---
    if st.session_state.selected_code is None or st.session_state.selected_code not in all_codes_sorted:
        st.session_state.selected_code = all_codes_sorted[0]
    code_sel = st.selectbox("Seleziona punto", options=all_codes_sorted,
                            index=all_codes_sorted.index(st.session_state.selected_code),
                            key='code_select')
    st.session_state.selected_code = code_sel

    # Dati punto selezionato
    pt = res[res['codice'] == st.session_state.selected_code].sort_values('misurata')
    current_mis = st.session_state.misurate_sel if st.session_state.misurate_sel else all_mis
    pt = pt[pt['misurata'].isin(current_mis)].copy()

    if len(pt) == 0:
        st.info("Il punto non ha misurate nella selezione corrente.")
    else:
        base_x = pt.iloc[0]['X']; base_y = pt.iloc[0]['Y']
        pt['cum_dX'] = pt['X'] - base_x
        pt['cum_dY'] = pt['Y'] - base_y
        pt['step_dX'] = pt['cum_dX'].diff().fillna(pt['cum_dX'])
        pt['step_dY'] = pt['cum_dY'].diff().fillna(pt['cum_dY'])

        cc1, cc2 = st.columns(2)
        with cc1:
            fig_vec = go.Figure()
            fig_vec.add_trace(go.Scatter(x=[0], y=[0], mode='markers', name='Origine'))
            fig_vec.add_trace(go.Scatter(x=pt['cum_dX'], y=pt['cum_dY'], mode='lines+markers', name='Cumulativo', showlegend=True))
            prev_x, prev_y = 0.0, 0.0
            for _, row in pt.iterrows():
                x1, y1 = row['cum_dX'], row['cum_dY']
                mis_label = f"Misurata {int(row['misurata'])}"
                fig_vec.add_trace(go.Scatter(x=[prev_x, x1], y=[prev_y, y1], mode='lines+markers', name=mis_label, showlegend=True))
                prev_x, prev_y = x1, y1
            fig_vec.update_layout(title=f"Spostamenti nel tempo – {st.session_state.selected_code}", xaxis_title="ΔX (m)", yaxis_title="ΔY (m)", margin=dict(l=10, r=10, t=40, b=10), legend=dict(yanchor='top', y=1, xanchor='left', x=1.02))
            fig_vec.update_yaxes(scaleanchor="x", scaleratio=1)
            fig_vec.update_xaxes(constrain='domain')
            st.plotly_chart(fig_vec, use_container_width=True)

        with cc2:
            fig_vel = go.Figure()
            fig_vel.add_trace(go.Scatter(x=pt['data'], y=pt['d2D'], mode='lines+markers', name='d2D (m) – rif. globale'))
            if pt['dZ'].notna().any():
                fig_vel.add_trace(go.Scatter(x=pt['data'], y=pt['dZ'], mode='lines+markers', name='dZ (m) – rif. globale'))
            fig_vel.add_trace(go.Scatter(x=pt['data'], y=pt['d3D'], mode='lines+markers', name='d3D (m) – rif. globale'))
            fig_vel.update_layout(title=f"Velocità degli spostamenti – {st.session_state.selected_code}", xaxis_title="Data (mese/anno)", yaxis_title="Valore (m)", margin=dict(l=10, r=10, t=40, b=10))
            fig_vel.update_xaxes(tickformat="%b %Y")
            st.plotly_chart(fig_vel, use_container_width=True)

st.divider()
