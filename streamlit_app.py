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
    df0 = df0.loc[:, [c for c in df0.columns if str(c).strip().lower() != 'unnamed: 0']]

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
# UI
# -----------------------------

st.set_page_config(page_title="Monitoraggio Topografico", layout="wide")

with st.sidebar:
    st.header("Caricamento")
    up = st.file_uploader("Carica file (.xlsx, .xls, .csv, .txt)", type=["xlsx","xls","csv","txt"], accept_multiple_files=False)
    st.caption("Nuovo formato supportato: prima riga 'TITOLO;SISTEMA' (SISTEMA ∈ {Sistema Locale, UTM32, UTM33}); seconda riga = header con 'Codice;Descrizione;Tipologia;data_0;X_0;Y_0;Z_0;...'.")

    st.divider()
    st.header("Impostazioni")
    st.caption("Seleziona le misurate da analizzare. Modalità predefinita: Intero periodo.")

if up is None:
    st.title("📐 STUDIO SACCHIN - MONITORAGGIO TOPOGRAFICO")
    st.info("🔼 Carica un file per iniziare.")
    st.stop()

# Parsing
try:
    df_long, meta = parse_upload(up)
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

    ref_m = st.selectbox("Misurata di riferimento (globale)", options=all_mis,
                         index=0, format_func=lambda m: f"Misurata {m} – {mis_to_date[m].date()}")

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

# KPI veloci
col_k1, col_k2, col_k3, col_k4 = st.columns(4)
col_k1.metric("N. punti", int(res_f['codice'].nunique()))
col_k2.metric("N. misurate", int(res_f['misurata'].nunique()))
col_k3.metric("Max d2D (m)", f"{res_f['d2D'].abs().max():.3f}" if not res_f.empty else "-")
col_k4.metric("Max |dZ| (m)", f"{res_f['dZ'].abs().max():.3f}" if not res_f.empty else "-")

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

# --- Applica filtri ---
if selected_codes:
    comp = comp[comp['codice'].isin(selected_codes)]
if disp_thresh and disp_thresh > 0:
    comp = comp[comp['vettore_2D'].abs() >= disp_thresh]

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

st.download_button(
    "⬇️ Scarica tabella (XLSX)",
    data=_to_xlsx(comp_visible),
    file_name=f"confronto_misurate_{int(ref_tab)}_vs_{int(cmp_tab)}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.divider()

# =========================
# Mappa + Inspector punto (mappa sopra, inspector sotto) – aggiornati
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
        lons, lats = [], []
        for _, row in first_pts.iterrows():
            lon, lat = transformer.transform(float(row['X']), float(row['Y']))
            lons.append(lon); lats.append(lat)
        first_pts = first_pts.assign(lon=lons, lat=lats)

        features = []
        for _, row in first_pts.iterrows():
            features.append({
                'type':'Feature',
                'properties':{'codice': str(row['codice']), 'descrizione': str(row['descrizione']), 'tipologia': (None if pd.isna(row.get('tipologia')) else str(row.get('tipologia')))},
                'geometry':{'type':'Point','coordinates':[row['lon'], row['lat']]}
            })
        geojson = {'type':'FeatureCollection','features':features}

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
            # funzione per creare l'HTML del pallino con codice
            def _label_html(code_str, bg_color):
                code_str = str(code_str)
                if len(code_str) <= 4:
                    # testo dentro al pallino
                    return f"""
                    <div style="
                        pointer-events:none;
                        display:flex; align-items:center; justify-content:center;
                        width:28px; height:28px; border-radius:50%;
                        background:{bg_color}; color:white; font-weight:700; font-size:12px;
                        border:2px solid rgba(0,0,0,0.25); box-shadow:0 0 4px rgba(0,0,0,0.2);
                    ">{code_str}</div>"""
                else:
                    # pallino + testo a lato
                    return f"""
                    <div style="pointer-events:none; display:flex; align-items:center; gap:6px;">
                        <div style="
                            width:14px; height:14px; border-radius:50%;
                            background:{bg_color}; border:2px solid rgba(0,0,0,0.25);
                            box-shadow:0 0 4px rgba(0,0,0,0.2);
                        "></div>
                        <span style="
                            padding:2px 6px; background:rgba(255,255,255,0.9);
                            border-radius:6px; border:1px solid rgba(0,0,0,0.15);
                            font-weight:600; font-size:12px; color:#111;
                        ">{code_str}</span>
                    </div>"""

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

    st.subheader("Inspector punto")

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
    current_mis = st.session_state.misurate_sel if st.session_state.misurate_sel else all_mis
    pt = pt[pt['misurata'].isin(current_mis)].copy()

    if len(pt) == 0:
        st.info("Il punto non ha misurate nella selezione corrente.")
    else:
        # Base locale sul PRIMO campione della selezione corrente
        base_x = pt.iloc[0]['X']; base_y = pt.iloc[0]['Y']; base_z = pt.iloc[0]['Z']

        # CUMULATE rispetto alla prima data della selezione – tutte partono da 0
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
            st.plotly_chart(fig_vec, use_container_width=True)

        # ---------- Velocità / Componenti degli spostamenti ----------
        with cc2:
            coord_sys = meta.get('coord_sys', None)
            if coord_sys in ('UTM32', 'UTM33', 'UTM'):                                   # >>> MOD
                lblX, lblY, lblZ = "Est", "Nord", "Quota"
            else:
                lblX, lblY, lblZ = "X", "Y", "Z"

            fig_vel = go.Figure()
            # COLORI richiesti: X/Est=Rosso, Y/Nord=Verde, Z/Quota=Blu                 # >>> MOD
            color_x = "red"
            color_y = "green"
            color_z = "blue"

            # Rinomina grafico: "Velocità degli spostamenti"                            # >>> MOD
            fig_vel.add_trace(go.Scatter(
                x=pt['data'], y=pt['cum_dX'], mode='lines+markers',
                name=f"Δ{lblX} (m)",
                line=dict(color=color_x), marker=dict(color=color_x)
            ))
            fig_vel.add_trace(go.Scatter(
                x=pt['data'], y=pt['cum_dY'], mode='lines+markers',
                name=f"Δ{lblY} (m)",
                line=dict(color=color_y), marker=dict(color=color_y)
            ))
            if pt['cum_dZ'].notna().any():
                fig_vel.add_trace(go.Scatter(
                    x=pt['data'], y=pt['cum_dZ'], mode='lines+markers',
                    name=f"Δ{lblZ} (m)",
                    line=dict(color=color_z), marker=dict(color=color_z)
                ))

            fig_vel.update_layout(
                title=f"Velocità degli spostamenti – punto {st.session_state.selected_code}",  # >>> MOD
                xaxis_title="Data", yaxis_title="Variazione (m)",
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation='h', yanchor='bottom', y=0, xanchor='center', x=0.5)
            )
            fig_vel.update_xaxes(tickformat="%b %Y")
            st.plotly_chart(fig_vel, use_container_width=True)

        # ---------- Tabella dettaglio misurate del punto ----------
        sel_code = st.session_state.selected_code
        st.markdown(f"#### Dati misurate del punto {sel_code}")

        # ricostruisco i dati del punto selezionato e ordino per data crescente
        sel_code = st.session_state.selected_code
        pt_tab = res[res['codice'].astype(str) == str(sel_code)].copy()

        # se hai un filtro misurate attivo, manteniamolo coerente con i grafici
        current_mis = st.session_state.misurate_sel if st.session_state.misurate_sel else None
        if current_mis:
            pt_tab = pt_tab[pt_tab['misurata'].isin(current_mis)]

        pt_tab['data'] = pd.to_datetime(pt_tab['data'])
        pt_tab = pt_tab.sort_values('data')

        if len(pt_tab) == 0:
            st.info("Nessuna misurata disponibile per il punto selezionato nella selezione corrente.")
        else:
            # riferimento: prima misurata disponibile
            base_x = pt_tab.iloc[0]['X']
            base_y = pt_tab.iloc[0]['Y']
            has_z  = 'Z' in pt_tab.columns and pt_tab['Z'].notna().any()
            base_z = pt_tab.iloc[0]['Z'] if has_z else np.nan

            # Δ rispetto alla prima misurata
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

            st.download_button(
                "⬇️ Scarica tabella (XLSX)",
                data=_to_xlsx_point(df_show),
                file_name=f"dati_punto_{sel_code}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )        
            
st.divider()
