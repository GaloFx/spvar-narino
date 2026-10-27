"""
SpVAR + ESDA · Dashboard Nariño (Streamlit)
==========================================
Área sembrada · Producción · Luces nocturnas (proxy VIIRS)
Municipio · Cultivo · Rango altitudinal · Piso térmico · Año
LISA · Selección óptima de W · Cuatro mecanismos · PDF

Mecanismos:
  φ  Autodinámica espacial (temporal propia)
  β  Cruce local
  ρ  Relación espacial directa
  θ  Relación espacial cruzada
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from libpysal.weights import lag_spatial
from plotly.subplots import make_subplots
from scipy.interpolate import griddata

# Asegurar imports del paquete src
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import VARIABLES, load_all
from src.pdf_report import generar_reporte_pdf
from src.spatial_analysis import (
    LISA_COLORS,
    calcular_lisa,
    global_indices,
    lisa_for_year,
    select_optimal_W,
)
from src.spvar import (
    auditoria_spvar,
    calcular_todos_impactos,
    estimar_spvar_completo,
    generar_interpretacion,
    prepare_spvar_panel,
)

# -----------------------------------------------------------------------------
# Configuración de página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SpVAR Nariño · ESDA",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos mínimos de alto contraste
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; }
    h1, h2, h3 { color: #0f172a; }
    .stMetric label { color: #475569 !important; }
    div[data-testid="stMetricValue"] { color: #0f172a; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Carga de datos y estimación (cacheado)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando panel y geometría…")
def bootstrap():
    data = load_all()
    gdf_panel = data["gdf_panel"]
    year_ref = int(gdf_panel["year"].max())
    gdf_ref = gdf_panel[gdf_panel["year"] == year_ref].copy().reset_index(drop=True)

    W, mejor_matriz, df_w, df_sig = select_optimal_W(gdf_ref, VARIABLES)
    df_globales = global_indices(gdf_ref, W, VARIABLES)

    panel_spvar = prepare_spvar_panel(gdf_panel, W, gdf_ref)
    modelos, mecanismos_all = estimar_spvar_completo(panel_spvar, VARIABLES)
    impactos = calcular_todos_impactos(modelos, W, VARIABLES)
    audit = auditoria_spvar(panel_spvar, W, modelos, VARIABLES)

    lisa_cache = {}
    for var in VARIABLES:
        gdf_l, ml = calcular_lisa(gdf_ref, var, W)
        lisa_cache[var] = {"gdf": gdf_l, "ml": ml}

    return {
        **data,
        "year_ref": year_ref,
        "gdf_ref": gdf_ref,
        "W": W,
        "mejor_matriz": mejor_matriz,
        "df_w": df_w,
        "df_sig": df_sig,
        "df_globales": df_globales,
        "panel_spvar": panel_spvar,
        "modelos": modelos,
        "mecanismos_all": mecanismos_all,
        "impactos": impactos,
        "audit": audit,
        "lisa_cache": lisa_cache,
    }


# -----------------------------------------------------------------------------
# Helpers de filtros y gráficos
# -----------------------------------------------------------------------------
def filtrar_cultivo(df, municipio, cultivo, grupo, piso, rango, anio):
    d = df.copy()
    if municipio and municipio != "Todos":
        d = d[d["municipio"] == municipio]
    if cultivo and cultivo != "Todos":
        d = d[d["cultivo"] == cultivo]
    if grupo and grupo != "Todos":
        d = d[d["grupo_cultivo"] == grupo]
    if piso and piso != "Todos":
        d = d[d["piso_termico"] == piso]
    if rango and rango != "Todos":
        d = d[d["rango_altitud"] == rango]
    if anio and str(anio) != "Todos":
        d = d[d["year"] == int(anio)]
    return d


def mapa_coropletico(gdf_panel, variable, anio, piso="Todos", rango="Todos"):
    gdf_yr = gdf_panel[gdf_panel["year"] == int(anio)].copy()
    if piso and piso != "Todos":
        gdf_yr = gdf_yr[gdf_yr["piso_termico"] == piso]
    if rango and rango != "Todos":
        gdf_yr = gdf_yr[gdf_yr["rango_altitud"] == rango]
    if gdf_yr.empty:
        return None
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    cmap = {
        "area_sembrada": "YlGn",
        "produccion": "Spectral_r",
        "luces_nocturnas": "Inferno",
    }.get(variable, "viridis")
    gdf_yr.plot(
        column=variable,
        cmap=cmap,
        scheme="Quantiles",
        k=6,
        legend=True,
        ax=ax,
        edgecolor="black",
        linewidth=0.4,
        legend_kwds={"loc": "lower left", "title": variable, "fmt": "{:.1f}"},
    )
    ax.set_title(f'{variable.replace("_", " ").title()} · Año {anio}', fontsize=13, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()
    return fig


def lisa_mapa(gdf_panel, gdf_ref, W, variable, anio, p_threshold=0.05):
    gdf_yr, mi = lisa_for_year(gdf_panel, gdf_ref, W, variable, int(anio), p_threshold)
    y = gdf_yr[variable].values.astype(float)
    cluster = gdf_yr["lisa_cluster"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    ax = axes[0]
    for cat, color in LISA_COLORS.items():
        subset = gdf_yr[gdf_yr["lisa_cluster"] == cat]
        if len(subset):
            subset.plot(ax=ax, color=color, edgecolor="black", linewidth=0.4, label=cat)
    ax.set_title(f"LISA · {variable} · {anio} (p < {p_threshold})", fontsize=12, fontweight="bold")
    ax.set_axis_off()
    ax.legend(loc="lower left", frameon=True, fontsize=9)

    ax2 = axes[1]
    lag_y = lag_spatial(W, y)
    y_std = (y - y.mean()) / (y.std() + 1e-12)
    lag_std = (lag_y - lag_y.mean()) / (lag_y.std() + 1e-12)
    ax2.scatter(
        y_std,
        lag_std,
        c=[LISA_COLORS.get(c, "#999") for c in cluster],
        edgecolors="k",
        linewidths=0.3,
        s=40,
        alpha=0.85,
    )
    ax2.axhline(0, color="grey", lw=0.8)
    ax2.axvline(0, color="grey", lw=0.8)
    ax2.set_xlabel("Variable (estandarizada)")
    ax2.set_ylabel("Lag espacial (estandarizado)")
    ax2.set_title(f"Moran Scatter · I = {mi:.3f}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    counts = gdf_yr["lisa_cluster"].value_counts()
    hh = (
        gdf_yr[gdf_yr["lisa_cluster"] == "HH"][["municipio", "lisa_I", "lisa_p", variable]]
        .sort_values("lisa_I", ascending=False)
    )
    ll = (
        gdf_yr[gdf_yr["lisa_cluster"] == "LL"][["municipio", "lisa_I", "lisa_p", variable]]
        .sort_values("lisa_I")
    )
    meaning = {
        "HH": "Alto-Alto (concentración de valores altos)",
        "LL": "Bajo-Bajo (concentración de valores bajos)",
        "HL": "Alto rodeado de bajos (outlier)",
        "LH": "Bajo rodeado de altos (outlier)",
        "NS": "No significativo",
    }
    md = [f"### Resumen LISA · **{variable}** · Año **{anio}** (p < {p_threshold})\n"]
    md.append("| Cluster | N | Significado |")
    md.append("|---------|---|-------------|")
    for cat in ["HH", "LL", "HL", "LH", "NS"]:
        md.append(f"| **{cat}** | {int(counts.get(cat, 0))} | {meaning[cat]} |")
    if len(hh):
        md.append("\n**Clusters HH (prioridad para intervención coordinada):**")
        for _, r in hh.head(8).iterrows():
            md.append(
                f"- {r['municipio']}: I={r['lisa_I']:.3f}, p={r['lisa_p']:.3f}, valor={r[variable]:,.1f}"
            )
    if len(ll):
        md.append("\n**Clusters LL:**")
        for _, r in ll.head(5).iterrows():
            md.append(
                f"- {r['municipio']}: I={r['lisa_I']:.3f}, p={r['lisa_p']:.3f}, valor={r[variable]:,.1f}"
            )
    return fig, "\n".join(md)


def ranking_cultivos(panel_cultivo, municipio, grupo, piso, rango, anio, top_n=12):
    d = filtrar_cultivo(panel_cultivo, municipio, "Todos", grupo, piso, rango, anio)
    if d.empty:
        return None
    rank = (
        d.groupby("cultivo", as_index=False)
        .agg(area=("area_sembrada", "sum"), produccion=("produccion", "sum"))
        .sort_values("produccion", ascending=False)
        .head(int(top_n))
    )
    fig = px.bar(
        rank,
        x="produccion",
        y="cultivo",
        orientation="h",
        color="area",
        color_continuous_scale="Teal",
        labels={"produccion": "Producción (t)", "cultivo": "Cultivo", "area": "Área (ha)"},
        title=f'Top cultivos · {anio if str(anio) != "Todos" else "2019-2024"}',
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=480,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#0f172a", size=12),
    )
    return fig


def evolucion(panel_cultivo, municipio, cultivo, grupo, piso, rango):
    d = filtrar_cultivo(panel_cultivo, municipio, cultivo, grupo, piso, rango, "Todos")
    if d.empty:
        return None
    serie = (
        d.groupby("year", as_index=False)
        .agg(area=("area_sembrada", "sum"), produccion=("produccion", "sum"))
        .sort_values("year")
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=serie["year"], y=serie["area"], name="Área (ha)", marker_color="#0369a1"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=serie["year"],
            y=serie["produccion"],
            name="Producción (t)",
            mode="lines+markers",
            line=dict(color="#b45309", width=3),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Evolución área vs producción",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(color="#0f172a"),
    )
    fig.update_yaxes(title_text="Área (ha)", secondary_y=False)
    fig.update_yaxes(title_text="Producción (t)", secondary_y=True)
    return fig


def piso_termico_chart(panel_cultivo, anio, grupo="Todos"):
    d = panel_cultivo.copy()
    if anio and str(anio) != "Todos":
        d = d[d["year"] == int(anio)]
    if grupo and grupo != "Todos":
        d = d[d["grupo_cultivo"] == grupo]
    if d.empty:
        return None
    agg = (
        d.groupby("piso_termico", as_index=False)
        .agg(produccion=("produccion", "sum"), area=("area_sembrada", "sum"))
        .sort_values("produccion", ascending=False)
    )
    fig = px.pie(
        agg,
        values="produccion",
        names="piso_termico",
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=f'Producción por piso térmico · {anio if str(anio) != "Todos" else "2019-2024"}',
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        font=dict(color="#0f172a"),
    )
    return fig


def scatter_3d(gdf_panel, anio):
    df = gdf_panel[gdf_panel["year"] == int(anio)].copy()
    if df.empty:
        return None
    df["crecimiento"] = np.log1p(df["produccion"]) - np.log1p(df["produccion"].mean())
    df["crecimiento_pos"] = df["crecimiento"] - df["crecimiento"].min() + 0.5
    fig = px.scatter_3d(
        df,
        x="lon",
        y="lat",
        z="luces_nocturnas",
        color="area_sembrada",
        size="crecimiento_pos",
        hover_name="municipio",
        hover_data=["produccion", "piso_termico", "rango_altitud"],
        color_continuous_scale="Turbo",
        size_max=22,
        title=f"Dispersión 3D · Año {anio}",
        labels={"lon": "Longitud", "lat": "Latitud", "luces_nocturnas": "Luces"},
    )
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="white",
        font=dict(color="#0f172a"),
    )
    return fig


def superficie_3d(gdf_panel, anio):
    df = gdf_panel[gdf_panel["year"] == int(anio)].copy()
    if df.empty:
        return None
    x, y, z = df["lon"].values, df["lat"].values, df["luces_nocturnas"].values
    xi = np.linspace(x.min(), x.max(), 70)
    yi = np.linspace(y.min(), y.max(), 70)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method="cubic")
    fig = go.Figure(
        data=[go.Surface(x=xi, y=yi, z=zi, colorscale="Inferno", colorbar=dict(title="Luces"))]
    )
    fig.update_layout(
        title=f"Superficie 3D luces · Año {anio}",
        scene=dict(xaxis_title="Longitud", yaxis_title="Latitud", zaxis_title="Luces"),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="white",
        font=dict(color="#0f172a"),
    )
    return fig


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    st.title("SpVAR + ESDA · Nariño")
    st.caption(
        "Mecanismos φ, β, ρ, θ · Selección óptima de W · LISA · Impactos espaciales · PDF  \n"
        "Filtros: Municipio · Cultivo · Rango altitudinal · Piso térmico · Año"
    )

    with st.spinner("Inicializando análisis espacial y SpVAR…"):
        ctx = bootstrap()

    panel_cultivo = ctx["panel_cultivo"]
    gdf_panel = ctx["gdf_panel"]
    gdf_ref = ctx["gdf_ref"]
    W = ctx["W"]
    mejor_matriz = ctx["mejor_matriz"]
    df_globales = ctx["df_globales"]
    mecanismos_all = ctx["mecanismos_all"]
    impactos = ctx["impactos"]
    audit = ctx["audit"]
    lisa_cache = ctx["lisa_cache"]
    year_ref = ctx["year_ref"]
    modelos = ctx["modelos"]
    panel_spvar = ctx["panel_spvar"]

    # --- Sidebar filtros ---
    st.sidebar.header("Filtros territoriales")
    municipios = ["Todos"] + sorted(panel_cultivo["municipio"].dropna().unique().tolist())
    cultivos = ["Todos"] + sorted(panel_cultivo["cultivo"].dropna().unique().tolist())
    grupos = ["Todos"] + sorted(panel_cultivo["grupo_cultivo"].dropna().unique().tolist())
    pisos = ["Todos"] + sorted(panel_cultivo["piso_termico"].dropna().unique().tolist())
    rangos = ["Todos"] + sorted(panel_cultivo["rango_altitud"].dropna().unique().tolist())
    anos = ["Todos"] + sorted([int(y) for y in panel_cultivo["year"].unique()])
    anos_mapa = sorted([int(y) for y in gdf_panel["year"].unique()])

    f_municipio = st.sidebar.selectbox("Municipio", municipios, index=0)
    f_cultivo = st.sidebar.selectbox("Cultivo", cultivos, index=0)
    f_grupo = st.sidebar.selectbox("Grupo de cultivo", grupos, index=0)
    f_piso = st.sidebar.selectbox("Piso térmico", pisos, index=0)
    f_rango = st.sidebar.selectbox("Rango altitudinal (msnm)", rangos, index=0)
    default_anio = anos[-1] if anos[-1] != "Todos" else 2024
    f_anio = st.sidebar.selectbox("Año", anos, index=anos.index(default_anio) if default_anio in anos else 0)

    # --- KPIs ---
    d_kpi = filtrar_cultivo(panel_cultivo, f_municipio, f_cultivo, f_grupo, f_piso, f_rango, f_anio)
    if d_kpi.empty:
        st.warning("Sin datos para los filtros seleccionados.")
        area, prod, rend, n_mpio, n_cult = 0, 0, 0, 0, 0
    else:
        area = d_kpi["area_sembrada"].sum()
        prod = d_kpi["produccion"].sum()
        rend = d_kpi["rendimiento"].mean()
        n_mpio = d_kpi["municipio"].nunique()
        n_cult = d_kpi["cultivo"].nunique()

    st.markdown(
        f"**Vista filtrada** · {f_anio if str(f_anio) != 'Todos' else '2019-2024'} · "
        f"{n_mpio} municipios · {n_cult} cultivos · {f_municipio} | {f_cultivo} | {f_piso}"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Área sembrada", f"{area:,.0f} ha")
    c2.metric("Producción", f"{prod:,.0f} t")
    c3.metric("Rendimiento medio", f"{rend:.2f} t/ha")
    c4.metric("Municipios", f"{n_mpio}")

    # --- Tabs ---
    tab_map, tab_lisa, tab_cult, tab_piso, tab_3d, tab_pdf, tab_interp, tab_tech = st.tabs(
        [
            "Mapas",
            "LISA",
            "Cultivos",
            "Piso térmico",
            "3D",
            "PDF",
            "Interpretación",
            "Resumen técnico",
        ]
    )

    with tab_map:
        st.subheader("Mapas coropléticos")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            var_map = st.selectbox("Variable", VARIABLES, index=1, key="var_map")
        with col_b:
            anio_map = st.selectbox("Año", anos_mapa, index=len(anos_mapa) - 1, key="anio_map")
        with col_c:
            piso_map = st.selectbox("Piso térmico", pisos, index=0, key="piso_map")
        if st.button("Generar mapa", type="primary", key="btn_map"):
            fig = mapa_coropletico(gdf_panel, var_map, anio_map, piso_map, f_rango)
            if fig:
                st.pyplot(fig)
            else:
                st.info("Sin datos para el mapa solicitado.")

    with tab_lisa:
        st.subheader("Clusters locales (Moran Local)")
        st.markdown(
            "**HH**: altos rodeados de altos · **LL**: bajos rodeados de bajos · "
            "**HL/LH**: outliers · **NS**: no significativo"
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            lisa_var = st.selectbox("Variable", VARIABLES, index=1, key="lisa_var")
        with col_b:
            lisa_anio = st.selectbox("Año", anos_mapa, index=len(anos_mapa) - 1, key="lisa_anio")
        with col_c:
            lisa_p = st.slider("Umbral p", 0.01, 0.10, 0.05, 0.01, key="lisa_p")
        if st.button("Calcular LISA", type="primary", key="btn_lisa"):
            fig, md = lisa_mapa(gdf_panel, gdf_ref, W, lisa_var, lisa_anio, lisa_p)
            st.pyplot(fig)
            st.markdown(md)

    with tab_cult:
        st.subheader("Ranking y evolución")
        top_n = st.slider("Top N", 5, 25, 12, 1, key="top_n")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Ranking", type="primary", key="btn_rank"):
                fig = ranking_cultivos(
                    panel_cultivo, f_municipio, f_grupo, f_piso, f_rango, f_anio, top_n
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Sin datos.")
        with col2:
            if st.button("Evolución temporal", key="btn_evo"):
                fig = evolucion(panel_cultivo, f_municipio, f_cultivo, f_grupo, f_piso, f_rango)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Sin datos.")

    with tab_piso:
        st.subheader("Distribución por piso térmico")
        col_a, col_b = st.columns(2)
        with col_a:
            anio_piso = st.selectbox("Año", anos, index=0, key="anio_piso")
        with col_b:
            grupo_piso = st.selectbox("Grupo", grupos, index=0, key="grupo_piso")
        if st.button("Distribución", type="primary", key="btn_piso"):
            fig = piso_termico_chart(panel_cultivo, anio_piso, grupo_piso)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sin datos.")

    with tab_3d:
        st.subheader("Visualizaciones 3D")
        anio_3d = st.selectbox("Año", anos_mapa, index=len(anos_mapa) - 1, key="anio_3d")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Scatter 3D", type="primary", key="btn_sc"):
                fig = scatter_3d(gdf_panel, anio_3d)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        with col2:
            if st.button("Superficie 3D", key="btn_surf"):
                fig = superficie_3d(gdf_panel, anio_3d)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    with tab_pdf:
        st.subheader("Reporte PDF por municipio y/o cultivo")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            pdf_mpio = st.selectbox("Municipio", municipios, index=0, key="pdf_mpio")
        with col_b:
            pdf_cult = st.selectbox("Cultivo", cultivos, index=0, key="pdf_cult")
        with col_c:
            pdf_anio = st.selectbox("Año", anos, index=0, key="pdf_anio")
        if st.button("Generar PDF", type="primary", key="btn_pdf"):
            path = generar_reporte_pdf(
                panel_cultivo,
                lisa_cache,
                year_ref,
                mejor_matriz,
                municipio=pdf_mpio,
                cultivo=pdf_cult,
                anio=pdf_anio,
            )
            with open(path, "rb") as f:
                st.download_button(
                    "Descargar PDF",
                    data=f,
                    file_name=Path(path).name,
                    mime="application/pdf",
                )

    with tab_interp:
        st.subheader("Traducción de φ, β, ρ, θ, impactos y Moran/LISA a política")
        if st.button("Generar interpretación", type="primary", key="btn_interp"):
            texto = generar_interpretacion(
                mejor_matriz, df_globales, mecanismos_all, impactos, VARIABLES
            )
            st.markdown(texto)

    with tab_tech:
        st.subheader("Parámetros y auditoría")
        st.markdown(
            f"""
### Parámetros
- Matriz de pesos: **{mejor_matriz}** (evaluación automática Queen / Rook / KNN k=3…8)
- Municipios: **{gdf_ref['codigo_mpio'].nunique()}**
- Años SpVAR: **{sorted(panel_spvar['year'].unique().tolist())}**
- Endógenas: `area_sembrada`, `produccion`, `luces_nocturnas`
- Estimador: OLS HC1
- Mecanismos: φ (autodinámica), β (cruce local), ρ (espacial directa), θ (espacial cruzada)
- Impactos: \( S = (I - \\rho W)^{{-1}} \) (directo, indirecto, total)
- LISA: Moran Local, 999 permutaciones

### Auditoría
"""
        )
        if audit["ok"]:
            st.success("Auditoría exitosa — todos los checks OK.")
        else:
            st.warning(f"Auditoría con {audit['n_errores']} advertencia(s).")

        for ch in audit["checks"]:
            icon = "✅" if ch["ok"] else "⚠️"
            st.markdown(f"{icon} **{ch['check']}**: {ch['msg']}")

        st.markdown("### Índices globales (Moran / Geary)")
        st.dataframe(df_globales.round(4), use_container_width=True)

        st.markdown("### Coeficientes SpVAR por mecanismo")
        for var in VARIABLES:
            mec = mecanismos_all[var]
            with st.expander(f"Ecuación: {var}"):
                st.write(
                    f"φ (autodinámica) = {mec['phi']['coef']:.4f} (p={mec['phi']['p']:.4f})"
                )
                st.write(
                    f"ρ (espacial directa) = {mec['rho']['coef']:.4f} (p={mec['rho']['p']:.4f})"
                )
                for s, d in mec["beta"].items():
                    st.write(f"β ({s}_lag) = {d['coef']:.4f} (p={d['p']:.4f})")
                for s, d in mec["theta"].items():
                    st.write(f"θ (W_{s}) = {d['coef']:.4f} (p={d['p']:.4f})")
                st.write("Impactos:", impactos[var])

        st.markdown(
            """
### Nota VIIRS
Proxy construido a partir de producción. Sustituir por dato satelital municipal cuando esté disponible.
"""
        )


if __name__ == "__main__":
    main()
