"""ESDA: selección de W, Moran global, Geary, LISA."""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen, Rook, KNN, lag_spatial
from esda.moran import Moran, Moran_Local
from esda.geary import Geary

from .data_loader import VARIABLES

LISA_COLORS = {
    "HH": "#d7191c",
    "LL": "#2c7bb6",
    "HL": "#fdae61",
    "LH": "#abd9e9",
    "NS": "#e0e0e0",
}


def select_optimal_W(
    gdf_ref: gpd.GeoDataFrame,
    variables: list[str] | None = None,
) -> tuple:
    """
    Evalúa Queen, Rook y KNN (k=3..8) con Moran Global.
    Criterio: máximo n de variables con p<0.05; empate → mayor |I| medio.
    Retorna: (W, nombre_matriz, df_resultados, df_sig)
    """
    variables = variables or VARIABLES
    candidatos = {
        "Queen": Queen.from_dataframe(gdf_ref, use_index=False),
        "Rook": Rook.from_dataframe(gdf_ref, use_index=False),
    }
    for k in range(3, 9):
        candidatos[f"KNN_{k}"] = KNN.from_dataframe(gdf_ref, k=k, use_index=False)

    resultados_w = []
    for nombre_w, w in candidatos.items():
        w.transform = "r"
        for var in variables:
            y = gdf_ref[var].values.astype(float)
            if np.nanstd(y) < 1e-12:
                continue
            mi = Moran(y, w)
            resultados_w.append(
                {
                    "Matriz": nombre_w,
                    "Variable": var,
                    "Moran_I": mi.I,
                    "p_value": mi.p_sim,
                    "Significativo": mi.p_sim < 0.05,
                }
            )

    df_w = pd.DataFrame(resultados_w)
    df_sig = (
        df_w[df_w["Significativo"]]
        .groupby("Matriz")
        .agg(
            n_sig=("Significativo", "sum"),
            mean_abs_I=("Moran_I", lambda x: np.mean(np.abs(x))),
        )
        .reset_index()
    )

    if len(df_sig):
        mejor = df_sig.sort_values(["n_sig", "mean_abs_I"], ascending=False).iloc[0]["Matriz"]
    else:
        tmp = (
            df_w.groupby("Matriz")["Moran_I"]
            .apply(lambda x: np.mean(np.abs(x)))
            .reset_index()
        )
        tmp.columns = ["Matriz", "mean_abs_I"]
        mejor = tmp.sort_values("mean_abs_I", ascending=False).iloc[0]["Matriz"]

    W = candidatos[mejor]
    W.transform = "r"
    return W, mejor, df_w, df_sig


def global_indices(gdf_ref: gpd.GeoDataFrame, W, variables: list[str] | None = None) -> pd.DataFrame:
    """Moran I y Geary C globales para cada variable."""
    variables = variables or VARIABLES
    rows = []
    for var in variables:
        y = gdf_ref[var].values.astype(float)
        mi = Moran(y, W)
        gc = Geary(y, W)
        rows.append(
            {
                "Variable": var,
                "Moran_I": mi.I,
                "Moran_p": mi.p_sim,
                "Geary_C": gc.C,
                "Geary_p": gc.p_sim,
            }
        )
    return pd.DataFrame(rows)


def calcular_lisa(
    gdf_in: gpd.GeoDataFrame,
    var: str,
    W,
    p_threshold: float = 0.05,
) -> tuple[gpd.GeoDataFrame, Moran_Local | None]:
    """Moran Local + etiquetas de cluster HH/LL/HL/LH/NS."""
    y = gdf_in[var].values.astype(float)
    if np.nanstd(y) < 1e-12:
        gdf_out = gdf_in.copy()
        gdf_out["lisa_I"] = 0.0
        gdf_out["lisa_p"] = 1.0
        gdf_out["lisa_q"] = 0
        gdf_out["lisa_cluster"] = "NS"
        return gdf_out, None

    ml = Moran_Local(y, W, permutations=999, seed=42)
    gdf_out = gdf_in.copy()
    gdf_out["lisa_I"] = ml.Is
    gdf_out["lisa_p"] = ml.p_sim
    gdf_out["lisa_q"] = ml.q
    labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    gdf_out["lisa_cluster"] = [
        labels.get(int(q), "NS") if p < p_threshold else "NS"
        for q, p in zip(ml.q, ml.p_sim)
    ]
    return gdf_out, ml


def lisa_for_year(
    gdf_panel: gpd.GeoDataFrame,
    gdf_ref: gpd.GeoDataFrame,
    W,
    variable: str,
    anio: int,
    p_threshold: float = 0.05,
) -> tuple[gpd.GeoDataFrame, float]:
    """
    Calcula LISA para un año dado, alineando IDs con gdf_ref (matriz W).
    Retorna (gdf con clusters, Moran I global del año).
    """
    ids_ref = gdf_ref["codigo_mpio"].values
    gdf_yr = gdf_panel[gdf_panel["year"] == anio].copy()
    gdf_yr = gdf_yr.set_index("codigo_mpio").reindex(ids_ref).reset_index()

    if gdf_yr.geometry.isna().any():
        geom_map = gdf_ref.set_index("codigo_mpio")["geometry"]
        gdf_yr["geometry"] = gdf_yr["codigo_mpio"].map(geom_map)
        gdf_yr = gpd.GeoDataFrame(gdf_yr, geometry="geometry", crs=gdf_ref.crs)

    y = gdf_yr[variable].values.astype(float)
    if np.isnan(y).any():
        y = np.where(np.isnan(y), np.nanmean(y), y)

    if np.nanstd(y) < 1e-12:
        gdf_yr["lisa_cluster"] = "NS"
        gdf_yr["lisa_I"] = 0.0
        gdf_yr["lisa_p"] = 1.0
        return gdf_yr, 0.0

    ml = Moran_Local(y, W, permutations=999, seed=42)
    labels_map = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    cluster = [
        labels_map.get(int(q), "NS") if p < float(p_threshold) else "NS"
        for q, p in zip(ml.q, ml.p_sim)
    ]
    gdf_yr = gdf_yr.copy()
    gdf_yr["lisa_cluster"] = cluster
    gdf_yr["lisa_I"] = ml.Is
    gdf_yr["lisa_p"] = ml.p_sim
    gdf_yr[variable] = y

    # Municipio desde referencia si falta
    if "municipio" in gdf_yr.columns:
        gdf_yr["municipio"] = gdf_yr["municipio"].fillna(
            gdf_yr["codigo_mpio"].map(gdf_ref.set_index("codigo_mpio")["municipio"])
        )

    mi_global = Moran(y, W).I
    return gdf_yr, mi_global
