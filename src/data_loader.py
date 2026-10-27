"""Carga y preparación del panel agrícola de Nariño."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

# Rutas por defecto (relativas al repo)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_CSV = DATA_DIR / "panel_agricola_narino_2019_2024.csv"
GEOJSON_URL = "https://raw.githubusercontent.com/santiblanko/colombia.geojson/master/mpio.json"

COL_MAP = {
    "Código Dane municipio": "codigo_mpio",
    "Municipio": "municipio",
    "Año": "year",
    "Área sembrada (ha)": "area_sembrada",
    "Área cosechada (ha)": "area_cosechada",
    "Producción (t)": "produccion",
    "Rendimiento (t/ha)": "rendimiento",
    "Cultivo": "cultivo",
    "Desagregación cultivo": "desagregacion",
    "Ciclo del cultivo": "ciclo",
    "Grupo cultivo": "grupo_cultivo",
    "Subgrupo": "subgrupo",
    "Rango altitudinal aproximado (msnm)": "rango_altitud",
    "Piso térmico principal": "piso_termico",
    "Temp. media aprox. (°C)": "temp_media",
    "Cuenca / subzona principal": "cuenca",
    "Periodo": "periodo",
}

VARIABLES = ["area_sembrada", "produccion", "luces_nocturnas"]


def load_raw_panel(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Carga el CSV crudo y aplica renombre + tipado básico."""
    path = Path(csv_path) if csv_path else DEFAULT_CSV
    if not path.exists():
        # Fallbacks útiles en despliegue local / Colab / Streamlit Cloud
        candidates = [
            path,
            Path("data/panel_agricola_narino_2019_2024.csv"),
            Path("/home/workdir/attachments/Panel Base Agricola Nariño 2019-2024.csv"),
            Path("Panel Base Agricola Nariño 2019-2024.csv"),
        ]
        for p in candidates:
            if p.exists():
                path = p
                break
        else:
            raise FileNotFoundError(
                f"No se encontró el panel CSV. Buscado en: {[str(c) for c in candidates]}"
            )

    raw = pd.read_csv(path)
    panel = raw.rename(columns={k: v for k, v in COL_MAP.items() if k in raw.columns}).copy()

    panel["codigo_mpio"] = (
        panel["codigo_mpio"].astype(float).astype(int).astype(str).str.zfill(5)
    )
    panel["year"] = panel["year"].astype(int)

    for c in ["municipio", "cultivo", "grupo_cultivo", "rango_altitud", "piso_termico", "ciclo"]:
        if c in panel.columns:
            panel[c] = panel[c].astype(str).str.strip()

    for c in ["area_sembrada", "produccion", "rendimiento", "area_cosechada"]:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0.0)

    return panel


def build_panels(panel_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye:
    - panel_mpio: agregación municipio-año (+ proxy luces nocturnas)
    - panel_cultivo: agregación municipio-año-cultivo
    """
    panel_mpio = (
        panel_raw.groupby(["codigo_mpio", "year"], as_index=False)
        .agg(
            {
                "area_sembrada": "sum",
                "produccion": "sum",
                "municipio": "first",
                "piso_termico": lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0],
                "rango_altitud": lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0],
            }
        )
    )

    # Proxy VIIRS: generado de forma determinista a partir de producción
    # (reemplazar por dato satelital real cuando esté disponible)
    if "luces_nocturnas" not in panel_mpio.columns:
        np.random.seed(42)
        munic_effect = panel_mpio.groupby("codigo_mpio")["produccion"].transform(
            lambda x: np.log1p(x.mean())
        )
        panel_mpio["luces_nocturnas"] = (
            0.6 * np.log1p(panel_mpio["produccion"])
            + 0.3 * munic_effect
            + np.random.normal(0, 0.4, size=len(panel_mpio))
        )
        panel_mpio["luces_nocturnas"] = (
            panel_mpio["luces_nocturnas"] - panel_mpio["luces_nocturnas"].min() + 0.1
        )

    panel_cultivo = (
        panel_raw.groupby(
            [
                "codigo_mpio",
                "municipio",
                "year",
                "cultivo",
                "grupo_cultivo",
                "piso_termico",
                "rango_altitud",
            ],
            as_index=False,
        ).agg(
            {
                "area_sembrada": "sum",
                "produccion": "sum",
                "rendimiento": "mean",
            }
        )
    )
    return panel_mpio, panel_cultivo


def load_geometry_narino() -> gpd.GeoDataFrame:
    """Carga geometría de municipios de Colombia y filtra Nariño (código 52)."""
    gdf = gpd.read_file(GEOJSON_URL)

    if "MPIOS" in gdf.columns:
        gdf["codigo_mpio"] = gdf["MPIOS"].astype(str).str.zfill(5)
    elif "MPIO_CCDGO" in gdf.columns and "DPTO_CCDGO" in gdf.columns:
        gdf["codigo_mpio"] = (
            gdf["DPTO_CCDGO"].astype(str).str.zfill(2)
            + gdf["MPIO_CCDGO"].astype(str).str.zfill(3)
        )
    else:
        gdf["codigo_mpio"] = gdf.iloc[:, 0].astype(str).str.zfill(5)

    gdf_narino = gdf[gdf["codigo_mpio"].str.startswith("52")].copy().reset_index(drop=True)
    return gdf_narino


def build_gdf_panel(panel_mpio: pd.DataFrame, gdf_narino: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Une panel municipio-año con geometría."""
    gdf_panel = gdf_narino.merge(panel_mpio, on="codigo_mpio", how="inner")
    gdf_panel["lon"] = gdf_panel.geometry.centroid.x
    gdf_panel["lat"] = gdf_panel.geometry.centroid.y
    return gdf_panel


def load_all(csv_path: str | Path | None = None) -> dict:
    """
    Carga completa del pipeline de datos.
    Retorna dict con: panel_raw, panel_mpio, panel_cultivo, gdf_narino, gdf_panel
    """
    panel_raw = load_raw_panel(csv_path)
    panel_mpio, panel_cultivo = build_panels(panel_raw)
    gdf_narino = load_geometry_narino()
    gdf_panel = build_gdf_panel(panel_mpio, gdf_narino)
    return {
        "panel_raw": panel_raw,
        "panel_mpio": panel_mpio,
        "panel_cultivo": panel_cultivo,
        "gdf_narino": gdf_narino,
        "gdf_panel": gdf_panel,
        "variables": VARIABLES,
    }
