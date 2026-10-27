"""Especificación formal SpVAR con cuatro mecanismos de transmisión."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from libpysal.weights import lag_spatial

from .data_loader import VARIABLES


def prepare_spvar_panel(gdf_panel: pd.DataFrame, W, gdf_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Construye rezagos temporales (t-1) y espaciales (W y_t) por variable.
    """
    panel_sorted = gdf_panel.sort_values(["codigo_mpio", "year"]).reset_index(drop=True)

    for var in VARIABLES:
        panel_sorted[f"{var}_lag"] = panel_sorted.groupby("codigo_mpio")[var].shift(1)

    def spatial_lag_by_year(df, var, W_ref, id_col="codigo_mpio"):
        lags = []
        ids = gdf_ref[id_col].values
        for yr in sorted(df["year"].unique()):
            sub = df[df["year"] == yr].set_index(id_col).reindex(ids)
            vals = sub[var].values.astype(float)
            if np.isnan(vals).any():
                vals = np.where(np.isnan(vals), np.nanmean(vals), vals)
            y_lag = lag_spatial(W_ref, vals)
            lags.append(pd.DataFrame({id_col: ids, "year": yr, f"W_{var}": y_lag}))
        return pd.concat(lags, ignore_index=True)

    for var in VARIABLES:
        lag_df = spatial_lag_by_year(panel_sorted, var, W)
        panel_sorted = panel_sorted.merge(lag_df, on=["codigo_mpio", "year"], how="left")

    panel_spvar = panel_sorted.dropna(subset=[f"{v}_lag" for v in VARIABLES]).copy()
    return panel_spvar


def estimar_ecuacion_spvar(df: pd.DataFrame, y_var: str, variables: list[str] | None = None, robust: str = "HC1"):
    """
    Estima una ecuación SpVAR (OLS HC1) y etiqueta coeficientes por mecanismo:
    φ (autodinámica), β (cruce local), ρ (espacial directa), θ (espacial cruzada).
    """
    variables = variables or VARIABLES
    X_cols = []
    for v in variables:
        X_cols += [f"{v}_lag", f"W_{v}"]

    y = df[y_var].astype(float)
    X = sm.add_constant(df[X_cols].astype(float))
    mod = sm.OLS(y, X).fit(cov_type=robust)

    mecanismos = {
        "phi": {},
        "beta": {},
        "rho": {},
        "theta": {},
    }
    for v in variables:
        coef_lag = float(mod.params.get(f"{v}_lag", np.nan))
        p_lag = float(mod.pvalues.get(f"{v}_lag", np.nan))
        coef_W = float(mod.params.get(f"W_{v}", np.nan))
        p_W = float(mod.pvalues.get(f"W_{v}", np.nan))
        if v == y_var:
            mecanismos["phi"] = {"coef": coef_lag, "p": p_lag, "var": v}
            mecanismos["rho"] = {"coef": coef_W, "p": p_W, "var": v}
        else:
            mecanismos["beta"][v] = {"coef": coef_lag, "p": p_lag}
            mecanismos["theta"][v] = {"coef": coef_W, "p": p_W}
    return mod, mecanismos


def estimar_spvar_completo(panel_spvar: pd.DataFrame, variables: list[str] | None = None):
    """Estima las tres ecuaciones SpVAR y retorna modelos + mecanismos."""
    variables = variables or VARIABLES
    modelos = {}
    mecanismos_all = {}
    for var in variables:
        mod, mec = estimar_ecuacion_spvar(panel_spvar, var, variables)
        modelos[var] = mod
        mecanismos_all[var] = mec
    return modelos, mecanismos_all


def calcular_impactos_espaciales(modelo, W, y_var: str) -> dict:
    """Impactos promedio (LeSage & Pace): S = (I − ρW)⁻¹."""
    rho = float(modelo.params.get(f"W_{y_var}", 0.0))
    if abs(rho) >= 0.99:
        rho = np.sign(rho) * 0.99
    n = W.n
    I = np.eye(n)
    Wmat = W.full()[0]
    try:
        S = np.linalg.inv(I - rho * Wmat)
    except np.linalg.LinAlgError:
        S = np.linalg.pinv(I - rho * Wmat)
    directo = float(np.mean(np.diag(S)))
    total = float(np.mean(S.sum(axis=1)))
    indirecto = total - directo
    return {"rho": rho, "directo": directo, "indirecto": indirecto, "total": total}


def calcular_todos_impactos(modelos: dict, W, variables: list[str] | None = None) -> dict:
    variables = variables or VARIABLES
    return {var: calcular_impactos_espaciales(modelos[var], W, var) for var in variables}


def auditoria_spvar(panel: pd.DataFrame, W, modelos: dict, variables: list[str] | None = None) -> dict:
    """
    Auditoría de calidad del SpVAR.
    Retorna dict con ok (bool), mensajes (list) y detalle de checks.
    """
    variables = variables or VARIABLES
    checks = []
    errores = []

    # 1. Sin nulos/infinitos en endógenas
    try:
        for v in variables:
            assert panel[v].notna().all() and np.isfinite(panel[v]).all()
        checks.append({"check": "endogenas_finitas", "ok": True, "msg": "Sin nulos ni infinitos en endógenas."})
    except AssertionError as e:
        errores.append(str(e))
        checks.append({"check": "endogenas_finitas", "ok": False, "msg": str(e)})

    # 2. Estabilidad temporal (max |λ| de diag(φ) < 1)
    try:
        A_diag = [float(modelos[v].params.get(f"{v}_lag", 0.0)) for v in variables]
        max_abs_eig = float(np.max(np.abs(np.linalg.eigvals(np.diag(A_diag)))))
        ok = max_abs_eig < 1.0
        msg = f"Estabilidad temporal: max|λ| = {max_abs_eig:.4f} {'< 1' if ok else '≥ 1 (inestable)'}"
        if not ok:
            errores.append(msg)
        checks.append({"check": "estabilidad_temporal", "ok": ok, "msg": msg, "max_abs_eig": max_abs_eig})
    except Exception as e:
        errores.append(str(e))
        checks.append({"check": "estabilidad_temporal", "ok": False, "msg": str(e)})

    # 3. W row-estandarizada
    try:
        row_sums = np.array(W.full()[0].sum(axis=1)).flatten()
        ok = bool(np.allclose(row_sums, 1.0, atol=1e-5))
        msg = "W row-estandarizada." if ok else "W NO está row-estandarizada."
        if not ok:
            errores.append(msg)
        checks.append({"check": "w_row_std", "ok": ok, "msg": msg})
    except Exception as e:
        errores.append(str(e))
        checks.append({"check": "w_row_std", "ok": False, "msg": str(e)})

    # 4. Dimensiones
    n_mpio = panel["codigo_mpio"].nunique()
    n_years = panel["year"].nunique()
    checks.append(
        {
            "check": "dimensiones",
            "ok": True,
            "msg": f"Dimensiones: n={n_mpio} × T={n_years}",
            "n": n_mpio,
            "T": n_years,
        }
    )

    # 5. |ρ| < 1
    try:
        for v in variables:
            rho = float(modelos[v].params.get(f"W_{v}", 0.0))
            assert abs(rho) < 1.0
        checks.append({"check": "rho_invertible", "ok": True, "msg": "|ρ| < 1 en todas las ecuaciones (invertibilidad de S)."})
    except Exception as e:
        errores.append(str(e))
        checks.append({"check": "rho_invertible", "ok": False, "msg": str(e)})

    return {
        "ok": len(errores) == 0,
        "n_errores": len(errores),
        "errores": errores,
        "checks": checks,
    }


def generar_interpretacion(
    mejor_matriz_nombre: str,
    df_globales: pd.DataFrame,
    mecanismos_all: dict,
    impactos: dict,
    variables: list[str] | None = None,
) -> str:
    """Traducción de resultados a lenguaje de política territorial."""
    variables = variables or VARIABLES
    lineas = []
    lineas.append("## Interpretación de resultados SpVAR y ESDA\n")
    lineas.append(
        f"Matriz de pesos seleccionada: **{mejor_matriz_nombre}** "
        "(criterio: significancia de Moran y magnitud de |I|).\n"
    )

    lineas.append("### 1. Autocorrelación espacial global (Moran I)")
    for _, row in df_globales.iterrows():
        var, I, p = row["Variable"], row["Moran_I"], row["Moran_p"]
        if p < 0.05 and I > 0:
            lineas.append(
                f"- **{var}**: autocorrelación positiva significativa (I={I:.3f}, p={p:.3f}). "
                "Los municipios con valores altos tienden a estar rodeados de municipios con valores altos. "
                "Conviene diseñar intervenciones de alcance subregional, no solo municipal."
            )
        elif p < 0.05 and I < 0:
            lineas.append(
                f"- **{var}**: patrón de dispersión significativo (I={I:.3f}, p={p:.3f}). "
                "Valores altos rodeados de bajos. Las intervenciones deben ser más localizadas."
            )
        else:
            lineas.append(
                f"- **{var}**: no hay evidencia de autocorrelación espacial (I={I:.3f}, p={p:.3f})."
            )

    lineas.append("\n### 2. Mecanismos SpVAR por variable")
    for var in variables:
        mec = mecanismos_all[var]
        lineas.append(f"\n**Ecuación de {var}**")
        phi, pphi = mec["phi"]["coef"], mec["phi"]["p"]
        if pphi < 0.05:
            lineas.append(
                f"- φ (autodinámica) = {phi:.3f} (p={pphi:.3f}): el valor del año anterior en el mismo municipio "
                "predice de forma significativa el presente. Las políticas deben ser sostenidas en el tiempo."
            )
        else:
            lineas.append(f"- φ (autodinámica) = {phi:.3f} (p={pphi:.3f}): baja persistencia temporal propia.")

        rho, prho = mec["rho"]["coef"], mec["rho"]["p"]
        if prho < 0.05:
            lineas.append(
                f"- ρ (espacial directa) = {rho:.3f} (p={prho:.3f}): los vecinos influyen de forma significativa "
                "en la misma variable del municipio focal."
            )
        else:
            lineas.append(
                f"- ρ (espacial directa) = {rho:.3f} (p={prho:.3f}): efecto vecinal de la misma variable no significativo."
            )

        for s, d in mec["beta"].items():
            if d["p"] < 0.05:
                lineas.append(
                    f"- β ({s} → {var}, cruce local) = {d['coef']:.3f} (p={d['p']:.3f}): "
                    f"el rezago de {s} en el mismo municipio afecta a {var}."
                )
        for s, d in mec["theta"].items():
            if d["p"] < 0.05:
                lineas.append(
                    f"- θ (W·{s} → {var}, espacial cruzada) = {d['coef']:.3f} (p={d['p']:.3f}): "
                    f"los vecinos en {s} afectan a {var} del municipio focal."
                )

    lineas.append("\n### 3. Impactos espaciales (directo, indirecto, total)")
    for var, imp in impactos.items():
        lineas.append(
            f"- **{var}**: ρ={imp['rho']:.4f}; Directo={imp['directo']:.4f}; "
            f"Indirecto (derrame)={imp['indirecto']:.4f}; Total={imp['total']:.4f}."
        )
        if abs(imp["indirecto"]) > 0.05:
            sentido = "positivo" if imp["indirecto"] > 0 else "negativo"
            lineas.append(
                f"  Una intervención en un municipio genera un derrame {sentido} relevante sobre sus vecinos. "
                "Los programas de alcance subregional capturan mejor este efecto."
            )
        else:
            lineas.append("  El derrame es débil; el diseño municipal autónomo es más defendible.")

    lineas.append("\n### 4. Implicaciones de política (resumen operativo)")
    lineas.append("- Si existen clusters LISA HH en producción o área, priorizar ahí la inversión pública coordinada.")
    lineas.append("- Si φ es alto y significativo, evitar programas de un solo año: la respuesta se acumula en el tiempo.")
    lineas.append("- Si ρ o los impactos indirectos son relevantes, alinear incentivos entre municipios vecinos.")
    lineas.append("- Diferenciar paquetes por piso térmico y rango altitudinal: la estructura productiva no es homogénea.")
    lineas.append("- Sustituir el proxy de luces nocturnas por VIIRS real a escala municipal cuando esté disponible.")
    return "\n".join(lineas)
