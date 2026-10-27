"""Generación de reportes PDF territoriales."""
from __future__ import annotations

import tempfile
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _fig_to_png_bytes(fig, dpi=120):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return buf


def generar_reporte_pdf(
    panel_cultivo: pd.DataFrame,
    lisa_cache: dict | None,
    year_ref: int,
    mejor_matriz_nombre: str,
    municipio: str = "Todos",
    cultivo: str = "Todos",
    anio: str | int = "Todos",
) -> str:
    """
    Genera PDF y retorna la ruta del archivo temporal.
    """
    try:
        anio_val = int(anio) if anio and str(anio) != "Todos" else None
        d = panel_cultivo.copy()
        if municipio and municipio != "Todos":
            d = d[d["municipio"] == municipio]
        if cultivo and cultivo != "Todos":
            d = d[d["cultivo"] == cultivo]
        if anio_val is not None:
            d = d[d["year"] == anio_val]

        if d.empty:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="reporte_vacio_")
            doc = SimpleDocTemplate(tmp.name, pagesize=A4)
            styles = getSampleStyleSheet()
            doc.build([Paragraph("Sin datos para los filtros seleccionados.", styles["Title"])])
            return tmp.name

        area_tot = d["area_sembrada"].sum()
        prod_tot = d["produccion"].sum()
        rend_med = d["rendimiento"].mean()
        n_cult = d["cultivo"].nunique()
        n_years = d["year"].nunique()

        top = (
            d.groupby("cultivo", as_index=False)
            .agg(area=("area_sembrada", "sum"), produccion=("produccion", "sum"), rend=("rendimiento", "mean"))
            .sort_values("produccion", ascending=False)
            .head(15)
        )

        serie = (
            d.groupby("year", as_index=False)
            .agg(area=("area_sembrada", "sum"), produccion=("produccion", "sum"))
            .sort_values("year")
        )

        fig_evo, ax = plt.subplots(figsize=(7.5, 3.8))
        ax.bar(serie["year"], serie["area"], color="#0369a1", alpha=0.75)
        ax2 = ax.twinx()
        ax2.plot(serie["year"], serie["produccion"], color="#b45309", marker="o", lw=2)
        ax.set_xlabel("Año")
        ax.set_ylabel("Área sembrada (ha)", color="#0369a1")
        ax2.set_ylabel("Producción (t)", color="#b45309")
        ax.set_title("Evolución área y producción")
        fig_evo.tight_layout()
        img_evo = _fig_to_png_bytes(fig_evo)

        lisa_txt = "No aplica (seleccione un municipio concreto para contexto LISA)."
        if municipio and municipio != "Todos" and lisa_cache:
            gdf_l = lisa_cache.get("produccion", {}).get("gdf")
            if gdf_l is not None:
                row = gdf_l[gdf_l["municipio"] == municipio]
                if len(row):
                    r = row.iloc[0]
                    lisa_txt = (
                        f"Cluster LISA producción (año ref {year_ref}): "
                        f"{r['lisa_cluster']} | I local={r['lisa_I']:.3f} | p={r['lisa_p']:.3f}"
                    )

        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
            prefix=f"reporte_{str(municipio)[:20]}_{str(cultivo)[:15]}_".replace(" ", "_"),
        )
        doc = SimpleDocTemplate(
            tmp.name,
            pagesize=A4,
            leftMargin=1.6 * cm,
            rightMargin=1.6 * cm,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
        )
        styles = getSampleStyleSheet()
        styles.add(
            ParagraphStyle(
                name="TituloC", parent=styles["Title"], fontSize=16, alignment=TA_CENTER, spaceAfter=8
            )
        )
        styles.add(
            ParagraphStyle(
                name="SubC",
                parent=styles["Normal"],
                fontSize=10,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#334155"),
                spaceAfter=12,
            )
        )
        styles.add(
            ParagraphStyle(
                name="H2b",
                parent=styles["Heading2"],
                fontSize=12,
                textColor=colors.HexColor("#0f172a"),
                spaceBefore=12,
                spaceAfter=6,
            )
        )
        styles.add(
            ParagraphStyle(name="BodyJ", parent=styles["Normal"], fontSize=9, alignment=TA_JUSTIFY, leading=12)
        )
        styles.add(
            ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#64748b"))
        )

        story = []
        story.append(Paragraph("Reporte Territorial SpVAR + ESDA", styles["TituloC"]))
        story.append(
            Paragraph(
                "Nariño · Área sembrada · Producción · LISA · Mecanismos de transmisión",
                styles["SubC"],
            )
        )
        story.append(
            Paragraph(
                f"<b>Municipio:</b> {municipio or 'Todos'} &nbsp;|&nbsp; "
                f"<b>Cultivo:</b> {cultivo or 'Todos'} &nbsp;|&nbsp; "
                f"<b>Año:</b> {anio if str(anio) != 'Todos' else '2019-2024'} &nbsp;|&nbsp; "
                f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                styles["SubC"],
            )
        )

        story.append(Paragraph("1. Indicadores clave", styles["H2b"]))
        kpi_data = [
            ["Área sembrada", "Producción", "Rendimiento medio", "Cultivos", "Años"],
            [f"{area_tot:,.0f} ha", f"{prod_tot:,.0f} t", f"{rend_med:.2f} t/ha", str(n_cult), str(n_years)],
        ]
        t = Table(kpi_data, colWidths=[3.2 * cm] * 5)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f1f5f9")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 8))

        story.append(Paragraph("2. Contexto espacial (LISA)", styles["H2b"]))
        story.append(Paragraph(lisa_txt, styles["BodyJ"]))
        story.append(
            Paragraph(
                "HH = concentración de valores altos. LL = concentración de valores bajos. HL/LH = outliers. NS = no significativo.",
                styles["Small"],
            )
        )

        story.append(Paragraph("3. Ranking de cultivos (por producción)", styles["H2b"]))
        header = ["#", "Cultivo", "Área (ha)", "Producción (t)", "Rend. (t/ha)"]
        rows = [header]
        for i, (_, r) in enumerate(top.iterrows(), 1):
            rows.append(
                [
                    str(i),
                    str(r["cultivo"])[:28],
                    f"{r['area']:,.1f}",
                    f"{r['produccion']:,.1f}",
                    f"{r['rend']:.2f}",
                ]
            )
        t2 = Table(rows, colWidths=[1 * cm, 6 * cm, 3 * cm, 3.5 * cm, 2.5 * cm])
        t2.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
                    ("ALIGN", (0, 0), (0, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(t2)
        story.append(Spacer(1, 10))

        story.append(Paragraph("4. Evolución temporal", styles["H2b"]))
        story.append(RLImage(img_evo, width=16 * cm, height=8 * cm))

        story.append(Paragraph("5. Nota metodológica", styles["H2b"]))
        story.append(
            Paragraph(
                f"Matriz de pesos: <b>{mejor_matriz_nombre}</b> (selección por Moran global y p&lt;0.05). "
                f"SpVAR con cuatro mecanismos (φ, β, ρ, θ), estimado por OLS HC1. "
                f"Impactos espaciales vía S=(I−ρW)^(−1). LISA con 999 permutaciones. "
                f"Proxy VIIRS construido a partir de producción cuando no hay dato satelital municipal.",
                styles["BodyJ"],
            )
        )
        story.append(Spacer(1, 12))
        story.append(
            Paragraph(
                "Documento generado por el Dashboard SpVAR Nariño. Uso para apoyo de decisión territorial.",
                styles["Small"],
            )
        )

        doc.build(story)
        return tmp.name
    except Exception as e:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="reporte_error_")
        doc = SimpleDocTemplate(tmp.name, pagesize=A4)
        styles = getSampleStyleSheet()
        doc.build([Paragraph(f"Error: {str(e)[:400]}", styles["Normal"])])
        return tmp.name
