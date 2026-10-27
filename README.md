# SpVAR + ESDA · Dashboard Nariño

Dashboard interactivo de **análisis espacial exploratorio (ESDA)** y **modelo SpVAR** para el panel agrícola de Nariño (2019–2024).

**Variables endógenas:** área sembrada · producción · luces nocturnas (proxy VIIRS)

**Filtros:** municipio · cultivo · grupo de cultivo · rango altitudinal · piso térmico · año

**Mecanismos de transmisión (especificación formal):**

| Símbolo | Mecanismo | Interpretación |
|---------|-----------|----------------|
| \(\phi\) | Autodinámica espacial | Persistencia del municipio en la misma variable |
| \(\beta\) | Cruce local | Efecto de otras variables del mismo municipio (rezago 1) |
| \(\rho\) | Relación espacial directa | Efecto de vecinos en la misma variable |
| \(\theta\) | Relación espacial cruzada | Efecto de vecinos en otras variables |

Incluye selección automática de matriz de pesos \(W\) (Queen / Rook / KNN), LISA, impactos espaciales (LeSage & Pace), auditoría de calidad y exportación PDF.

---

## Estructura del repositorio

```
spvar-narino/
├── app.py                 # Aplicación Streamlit (punto de entrada)
├── requirements.txt
├── packages.txt           # Dependencias de sistema (Streamlit Cloud / Docker)
├── README.md
├── .gitignore
├── data/
│   └── panel_agricola_narino_2019_2024.csv
└── src/
    ├── __init__.py
    ├── data_loader.py     # Carga panel + geometría municipios
    ├── spatial_analysis.py# ESDA, W óptima, LISA
    ├── spvar.py           # Estimación SpVAR + impactos + auditoría
    └── pdf_report.py      # Reportes PDF territoriales
```

---

## Requisitos

- Python 3.10+
- Ver `requirements.txt` (pandas, geopandas, libpysal, esda, streamlit, plotly, statsmodels, reportlab, …)

En Linux / Streamlit Cloud se necesitan librerías de sistema para GeoPandas (listadas en `packages.txt`).

---

## Instalación local

```bash
git clone https://github.com/<tu-usuario>/spvar-narino.git
cd spvar-narino

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Si usas Ubuntu/Debian y fallan los imports de geopandas:

```bash
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev libspatialindex-dev
```

---

## Ejecutar el dashboard

```bash
streamlit run app.py
```

La app abre en `http://localhost:8501`.

Al iniciar:

1. Carga el panel CSV y la geometría de municipios de Nariño (GeoJSON público).
2. Selecciona la matriz \(W\) óptima (Moran global).
3. Estima el SpVAR (OLS HC1) y calcula impactos espaciales.
4. Ejecuta la auditoría de calidad (nulos, estabilidad temporal, \(W\) row-estándar, \(|\rho|<1\)).

---

## Despliegue en Streamlit Community Cloud

1. Sube este repositorio a GitHub (público o privado).
2. En [share.streamlit.io](https://share.streamlit.io):
   - New app → selecciona el repo.
   - Main file path: `app.py`
   - Python version: 3.11 (recomendado).
3. Streamlit Cloud instalará `requirements.txt` y las dependencias de `packages.txt`.

**Nota:** la primera ejecución descarga el GeoJSON de municipios; se cachea con `@st.cache_resource`.

---

## Datos

- **Fuente:** panel agrícola Nariño 2019–2024 (municipio × cultivo × año).
- **Proxy VIIRS (`luces_nocturnas`):** generado de forma determinista a partir de producción cuando no hay dato satelital municipal. Sustituir por VIIRS real cuando esté disponible.
- Geometría: [santiblanko/colombia.geojson](https://github.com/santiblanko/colombia.geojson) (municipios, filtro código DANE 52).

---

## Auditoría incluida

Al cargar el modelo se verifica:

1. Endógenas sin nulos ni infinitos.
2. Estabilidad temporal: \(\max|\lambda| < 1\) en la diagonal de \(\phi\).
3. \(W\) row-estandarizada.
4. Dimensiones \(n \times T\).
5. \(|\rho| < 1\) (invertibilidad de \(S = (I-\rho W)^{-1}\)).

Los resultados aparecen en la pestaña **Resumen técnico**.

---

## Licencia y uso

Uso orientado a apoyo de decisión territorial. Citar la fuente de datos agrícola y el repositorio al reutilizar.

---

## Referencias metodológicas

- Anselin, L. — Local Indicators of Spatial Association (LISA).
- LeSage, J. & Pace, R. K. — *Introduction to Spatial Econometrics* (impactos directo/indirecto/total).
- Modelos VAR espaciales / SpVAR con mecanismos aislados \(\phi, \beta, \rho, \theta\).
