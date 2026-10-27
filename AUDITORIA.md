# Auditoría del proyecto SpVAR Nariño

Fecha de auditoría: 2026-10-27  
Alcance: panel CSV, notebook original (Gradio), adaptación Streamlit + estructura GitHub.

---

## 1. Datos (panel agrícola)

| Check | Resultado |
|-------|-----------|
| Archivo | `data/panel_agricola_narino_2019_2024.csv` |
| Filas × columnas | 9 742 × 23 |
| Municipios | 64 (todos código DANE 52xxx) |
| Años | 2019–2024 (completo) |
| Cultivos | 103 |
| Nulos en área/producción | 0 |
| Valores negativos | 0 |
| Panel municipio-año | 384 obs (64 × 6) — **balanceado** |
| Códigos fuera de Nariño | 0 |

**Notas:**

- Existen 449 filas con producción = 0 (posible no cosecha / ciclo incompleto); se tratan como 0, no como missing.
- El proxy `luces_nocturnas` se genera de forma **determinista** (seed=42) a partir de producción. Documentado en UI y README; debe sustituirse por VIIRS municipal real cuando exista.

---

## 2. Código original (notebook Gradio)

### Fortalezas
- Especificación SpVAR formal con cuatro mecanismos (phi, beta, rho, theta).
- Selección automática de W (Queen / Rook / KNN 3–8) por significancia Moran.
- LISA con 999 permutaciones y etiquetas HH/LL/HL/LH/NS.
- Impactos espaciales vía S=(I−ρW)^(−1).
- Auditoría de estabilidad temporal, row-std de W y |ρ|<1.
- Exportación PDF con reportlab.

### Riesgos / deuda detectada
| Ítem | Severidad | Acción en la adaptación |
|------|-----------|-------------------------|
| Dependencia de Google Colab / Drive | Media | Rutas relativas + fallbacks; CSV versionado en data/ |
| Gradio no es el estándar para GitHub/Cloud | Baja | Migrado a **Streamlit** |
| Queen.from_dataframe sin use_index (deprecación libpysal) | Baja | Añadido use_index=False |
| Proxy VIIRS poco visible en UI | Baja | Visible en Resumen técnico e interpretación |
| Instalación !pip en notebook | Baja | requirements.txt + packages.txt |
| Notebook monolítico | Media | Modularizado en src/ |

---

## 3. Adaptación Streamlit + GitHub

### Estructura
```
spvar-narino/
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
├── AUDITORIA.md
├── .gitignore
├── .streamlit/config.toml
├── data/panel_agricola_narino_2019_2024.csv
└── src/
    ├── data_loader.py
    ├── spatial_analysis.py
    ├── spvar.py
    └── pdf_report.py
```

### Checks SpVAR (runtime)
Implementados en `src/spvar.auditoria_spvar` y mostrados en la pestaña **Resumen técnico**:

1. Endógenas finitas (sin NaN/Inf).
2. Estabilidad temporal: max |λ(diag φ)| < 1.
3. W row-estandarizada.
4. Dimensiones n × T.
5. |ρ| < 1 (invertibilidad de S).

### Despliegue
- Local: `streamlit run app.py`
- Streamlit Cloud: apuntar a `app.py`; usa requirements.txt + packages.txt.
- Primera carga descarga GeoJSON de municipios (cache @st.cache_resource).

---

## 4. Limitaciones conocidas (no bloqueantes)

1. Proxy VIIRS: no es dato satelital real.
2. OLS HC1 vs estimadores espaciales simultáneos: la especificación aísla mecanismos vía rezagos; coherente con el diseño del notebook original.
3. Geometría online: requiere red en el primer arranque; para offline, cachear el GeoJSON en data/.
4. Tamaño del CSV (~2.3 MB): aceptable para GitHub.

---

## 5. Conclusión

| Área | Estado |
|------|--------|
| Calidad del panel | OK (balanceado, sin nulos críticos) |
| Especificación SpVAR / ESDA | Conservada y modularizada |
| Auditoría automática | Implementada en app |
| Listo para GitHub | Sí |
| Listo para Streamlit Cloud | Sí (con packages.txt) |

El repositorio puede subirse tal cual y desplegarse con `streamlit run app.py`.
