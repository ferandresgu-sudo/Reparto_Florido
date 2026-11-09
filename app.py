import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

st.set_page_config(page_title="Forecast Nov-Dic 2026", layout="wide")
st.markdown("## Forecast de ventas | Noviembre y diciembre 2026")
st.caption("Histórico mensual comparable y pronóstico conectado, con filtros de categoría, combo y producto.")

MESES = {
    "ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4,
    "MAYO": 5, "JUNIO": 6, "JULIO": 7, "AGOSTO": 8,
    "SEPTIEMBRE": 9, "OCTUBRE": 10, "NOVIEMBRE": 11,
    "DICIEMBRE": 12,
}

CAT_FEATURES = ["CODIGO", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS"]
LAGS = [1, 2, 3, 6, 12, 24]
NUM_FEATURES = (
    [f"lag{x}" for x in LAGS]
    + [
        "mean3", "std3", "mean6", "std6", "mean12", "std12",
        "sin_mes", "cos_mes", "MES_NUM", "ANO", "tendencia_3_12",
    ]
)
FEATURES = NUM_FEATURES + CAT_FEATURES

# Configuración seleccionada en el backtest Nov-Dic 2025.
PESO_VENTA_DIRECTA = 0.85
PESO_ESTRUCTURAL = 0.15
PESO_FORECAST_ESTACIONAL = 0.85
UMBRAL_ESTACIONAL = 1.20


def normalizar_codigo(serie):
    return serie.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


@st.cache_data(show_spinner=False)
def cargar_archivos(contenidos, nombres, version_cache="carga-v3-por-archivo"):
    partes = []
    requeridas = {"CODIGO", "VENTA", "UNIDADES", "MES", "AÑO"}
    utiles = requeridas | {
        "PRECIO PROMEDIO", "DESCRIPCION", "CATEGORIA",
        "FAMILIA", "TIPO", "COMBOS",
    }

    diagnostico = []
    for contenido, nombre in zip(contenidos, nombres):
        nombre_mayus = str(nombre).upper()
        ano_archivo = next((a for a in [2024, 2025, 2026] if str(a) in nombre_mayus), None)
        hojas = pd.read_excel(
            io.BytesIO(contenido), sheet_name=None, engine="openpyxl"
        )
        filas_archivo = 0
        hojas_validas = 0
        anos_detectados = set()
        for hoja, df_hoja in hojas.items():
            if df_hoja is None or df_hoja.empty:
                continue
            h = df_hoja.copy()
            h.columns = [str(c).strip().upper() for c in h.columns]

            # AÑO puede venir vacío, en texto, con formato decimal o incorrecto.
            # El nombre VENTAS 2024/2025/2026 es la referencia más confiable.
            if "AÑO" not in h.columns:
                if ano_archivo is None:
                    diagnostico.append({"Archivo": nombre, "Hoja": hoja, "Estado": "Omitida", "Detalle": "No existe AÑO y no se pudo inferir del nombre"})
                    continue
                h["AÑO"] = ano_archivo
            else:
                ano_col = pd.to_numeric(h["AÑO"], errors="coerce")
                if ano_archivo is not None:
                    # Si el archivo está claramente identificado, corrige columnas copiadas de 2024.
                    ano_col = pd.Series(ano_archivo, index=h.index, dtype="float64")
                h["AÑO"] = ano_col

            faltantes = sorted(requeridas - set(h.columns))
            if faltantes:
                diagnostico.append({"Archivo": nombre, "Hoja": hoja, "Estado": "Omitida", "Detalle": "Faltan: " + ", ".join(faltantes)})
                continue

            temp = h[[c for c in h.columns if c in utiles]].copy()
            partes.append(temp)
            filas_archivo += len(temp)
            hojas_validas += 1
            anos_detectados.update(pd.to_numeric(temp["AÑO"], errors="coerce").dropna().astype(int).unique().tolist())

        diagnostico.append({
            "Archivo": nombre, "Hoja": "TOTAL",
            "Estado": "Cargado" if hojas_validas else "Sin hojas válidas",
            "Detalle": f"{hojas_validas} hojas | {filas_archivo:,} filas | años {sorted(anos_detectados)}"
        })

    if not partes:
        raise ValueError(
            "No se encontraron hojas con CODIGO, VENTA, UNIDADES, MES y AÑO."
        )

    df = pd.concat(partes, ignore_index=True)
    df["CODIGO"] = normalizar_codigo(df["CODIGO"])
    df["MES_NUM"] = (
        df["MES"].astype(str).str.strip().str.upper().map(MESES)
    )
    df["AÑO"] = pd.to_numeric(df["AÑO"], errors="coerce")
    df["VENTA"] = pd.to_numeric(df["VENTA"], errors="coerce").fillna(0).clip(lower=0)
    df["UNIDADES"] = pd.to_numeric(df["UNIDADES"], errors="coerce").fillna(0).clip(lower=0)

    if "COMBOS" not in df.columns:
        df["COMBOS"] = 0
    df["COMBOS"] = (
        pd.to_numeric(df["COMBOS"], errors="coerce")
        .fillna(0).clip(0, 1).astype(int)
    )

    for col in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO"]:
        if col not in df.columns:
            df[col] = "SIN_DATO"
        df[col] = df[col].fillna("SIN_DATO").astype(str).str.strip()

    df = df.dropna(subset=["CODIGO", "MES_NUM", "AÑO"])
    df["AÑO"] = df["AÑO"].astype(int)
    df["MES_NUM"] = df["MES_NUM"].astype(int)
    df["Periodo"] = df["AÑO"] * 100 + df["MES_NUM"]

    agrupado = (
        df.groupby(["CODIGO", "Periodo", "MES_NUM", "AÑO"], as_index=False)
        .agg(
            VENTA=("VENTA", "sum"),
            UNIDADES=("UNIDADES", "sum"),
            DESCRIPCION=("DESCRIPCION", "first"),
            CATEGORIA=("CATEGORIA", "first"),
            FAMILIA=("FAMILIA", "first"),
            TIPO=("TIPO", "first"),
            COMBOS=("COMBOS", "first"),
        )
    )
    agrupado["PRECIO_PROMEDIO"] = np.where(
        agrupado["UNIDADES"] > 0,
        agrupado["VENTA"] / agrupado["UNIDADES"],
        np.nan,
    )
    return completar_calendario(agrupado), pd.DataFrame(diagnostico)


def completar_calendario(df):
    salida = []
    for codigo, grupo in df.groupby("CODIGO", sort=False):
        x = grupo.sort_values("Periodo").copy()
        fechas = pd.to_datetime(
            dict(year=x["AÑO"], month=x["MES_NUM"], day=1)
        )
        x.index = fechas
        x = x.reindex(pd.date_range(fechas.min(), fechas.max(), freq="MS"))
        x["CODIGO"] = codigo
        x["AÑO"] = x.index.year
        x["MES_NUM"] = x.index.month
        x["Periodo"] = x["AÑO"] * 100 + x["MES_NUM"]
        x["VENTA"] = x["VENTA"].fillna(0.0)
        x["UNIDADES"] = x["UNIDADES"].fillna(0.0)
        for col in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS"]:
            x[col] = x[col].ffill().bfill()
        x["PRECIO_PROMEDIO"] = np.where(
            x["UNIDADES"] > 0, x["VENTA"] / x["UNIDADES"], np.nan
        )
        x["PRECIO_PROMEDIO"] = x["PRECIO_PROMEDIO"].ffill()
        salida.append(x.reset_index(drop=True))

    z = pd.concat(salida, ignore_index=True)
    mediana_categoria = z.groupby("CATEGORIA")["PRECIO_PROMEDIO"].transform("median")
    z["PRECIO_PROMEDIO"] = (
        z["PRECIO_PROMEDIO"]
        .fillna(mediana_categoria)
        .fillna(z["PRECIO_PROMEDIO"].median())
        .clip(lower=0)
    )
    z["COMBOS"] = pd.to_numeric(z["COMBOS"], errors="coerce").fillna(0).astype(int)
    return z


def crear_features(df, objetivo):
    z = df.sort_values(["CODIGO", "Periodo"]).copy()
    grupo = z.groupby("CODIGO")[objetivo]

    for lag in LAGS:
        z[f"lag{lag}"] = grupo.shift(lag)

    for ventana in [3, 6, 12]:
        z[f"mean{ventana}"] = grupo.transform(
            lambda s: s.shift(1).rolling(ventana, min_periods=2).mean()
        )
        z[f"std{ventana}"] = grupo.transform(
            lambda s: s.shift(1).rolling(ventana, min_periods=2).std()
        ).fillna(0)

    z["sin_mes"] = np.sin(2 * np.pi * z["MES_NUM"] / 12)
    z["cos_mes"] = np.cos(2 * np.pi * z["MES_NUM"] / 12)
    z["ANO"] = z["AÑO"]
    z["tendencia_3_12"] = np.where(
        z["mean12"].abs() > 1e-9, z["mean3"] / z["mean12"], 1.0
    )

    for col in CAT_FEATURES:
        z[col] = z[col].fillna("SIN_DATO").astype(str)
    return z


def nuevo_modelo():
    if CatBoostRegressor is None:
        raise ImportError("Instala CatBoost con: python -m pip install catboost")
    return CatBoostRegressor(
        iterations=500,
        depth=7,
        learning_rate=0.05,
        loss_function="MAE",
        eval_metric="MAE",
        l2_leaf_reg=10,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )


def entrenar_objetivo(df, objetivo):
    f = crear_features(df, objetivo).dropna(subset=["lag1", objetivo]).copy()
    medianas = f[NUM_FEATURES].median().fillna(0)
    f[NUM_FEATURES] = f[NUM_FEATURES].fillna(medianas)
    pesos = np.where(f["MES_NUM"].isin([11, 12]), 2.5, 1.0)
    pesos *= 1 + 0.15 * (f["AÑO"] - f["AÑO"].min())

    modelo = nuevo_modelo()
    modelo.fit(
        f[FEATURES],
        np.log1p(f[objetivo].clip(lower=0)),
        cat_features=CAT_FEATURES,
        sample_weight=pesos,
    )
    return modelo, medianas


@st.cache_resource(show_spinner=False)
def entrenar_modelos(df):
    return {
        objetivo: entrenar_objetivo(df, objetivo)
        for objetivo in ["VENTA", "UNIDADES", "PRECIO_PROMEDIO"]
    }


def vector_futuro(historial, periodo, objetivo, medianas):
    hist = historial.sort_values("Periodo")
    valores = hist[objetivo].astype(float).tolist()
    fila = {
        f"lag{lag}": valores[-lag] if len(valores) >= lag else np.nan
        for lag in LAGS
    }

    for ventana in [3, 6, 12]:
        arreglo = np.asarray(valores[-ventana:], dtype=float)
        fila[f"mean{ventana}"] = arreglo.mean() if len(arreglo) >= 2 else np.nan
        fila[f"std{ventana}"] = arreglo.std(ddof=1) if len(arreglo) >= 2 else 0.0

    mes = periodo % 100
    ano = periodo // 100
    fila["sin_mes"] = np.sin(2 * np.pi * mes / 12)
    fila["cos_mes"] = np.cos(2 * np.pi * mes / 12)
    fila["MES_NUM"] = mes
    fila["ANO"] = ano
    fila["tendencia_3_12"] = (
        fila["mean3"] / fila["mean12"]
        if pd.notna(fila["mean12"]) and fila["mean12"] > 0
        else 1.0
    )

    ultimo = hist.iloc[-1]
    for col in CAT_FEATURES:
        fila[col] = str(ultimo.get(col, "SIN_DATO"))

    x = pd.DataFrame([fila], columns=FEATURES)
    x[NUM_FEATURES] = x[NUM_FEATURES].fillna(medianas)
    return x


def predecir_objetivo(modelo, medianas, historial, periodos, objetivo):
    hist = historial.copy()
    predicciones = []

    for periodo in periodos:
        pred = float(
            np.expm1(
                modelo.predict(vector_futuro(hist, periodo, objetivo, medianas))[0]
            )
        )
        pred = max(0.0, pred)
        predicciones.append(pred)

        nueva = hist.iloc[-1].copy()
        nueva["Periodo"] = periodo
        nueva["AÑO"] = periodo // 100
        nueva["MES_NUM"] = periodo % 100
        nueva[objetivo] = pred
        hist = pd.concat([hist, pd.DataFrame([nueva])], ignore_index=True)

    return predicciones


def indice_estacional(historial, ano_objetivo):
    h = historial[historial["AÑO"] < ano_objetivo]
    temporada = h[h["MES_NUM"].isin([11, 12])]["VENTA"]
    resto = h[~h["MES_NUM"].isin([11, 12])]["VENTA"]
    if len(temporada) < 2 or len(resto) < 6 or resto.mean() <= 0:
        return 1.0
    return float(temporada.mean() / resto.mean())


def crecimiento_ytd(historial, ano_objetivo):
    actual = historial[
        (historial["AÑO"] == ano_objetivo) & (historial["MES_NUM"] <= 10)
    ]
    previo = historial[
        (historial["AÑO"] == ano_objetivo - 1) & (historial["MES_NUM"] <= 10)
    ]
    meses = sorted(set(actual["MES_NUM"]) & set(previo["MES_NUM"]))
    if len(meses) < 6:
        return 1.0
    va = actual[actual["MES_NUM"].isin(meses)]["VENTA"].sum()
    vp = previo[previo["MES_NUM"].isin(meses)]["VENTA"].sum()
    return float(np.clip(va / vp, 0.80, 1.40)) if vp > 0 else 1.0


def aplicar_ajuste_estacional(pred_base, historial, periodo):
    mes = periodo % 100
    ano = periodo // 100
    indice = indice_estacional(historial, ano)
    mismos = historial[
        (historial["MES_NUM"] == mes) & (historial["AÑO"] < ano)
    ].sort_values("AÑO").tail(2)

    if mismos.empty or indice < UMBRAL_ESTACIONAL:
        return float(pred_base)

    valores = mismos["VENTA"].astype(float).to_numpy()
    referencia = (
        valores[-1]
        if len(valores) == 1
        else 0.65 * valores[-1] + 0.35 * valores[-2]
    )
    crec = crecimiento_ytd(historial, ano)
    referencia_ytd = referencia * crec
    ajustada = (
        PESO_FORECAST_ESTACIONAL * pred_base
        + (1 - PESO_FORECAST_ESTACIONAL) * referencia_ytd
    )
    piso = valores[-1] * (1 + 0.35 * max(crec - 1, 0))
    return max(float(ajustada), float(piso))


@st.cache_data(show_spinner=False)
def generar_forecast(df):
    modelos = entrenar_modelos(df)
    filas = []
    periodos = [202611, 202612]

    for codigo, historial in df.groupby("CODIGO", sort=False):
        hist = historial.sort_values("Periodo")
        if hist.empty:
            continue
        # Solo pronosticar SKU activos con información reciente de 2026.
        # Evita saltar directamente de 2024/2025 a noviembre de 2026.
        ultimo_periodo = int(hist["Periodo"].max())
        if ultimo_periodo < 202608:
            continue
        meta = hist.iloc[-1]

        venta_directa = predecir_objetivo(
            *modelos["VENTA"], hist, periodos, "VENTA"
        )
        unidades = predecir_objetivo(
            *modelos["UNIDADES"], hist, periodos, "UNIDADES"
        )
        precio = predecir_objetivo(
            *modelos["PRECIO_PROMEDIO"], hist, periodos, "PRECIO_PROMEDIO"
        )

        for periodo, vd, unid, precio_prom in zip(
            periodos, venta_directa, unidades, precio
        ):
            estructural = unid * precio_prom
            base = (
                PESO_VENTA_DIRECTA * vd
                + PESO_ESTRUCTURAL * estructural
            )
            final = aplicar_ajuste_estacional(base, hist, periodo)
            filas.append({
                "CODIGO": codigo,
                "DESCRIPCION": str(meta.get("DESCRIPCION", "")),
                "CATEGORIA": str(meta.get("CATEGORIA", "SIN_DATO")).strip(),
                "COMBOS": int(meta.get("COMBOS", 0)),
                "Periodo": periodo,
                "Prediccion_final": final,
            })

    return pd.DataFrame(filas)


def preparar_grafica(df_historico, forecast, categoria, combo, producto):
    """Prepara histórico comparable y una línea forecast conectada a octubre 2026."""
    futuro = forecast.copy()

    if categoria != "TODAS":
        futuro = futuro[futuro["CATEGORIA"] == categoria]
    if combo != "TODOS":
        valor_combo = 1 if combo == "COMBOS" else 0
        futuro = futuro[futuro["COMBOS"] == valor_combo]
    if producto != "TODOS":
        futuro = futuro[futuro["CODIGO"].astype(str) == producto]

    # El histórico se restringe al mismo universo de SKU que sí tienen forecast.
    codigos_activos = futuro["CODIGO"].astype(str).unique().tolist()
    hist = df_historico[df_historico["CODIGO"].astype(str).isin(codigos_activos)].copy()
    if hist.empty or futuro.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    historico = hist.groupby(["AÑO", "MES_NUM"], as_index=False)["VENTA"].sum()
    historico["Fecha"] = pd.to_datetime(dict(
        year=historico["AÑO"].astype(int),
        month=historico["MES_NUM"].astype(int), day=1
    ))
    historico = historico[historico["Fecha"] <= pd.Timestamp("2026-10-01")]
    historico = historico.rename(columns={"VENTA": "Valor"})
    historico["Serie"] = "Venta real"

    previsto = futuro.groupby("Periodo", as_index=False)["Prediccion_final"].sum()
    previsto["Fecha"] = pd.to_datetime(dict(
        year=(previsto["Periodo"] // 100).astype(int),
        month=(previsto["Periodo"] % 100).astype(int), day=1
    ))
    previsto = previsto.rename(columns={"Prediccion_final": "Valor"})
    previsto["Serie"] = "Forecast"

    # Punto puente: octubre real se agrega al forecast para conectar las líneas.
    octubre = historico[historico["Fecha"] == pd.Timestamp("2026-10-01")][["Fecha", "Valor"]].copy()
    if octubre.empty:
        ultima = historico.sort_values("Fecha").tail(1)[["Fecha", "Valor"]].copy()
        puente = ultima
    else:
        puente = octubre
    puente["Serie"] = "Forecast"
    forecast_conectado = pd.concat(
        [puente[["Fecha", "Valor", "Serie"]], previsto[["Fecha", "Valor", "Serie"]]],
        ignore_index=True,
    ).sort_values("Fecha")

    return (
        historico[["Fecha", "Valor", "Serie"]].sort_values("Fecha"),
        forecast_conectado,
        previsto[["Fecha", "Valor", "Serie"]].sort_values("Fecha"),
    )


archivos = st.file_uploader(
    "Carga los archivos de ventas 2024, 2025 y 2026",
    type=["xlsx"],
    accept_multiple_files=True,
)

if archivos:
    try:
        df, diagnostico_carga = cargar_archivos(
            [archivo.getvalue() for archivo in archivos],
            [archivo.name for archivo in archivos],
            version_cache="carga-v3-por-archivo",
        )
        anos_cargados = sorted(df["AÑO"].dropna().astype(int).unique().tolist())
        conteo_ano = df.groupby("AÑO")["CODIGO"].nunique().to_dict()
        st.success(
            "Archivos procesados | "
            + " | ".join(f"{int(a)}: {int(conteo_ano.get(a, 0))} SKU" for a in anos_cargados)
        )
        with st.expander("Diagnóstico de carga de archivos"):
            st.dataframe(diagnostico_carga, use_container_width=True, hide_index=True)
        faltantes = sorted({2024, 2025, 2026} - set(anos_cargados))
        if faltantes:
            st.error(
                f"Después de procesar los Excel siguen faltando los años {faltantes}. "
                "Abre Diagnóstico de carga para identificar la hoja omitida."
            )
            st.stop()

        if "forecast_2026" not in st.session_state:
            st.info("Genera el forecast para habilitar la gráfica de noviembre-diciembre de 2026.")

        if st.button("Generar forecast noviembre-diciembre 2026", type="primary"):
            with st.spinner("Entrenando el modelo y generando el forecast..."):
                st.session_state["forecast_2026"] = generar_forecast(df)
            st.rerun()

        if "forecast_2026" in st.session_state:
            forecast = st.session_state["forecast_2026"]
            if forecast.empty:
                st.error("No se encontraron SKU activos con datos recientes de 2026.")
                st.stop()

            st.markdown("### Filtros")
            categorias = ["TODAS"] + sorted(
                forecast["CATEGORIA"].dropna().astype(str).str.strip().unique().tolist()
            )
            c1, c2, c3 = st.columns([2, 1.25, 3])
            with c1:
                categoria = st.selectbox("Categoría", categorias)
            with c2:
                combo = st.segmented_control(
                    "Tipo de producto",
                    options=["TODOS", "SIN COMBOS", "COMBOS"],
                    default="SIN COMBOS",
                )
                if combo is None:
                    combo = "SIN COMBOS"

            universo = forecast.copy()
            if categoria != "TODAS":
                universo = universo[universo["CATEGORIA"] == categoria]
            if combo != "TODOS":
                universo = universo[
                    universo["COMBOS"] == (1 if combo == "COMBOS" else 0)
                ]

            productos = universo[["CODIGO", "DESCRIPCION"]].drop_duplicates()
            productos["ETIQUETA"] = (
                productos["CODIGO"].astype(str) + " | "
                + productos["DESCRIPCION"].astype(str)
            )
            mapa_productos = dict(
                zip(productos["ETIQUETA"], productos["CODIGO"].astype(str))
            )
            with c3:
                seleccion_producto = st.selectbox(
                    "Producto", ["TODOS"] + sorted(mapa_productos)
                )
            producto = "TODOS" if seleccion_producto == "TODOS" else mapa_productos[seleccion_producto]

            historico, forecast_linea, forecast_puntos = preparar_grafica(
                df, forecast, categoria, combo, producto
            )
            if historico.empty or forecast_puntos.empty:
                st.warning("No hay datos comparables para la selección actual.")
                st.stop()

            nov = forecast_puntos.loc[
                forecast_puntos["Fecha"] == pd.Timestamp("2026-11-01"), "Valor"
            ].sum()
            dic = forecast_puntos.loc[
                forecast_puntos["Fecha"] == pd.Timestamp("2026-12-01"), "Valor"
            ].sum()
            oct_real = historico.loc[
                historico["Fecha"] == pd.Timestamp("2026-10-01"), "Valor"
            ].sum()
            crecimiento_nov = (nov / oct_real - 1) * 100 if oct_real > 0 else np.nan
            crecimiento_dic = (dic / nov - 1) * 100 if nov > 0 else np.nan

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Octubre 2026 real", f"${oct_real:,.0f}")
            k2.metric("Noviembre 2026", f"${nov:,.0f}",
                      f"{crecimiento_nov:+.1f}% vs octubre" if pd.notna(crecimiento_nov) else None)
            k3.metric("Diciembre 2026", f"${dic:,.0f}",
                      f"{crecimiento_dic:+.1f}% vs noviembre" if pd.notna(crecimiento_dic) else None)
            k4.metric("SKU incluidos", f"{universo['CODIGO'].nunique():,}")

            fecha_min = pd.Timestamp("2024-01-01")
            fecha_max = pd.Timestamp("2026-12-01")
            eje_x = alt.X(
                "Fecha:T", title=None,
                scale=alt.Scale(domain=[fecha_min, fecha_max]),
                axis=alt.Axis(
                    format="%b %Y", labelAngle=-35,
                    tickCount={"interval": "month", "step": 2},
                    grid=False, labelPadding=10,
                ),
            )
            eje_y = alt.Y(
                "Valor:Q", title="Venta mensual",
                scale=alt.Scale(zero=False, nice=True),
                axis=alt.Axis(format="$,.2s", grid=True, gridOpacity=0.14),
            )
            tooltip = [
                alt.Tooltip("Fecha:T", title="Mes", format="%B %Y"),
                alt.Tooltip("Valor:Q", title="Venta", format="$,.2f"),
            ]

            linea_real = alt.Chart(historico).mark_line(
                color="#3B82F6", strokeWidth=3.2, interpolate="monotone"
            ).encode(x=eje_x, y=eje_y, tooltip=tooltip)
            puntos_reales = alt.Chart(historico).mark_circle(
                color="#3B82F6", size=44, opacity=0.9
            ).encode(x=eje_x, y=eje_y, tooltip=tooltip)

            linea_forecast = alt.Chart(forecast_linea).mark_line(
                color="#F59E0B", strokeWidth=3.5,
                strokeDash=[8, 5], interpolate="monotone"
            ).encode(x=eje_x, y=eje_y, tooltip=tooltip)
            puntos_forecast = alt.Chart(forecast_puntos).mark_circle(
                color="#F59E0B", size=125, stroke="white", strokeWidth=1.5
            ).encode(x=eje_x, y=eje_y, tooltip=tooltip)

            limite = pd.DataFrame({"Fecha": [pd.Timestamp("2026-11-01")]})
            regla = alt.Chart(limite).mark_rule(
                color="#94A3B8", strokeDash=[4, 4], opacity=0.75
            ).encode(x="Fecha:T")
            etiqueta = alt.Chart(pd.DataFrame({
                "Fecha": [pd.Timestamp("2026-11-01")],
                "Texto": ["Inicio forecast"]
            })).mark_text(
                align="left", dx=7, dy=-205,
                color="#94A3B8", fontSize=12
            ).encode(x="Fecha:T", text="Texto:N")

            chart = alt.layer(
                linea_real, puntos_reales,
                linea_forecast, puntos_forecast,
                regla, etiqueta,
            ).properties(
                height=520,
                title=alt.TitleParams(
                    text="Evolución mensual y forecast Nov-Dic 2026",
                    subtitle="Línea azul: ventas reales | Línea naranja punteada: pronóstico",
                    anchor="start", fontSize=18, subtitleFontSize=12,
                    offset=16,
                ),
            ).configure_view(strokeWidth=0).configure_axis(
                labelFontSize=12, titleFontSize=13
            )

            st.altair_chart(chart, use_container_width=True)
            st.caption(
                "El histórico utiliza exactamente los mismos SKU activos incluidos en el forecast. "
                "Los productos sin actividad reciente en 2026 se excluyen para evitar saltos artificiales."
            )

    except Exception as error:
        st.exception(error)
else:
    st.info("Carga los tres archivos Excel para comenzar.")
