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
st.title("Forecast de ventas noviembre-diciembre 2026")
st.caption("Gráfica mensual con segmentadores de categoría, combos y producto.")

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
def cargar_archivos(contenidos, nombres):
    partes = []
    requeridas = {"CODIGO", "VENTA", "UNIDADES", "MES", "AÑO"}
    utiles = requeridas | {
        "PRECIO PROMEDIO", "DESCRIPCION", "CATEGORIA",
        "FAMILIA", "TIPO", "COMBOS",
    }

    for contenido, nombre in zip(contenidos, nombres):
        hojas = pd.read_excel(
            io.BytesIO(contenido), sheet_name=None, engine="openpyxl"
        )
        for hoja, df_hoja in hojas.items():
            if df_hoja is None or df_hoja.empty:
                continue
            h = df_hoja.copy()
            h.columns = [str(c).strip().upper() for c in h.columns]
            if not requeridas.issubset(h.columns):
                continue
            partes.append(h[[c for c in h.columns if c in utiles]].copy())

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
    return completar_calendario(agrupado)


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
    hist = df_historico.copy()
    futuro = forecast.copy()

    if categoria != "TODAS":
        hist = hist[hist["CATEGORIA"] == categoria]
        futuro = futuro[futuro["CATEGORIA"] == categoria]

    if combo != "TODOS":
        valor_combo = 1 if combo == "COMBOS" else 0
        hist = hist[hist["COMBOS"] == valor_combo]
        futuro = futuro[futuro["COMBOS"] == valor_combo]

    if producto != "TODOS":
        hist = hist[hist["CODIGO"].astype(str) == producto]
        futuro = futuro[futuro["CODIGO"].astype(str) == producto]

    historico = hist.groupby(["AÑO", "MES_NUM"], as_index=False)["VENTA"].sum()
    historico["Fecha"] = pd.to_datetime(
        dict(
            year=historico["AÑO"].astype(int),
            month=historico["MES_NUM"].astype(int),
            day=1,
        )
    )
    historico = historico.rename(columns={"VENTA": "Valor"})
    historico["Serie"] = "Venta real"

    previsto = futuro.groupby("Periodo", as_index=False)["Prediccion_final"].sum()
    previsto["Fecha"] = pd.to_datetime(
        dict(
            year=(previsto["Periodo"] // 100).astype(int),
            month=(previsto["Periodo"] % 100).astype(int),
            day=1,
        )
    )
    previsto = previsto.rename(columns={"Prediccion_final": "Valor"})
    previsto["Serie"] = "Forecast Nov-Dic 2026"

    return pd.concat(
        [
            historico[["Fecha", "Valor", "Serie"]],
            previsto[["Fecha", "Valor", "Serie"]],
        ],
        ignore_index=True,
    ).sort_values("Fecha")


archivos = st.file_uploader(
    "Carga los archivos de ventas 2024, 2025 y 2026",
    type=["xlsx"],
    accept_multiple_files=True,
)

if archivos:
    try:
        df = cargar_archivos(
            [archivo.getvalue() for archivo in archivos],
            [archivo.name for archivo in archivos],
        )

        if "forecast_2026" not in st.session_state:
            st.info("Genera el forecast para habilitar la gráfica de noviembre-diciembre de 2026.")

        if st.button("Generar forecast noviembre-diciembre 2026", type="primary"):
            with st.spinner("Entrenando el modelo y generando el forecast..."):
                st.session_state["forecast_2026"] = generar_forecast(df)
            st.rerun()

        if "forecast_2026" in st.session_state:
            forecast = st.session_state["forecast_2026"]

            categorias = ["TODAS"] + sorted(
                df["CATEGORIA"].dropna().astype(str).str.strip().unique().tolist()
            )

            c1, c2, c3 = st.columns([2, 1, 3])
            with c1:
                categoria = st.selectbox("Categoría", categorias)
            with c2:
                combo = st.selectbox(
                    "Combos",
                    ["TODOS", "SIN COMBOS", "COMBOS"],
                    index=1,
                )

            universo = df.copy()
            if categoria != "TODAS":
                universo = universo[universo["CATEGORIA"] == categoria]
            if combo != "TODOS":
                universo = universo[
                    universo["COMBOS"] == (1 if combo == "COMBOS" else 0)
                ]

            productos = universo[["CODIGO", "DESCRIPCION"]].drop_duplicates()
            productos["ETIQUETA"] = (
                productos["CODIGO"].astype(str)
                + " | "
                + productos["DESCRIPCION"].astype(str)
            )
            mapa_productos = dict(zip(productos["ETIQUETA"], productos["CODIGO"].astype(str)))

            with c3:
                seleccion_producto = st.selectbox(
                    "Producto",
                    ["TODOS"] + sorted(mapa_productos),
                )
            producto = (
                "TODOS"
                if seleccion_producto == "TODOS"
                else mapa_productos[seleccion_producto]
            )

            datos_grafica = preparar_grafica(
                df, forecast, categoria, combo, producto
            )

            if datos_grafica.empty:
                st.warning("No hay información para la selección actual.")
            else:
                grafica = (
                    alt.Chart(datos_grafica)
                    .mark_line(point=True, strokeWidth=3)
                    .encode(
                        x=alt.X(
                            "Fecha:T",
                            title="Mes",
                            axis=alt.Axis(format="%b %Y", labelAngle=-45),
                        ),
                        y=alt.Y(
                            "Valor:Q",
                            title="Venta",
                            axis=alt.Axis(format="$,.2s"),
                        ),
                        color=alt.Color(
                            "Serie:N",
                            title="Serie",
                            scale=alt.Scale(
                                domain=["Venta real", "Forecast Nov-Dic 2026"],
                                range=["#2563EB", "#F59E0B"],
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("Fecha:T", title="Mes", format="%B %Y"),
                            alt.Tooltip("Serie:N", title="Serie"),
                            alt.Tooltip("Valor:Q", title="Venta", format="$,.2f"),
                        ],
                    )
                    .properties(height=520)
                    .interactive()
                )
                st.altair_chart(grafica, use_container_width=True)

    except Exception as error:
        st.exception(error)
else:
    st.info("Carga los tres archivos Excel para comenzar.")
