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

with st.expander("¿Cómo funciona el modelo?", expanded=False):
    st.markdown(
        """
### Resumen

El forecast combina **tres modelos globales CatBoost** y una corrección estacional controlada. El objetivo es estimar las ventas de noviembre y diciembre de 2026 sin limitarse a copiar el cierre de 2025.

### 1. Información utilizada

El modelo aprende con datos mensuales de 2024, 2025 y 2026 hasta octubre:

- Venta monetaria.
- Unidades vendidas.
- Precio promedio, recalculado como `VENTA / UNIDADES`.
- Código, categoría, familia, tipo de producto y condición de combo.
- Rezagos de 1, 2, 3, 6, 12 y 24 meses.
- Promedios y variabilidad de los últimos 3, 6 y 12 meses.
- Mes del año y tendencia reciente.

### 2. Tres pronósticos CatBoost

Se entrenan tres modelos:

1. **Venta directa:** pronostica el importe monetario.
2. **Unidades:** pronostica cuántas unidades se venderán.
3. **Precio promedio:** pronostica el precio medio esperado.

La venta estructural se calcula así:

`Venta estructural = Unidades pronosticadas × Precio pronosticado`

La predicción base combina:

`Predicción base = 85% venta directa + 15% venta estructural`

### 3. Estacionalidad de noviembre y diciembre

El ajuste externo solo se activa cuando el SKU presenta un índice estacional mínimo de **1.20**. Esto significa que históricamente noviembre y diciembre deben vender al menos 20% más que el resto del año.

Cuando se activa, se construye una referencia con:

- 65% del mismo mes más reciente.
- 35% del mismo mes de hace dos años.
- Crecimiento enero-octubre del año objetivo frente al año anterior.

La señal queda combinada como:

`Predicción estacional = 85% predicción base + 15% referencia histórica YTD`

### 4. Piso parcial validado

La venta del mismo mes anterior no se usa como piso rígido. Solo funciona como protección ante una subestimación extrema.

- En **noviembre** se recupera 45% de la distancia hacia el piso parcial.
- En **diciembre** se recupera 60%, porque diciembre mostró mayor subestimación en el backtest.
- Si la predicción ya está por encima del piso, no se agrega corrección.

Esto permite que el forecast quede por encima o por debajo de 2025 según la tendencia, sin copiar exactamente ese valor.

### 5. Validación histórica

Ejecuta el backtest de noviembre y diciembre de 2025 para comparar WAPE, MAE, RMSE y R² con y sin sucursales en el mismo conjunto de evaluación.

### 6. Interpretación de la gráfica

- **Línea azul:** ventas reales hasta octubre de 2026.
- **Línea naranja punteada:** pronóstico de noviembre y diciembre de 2026.
- Los filtros permiten analizar categoría, productos sin combos, combos o un SKU individual.
- El histórico utiliza el mismo universo de SKU activos incluidos en el forecast.

### 7. Alcance

El resultado es adecuado como forecast operativo con revisión comercial. Los combos, lanzamientos, bajas, promociones futuras, faltantes y cambios extraordinarios de distribución requieren validación adicional porque esas decisiones no siempre están presentes en el histórico.
        """
    )

with st.expander("Fórmulas principales", expanded=False):
    st.code(
        """Venta estructural = Unidades predichas * Precio predicho

Predicción base = 0.85 * Venta directa + 0.15 * Venta estructural

Predicción estacional = 0.85 * Predicción base + 0.15 * Referencia histórica YTD

Si Predicción combinada < Piso parcial:
    Noviembre: Predicción final = combinada + 0.45 * (piso - combinada)
    Diciembre: Predicción final = combinada + 0.60 * (piso - combinada)
Si no:
    Predicción final = Predicción combinada""",
        language="text",
    )

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

# Calibración selectiva V2 validada sobre el backtest Nov-Dic 2025 sin combos.
CALIBRACION_CATEGORIAS_V2 = {
    "LACTEOS CULINARIOS": 1.00,  # Sin uplift: contracción comercial conocida.
    "CHOCOLATE MESA": 1.10,
    "NESTLE PROFESSIONAL": 1.10,
    "RTD": 1.08,
    "CHOCOLATE IMPULSO": 0.90,
    "CARNATION POLVO": 0.92,
    "CEREALES": 0.97,
    "AGUAS": 0.97,
}
FACTOR_CALIBRACION_MIN = 0.90
FACTOR_CALIBRACION_MAX = 1.10
APLICAR_CALIBRACION_A_COMBOS = False
# Ajuste comercial solicitado para CHOCOLATE MESA, solo noviembre 2026.
REDUCCION_NOV_CHOCOLATE_MESA = 0.094
FACTOR_NOV_CHOCOLATE_MESA = 1.0 - REDUCCION_NOV_CHOCOLATE_MESA

# Control comercial de diciembre para LACTEOS CULINARIOS.
# Diciembre se ancla al noviembre actualizado del escenario, no al forecast alto original.
APLICAR_TOPE_DICIEMBRE_LACTEOS = True
PISO_DIC_VS_NOV_LACTEOS = 0.95       # Diciembre no cae más de 5% vs noviembre actualizado.
CRECIMIENTO_BASE_DIC_LACTEOS = 0.05  # Escenario central: +5% vs noviembre actualizado.
TOPE_DIC_VS_NOV_LACTEOS = 1.18       # Máximo: +18% vs noviembre actualizado.
USAR_RELACION_HISTORICA_LACTEOS = True

# Control definitivo del forecast base de Lácteos Culinarios.
# Si el modelo supera este nivel, diciembre se ajusta a noviembre × 1.18.
APLICAR_CONTROL_BASE_DIC_LACTEOS = True
CRECIMIENTO_DIC_LACTEOS = 0.18
FACTOR_DIC_VS_NOV_LACTEOS = 1.0 + CRECIMIENTO_DIC_LACTEOS


def normalizar_codigo(serie):
    return serie.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def _normalizar_encabezado(valor):
    """Normaliza encabezados de Excel sin depender de espacios, acentos o saltos."""
    import re
    import unicodedata
    texto = unicodedata.normalize("NFKD", str(valor))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.replace("\n", " ").replace("\r", " ").strip().upper()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def _buscar_columna(columnas, aliases, prefijos=None):
    """Busca encabezados exactos y también variantes como UNIDADES 2025."""
    mapa = {_normalizar_encabezado(c): c for c in columnas}
    for alias in aliases:
        alias_normalizado = _normalizar_encabezado(alias)
        if alias_normalizado in mapa:
            return mapa[alias_normalizado]
    for prefijo in prefijos or []:
        prefijo_normalizado = _normalizar_encabezado(prefijo)
        for encabezado_normalizado, encabezado_original in mapa.items():
            if encabezado_normalizado.startswith(prefijo_normalizado):
                return encabezado_original
    return None


def _inferir_ano(nombre_archivo):
    import re
    coincidencia = re.search(r"(?:19|20)\d{2}", str(nombre_archivo))
    return int(coincidencia.group()) if coincidencia else None


def _inferir_mes(nombre_hoja):
    nombre = _normalizar_encabezado(nombre_hoja)
    for mes, numero in MESES.items():
        if mes in nombre:
            return numero
    return None


def cargar_archivos(archivos):
    """Carga robusta de los tres Excel.

    El año se toma del nombre VENTAS 2024/2025/2026. La función acepta una hoja
    por mes o una tabla consolidada, y reconoce variaciones de encabezados.
    No usa caché para evitar reutilizar una lectura anterior incompleta.
    """
    partes = []
    diagnostico = []

    aliases = {
        "CODIGO": ["CODIGO", "CÓDIGO", "SKU", "COD PRODUCTO", "COD. PRODUCTO"],
        "VENTA": ["VENTA", "VENTAS", "MONTO VENTA", "VENTA NETA"],
        "UNIDADES": ["UNIDADES", "UNIDAD", "UNIDADES VENDIDAS", "CANTIDAD", "PIEZAS"],
        "MES": ["MES", "NOMBRE MES"],
        "DESCRIPCION": ["DESCRIPCION", "DESCRIPCIÓN", "PRODUCTO", "NOMBRE PRODUCTO"],
        "CATEGORIA": ["CATEGORIA", "CATEGORÍA"],
        "FAMILIA": ["FAMILIA"],
        "TIPO": ["TIPO"],
        "COMBOS": ["COMBOS", "COMBO", "ES COMBO"],
        "CANTIDAD_SUCURSALES": ["CANTIDAD_SUCURSALES", "SUCURSALES_ACTIVAS_ESTIMADAS"],
    }

    for archivo in archivos:
        nombre = archivo.name
        ano_archivo = _inferir_ano(nombre)
        if ano_archivo not in {2024, 2025, 2026}:
            diagnostico.append({
                "Archivo": nombre, "Hoja": "-", "Estado": "ERROR",
                "Filas": 0, "Año asignado": ano_archivo,
                "Detalle": "El nombre debe contener 2024, 2025 o 2026",
            })
            continue

        try:
            contenido = archivo.getvalue()
            libro = pd.ExcelFile(io.BytesIO(contenido), engine="openpyxl")
        except Exception as exc:
            diagnostico.append({
                "Archivo": nombre, "Hoja": "-", "Estado": "ERROR",
                "Filas": 0, "Año asignado": ano_archivo,
                "Detalle": f"No fue posible abrir el Excel: {exc}",
            })
            continue

        for hoja in libro.sheet_names:
            try:
                h = pd.read_excel(libro, sheet_name=hoja)
            except Exception as exc:
                diagnostico.append({
                    "Archivo": nombre, "Hoja": hoja, "Estado": "ERROR",
                    "Filas": 0, "Año asignado": ano_archivo,
                    "Detalle": f"No fue posible leer la hoja: {exc}",
                })
                continue

            if h is None or h.empty:
                diagnostico.append({
                    "Archivo": nombre, "Hoja": hoja, "Estado": "VACÍA",
                    "Filas": 0, "Año asignado": ano_archivo,
                    "Detalle": "Hoja sin registros",
                })
                continue

            columnas = {
                destino: _buscar_columna(
                    h.columns,
                    opciones,
                    prefijos=["UNIDADES"] if destino == "UNIDADES" else None,
                )
                for destino, opciones in aliases.items()
            }
            obligatorias = ["CODIGO", "VENTA", "UNIDADES"]
            faltantes = [c for c in obligatorias if columnas[c] is None]
            mes_hoja = _inferir_mes(hoja)
            if columnas["MES"] is None and mes_hoja is None:
                faltantes.append("MES")

            if faltantes:
                diagnostico.append({
                    "Archivo": nombre, "Hoja": hoja, "Estado": "OMITIDA",
                    "Filas": len(h), "Año asignado": ano_archivo,
                    "Detalle": "Faltan columnas: " + ", ".join(faltantes),
                })
                continue

            temp = pd.DataFrame(index=h.index)
            temp["CODIGO"] = h[columnas["CODIGO"]]
            temp["VENTA"] = h[columnas["VENTA"]]
            temp["UNIDADES"] = h[columnas["UNIDADES"]]
            c = columnas["CANTIDAD_SUCURSALES"]
            temp["CANTIDAD_SUCURSALES"] = pd.to_numeric(h[c], errors="coerce") if c is not None else np.nan
            temp["AÑO"] = ano_archivo

            if columnas["MES"] is not None:
                mes_original = h[columnas["MES"]]
                mes_texto = mes_original.astype(str).map(_normalizar_encabezado)
                mes_numero = pd.to_numeric(mes_original, errors="coerce")
                temp["MES_NUM"] = mes_texto.map(MESES).fillna(mes_numero)
            else:
                temp["MES_NUM"] = mes_hoja

            for col in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO"]:
                temp[col] = (
                    h[columnas[col]] if columnas[col] is not None else "SIN_DATO"
                )
            temp["COMBOS"] = (
                h[columnas["COMBOS"]] if columnas["COMBOS"] is not None else 0
            )

            temp["CODIGO"] = normalizar_codigo(temp["CODIGO"])
            temp["VENTA"] = pd.to_numeric(temp["VENTA"], errors="coerce")
            temp["UNIDADES"] = pd.to_numeric(temp["UNIDADES"], errors="coerce")
            temp["MES_NUM"] = pd.to_numeric(temp["MES_NUM"], errors="coerce")
            temp["COMBOS"] = pd.to_numeric(temp["COMBOS"], errors="coerce").fillna(0)

            temp = temp.dropna(subset=["CODIGO", "VENTA", "UNIDADES", "MES_NUM"])
            temp = temp[temp["MES_NUM"].between(1, 12)]
            temp["MES_NUM"] = temp["MES_NUM"].astype(int)
            temp["VENTA"] = temp["VENTA"].clip(lower=0)
            temp["UNIDADES"] = temp["UNIDADES"].clip(lower=0)
            temp["COMBOS"] = temp["COMBOS"].clip(0, 1).astype(int)
            temp["Periodo"] = temp["AÑO"] * 100 + temp["MES_NUM"]

            for col in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO"]:
                temp[col] = temp[col].fillna("SIN_DATO").astype(str).str.strip()

            if temp.empty:
                diagnostico.append({
                    "Archivo": nombre, "Hoja": hoja, "Estado": "OMITIDA",
                    "Filas": 0, "Año asignado": ano_archivo,
                    "Detalle": "No quedaron filas válidas después de limpiar",
                })
                continue

            partes.append(temp)
            diagnostico.append({
                "Archivo": nombre, "Hoja": hoja, "Estado": "CARGADA",
                "Filas": len(temp), "Año asignado": ano_archivo,
                "Detalle": (
                    f"{temp['CODIGO'].nunique()} SKU | meses {sorted(temp['MES_NUM'].unique().tolist())} | "
                    f"unidades leídas desde: {columnas['UNIDADES']}"
                ),
            })

    if not partes:
        raise ValueError(
            "Ninguna hoja pudo cargarse. Abre el diagnóstico y revisa los encabezados."
        )

    df = pd.concat(partes, ignore_index=True)
    sucursales_mes = tabla_sucursales(df)
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
    agrupado["REGISTRO_ORIGINAL"] = True
    salida = completar_calendario(agrupado)
    salida["CANTIDAD_SUCURSALES"] = salida["Periodo"].map(sucursales_mes)
    return salida, pd.DataFrame(diagnostico)


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


def nuevo_modelo(iteraciones=500):
    if CatBoostRegressor is None:
        raise ImportError("Instala CatBoost con: python -m pip install catboost")
    return CatBoostRegressor(
        iterations=iteraciones,
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


def entrenar_objetivo(df, objetivo, iteraciones=500):
    f = crear_features(df, objetivo).dropna(subset=["lag1", objetivo]).copy()
    medianas = f[NUM_FEATURES].median().fillna(0)
    f[NUM_FEATURES] = f[NUM_FEATURES].fillna(medianas)
    pesos = np.where(f["MES_NUM"].isin([11, 12]), 2.5, 1.0)
    pesos *= 1 + 0.15 * (f["AÑO"] - f["AÑO"].min())

    modelo = nuevo_modelo(iteraciones)
    modelo.fit(
        f[FEATURES],
        np.log1p(f[objetivo].clip(lower=0)),
        cat_features=CAT_FEATURES,
        sample_weight=pesos,
    )
    return modelo, medianas


@st.cache_resource(show_spinner=False)
def entrenar_modelos(df, iteraciones=500):
    return {
        objetivo: entrenar_objetivo(df, objetivo, iteraciones)
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
    """Ajuste estacional final validado: piso parcial 45% Nov / 60% Dic.

    No copia automáticamente el mismo mes anterior. Si la predicción queda por
    debajo del piso parcial, recupera solo una parte de la diferencia.
    """
    mes = periodo % 100
    ano = periodo // 100
    indice = indice_estacional(historial, ano)
    mismos = historial[
        (historial["MES_NUM"] == mes) & (historial["AÑO"] < ano)
    ].sort_values("AÑO").tail(2)

    diagnostico = {
        "Indice_estacional": float(indice),
        "Referencia_ultimo_ano": np.nan,
        "Crecimiento_YTD": 1.0,
        "Prediccion_estacional": float(pred_base),
        "Prediccion_combinada": float(pred_base),
        "Piso_parcial": np.nan,
        "Porcentaje_recuperacion": 0.0,
        "Correccion_piso": 0.0,
        "Ajuste_aplicado": False,
    }

    if mismos.empty or indice < UMBRAL_ESTACIONAL:
        return max(0.0, float(pred_base)), diagnostico

    valores = mismos["VENTA"].astype(float).to_numpy()
    ultimo_ano = float(valores[-1])
    referencia_historica = (
        ultimo_ano if len(valores) == 1
        else 0.65 * ultimo_ano + 0.35 * float(valores[-2])
    )
    crec = crecimiento_ytd(historial, ano)
    referencia_ytd = referencia_historica * crec

    prediccion_estacional = (
        PESO_FORECAST_ESTACIONAL * float(pred_base)
        + (1.0 - PESO_FORECAST_ESTACIONAL) * referencia_ytd
    )

    crecimiento_suavizado = float(np.clip(
        1.0 + 0.50 * (crec - 1.0), 0.90, 1.20
    ))
    referencia_suavizada = ultimo_ano * crecimiento_suavizado
    prediccion_combinada = (
        0.90 * prediccion_estacional + 0.10 * referencia_suavizada
    )

    factor_piso = float(np.clip(
        1.0 + 0.35 * (crec - 1.0), 0.94, 1.15
    ))
    piso_parcial = ultimo_ano * factor_piso

    porcentaje_recuperacion = 0.45 if mes == 11 else 0.60 if mes == 12 else 0.40
    if prediccion_combinada < piso_parcial:
        correccion_piso = porcentaje_recuperacion * (
            piso_parcial - prediccion_combinada
        )
        prediccion_final = prediccion_combinada + correccion_piso
    else:
        correccion_piso = 0.0
        prediccion_final = prediccion_combinada

    diagnostico.update({
        "Referencia_ultimo_ano": ultimo_ano,
        "Crecimiento_YTD": float(crec),
        "Prediccion_estacional": float(prediccion_estacional),
        "Prediccion_combinada": float(prediccion_combinada),
        "Piso_parcial": float(piso_parcial),
        "Porcentaje_recuperacion": float(porcentaje_recuperacion),
        "Correccion_piso": float(correccion_piso),
        "Ajuste_aplicado": True,
    })
    return max(0.0, float(prediccion_final)), diagnostico

def normalizar_categoria_calibracion(categoria):
    import unicodedata
    texto = unicodedata.normalize("NFKD", str(categoria))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return " ".join(texto.upper().strip().split())


def aplicar_calibracion_categoria_v2(prediccion, categoria, es_combo=0):
    """Aplica factor V2 solo a categorías autorizadas y productos sin combo."""
    prediccion = max(0.0, float(prediccion))
    categoria_norm = normalizar_categoria_calibracion(categoria)
    if int(es_combo) == 1 and not APLICAR_CALIBRACION_A_COMBOS:
        return prediccion, 1.0, False, "Combo sin calibración"
    factor = float(np.clip(
        CALIBRACION_CATEGORIAS_V2.get(categoria_norm, 1.0),
        FACTOR_CALIBRACION_MIN,
        FACTOR_CALIBRACION_MAX,
    ))
    aplicada = not np.isclose(factor, 1.0)
    motivo = f"Factor V2 de {categoria_norm}" if aplicada else "Categoría sin corrección"
    return prediccion * factor, factor, aplicada, motivo


def extender_historial_hasta_octubre_2026(historial, corte=202610):
    """Completa meses faltantes hasta octubre 2026 para SKU inactivos.

    Los meses posteriores al último registro se consideran sin venta y sin
    unidades. El precio se conserva como última referencia disponible. Esto
    evita tratar julio como si fuera el mes inmediatamente anterior a noviembre.
    """
    hist = historial.sort_values("Periodo").copy()
    ultimo_periodo = int(hist["Periodo"].max())
    if ultimo_periodo >= corte:
        return hist

    ultima_fecha = pd.Timestamp(
        year=ultimo_periodo // 100,
        month=ultimo_periodo % 100,
        day=1,
    )
    fechas_faltantes = pd.date_range(
        ultima_fecha + pd.offsets.MonthBegin(1),
        pd.Timestamp(year=corte // 100, month=corte % 100, day=1),
        freq="MS",
    )
    if len(fechas_faltantes) == 0:
        return hist

    meta = hist.iloc[-1].copy()
    nuevas = []
    for fecha in fechas_faltantes:
        fila = meta.copy()
        fila["AÑO"] = int(fecha.year)
        fila["MES_NUM"] = int(fecha.month)
        fila["Periodo"] = int(fecha.year * 100 + fecha.month)
        fila["VENTA"] = 0.0
        fila["UNIDADES"] = 0.0
        # El precio se conserva como referencia; no se inventa un precio cero.
        fila["PRECIO_PROMEDIO"] = float(meta.get("PRECIO_PROMEDIO", 0.0) or 0.0)
        nuevas.append(fila)
    return pd.concat([hist, pd.DataFrame(nuevas)], ignore_index=True)


def meses_entre_periodos(periodo_inicial, periodo_final=202610):
    ai, mi = divmod(int(periodo_inicial), 100)
    af, mf = divmod(int(periodo_final), 100)
    return max(0, (af - ai) * 12 + (mf - mi))


def ajustar_noviembre_chocolate_mesa(forecast):
    """Reduce 9.4% el forecast monetario de CHOCOLATE MESA en noviembre 2026.

    El factor se aplica proporcionalmente a todos los SKU de la categoria. Las
    unidades se recalculan con el precio reconciliado para mantener coherencia.
    Diciembre no se modifica.
    """
    out = forecast.copy()
    out["Prediccion_antes_ajuste_chocolate_mesa"] = out["Prediccion_final"]
    out["Factor_ajuste_nov_chocolate_mesa"] = 1.0
    out["Ajuste_nov_chocolate_mesa_aplicado"] = False
    if out.empty:
        return out
    cat = out["CATEGORIA"].map(normalizar_categoria_calibracion)
    mascara = cat.eq("CHOCOLATE MESA") & out["Periodo"].eq(202611)
    out.loc[mascara, "Prediccion_final"] *= FACTOR_NOV_CHOCOLATE_MESA
    out.loc[mascara, "Unidades_pred"] = np.where(
        out.loc[mascara, "Precio_pred"] > 0,
        out.loc[mascara, "Prediccion_final"] / out.loc[mascara, "Precio_pred"],
        0.0,
    )
    out.loc[mascara, "Venta_estructural"] = (
        out.loc[mascara, "Unidades_pred"] * out.loc[mascara, "Precio_pred"]
    )
    out.loc[mascara, "Factor_ajuste_nov_chocolate_mesa"] = FACTOR_NOV_CHOCOLATE_MESA
    out.loc[mascara, "Ajuste_nov_chocolate_mesa_aplicado"] = True
    return out

def ajustar_diciembre_lacteos_18(forecast):
    """Ajusta diciembre de Lácteos Culinarios a un máximo de +18% vs noviembre.

    La reducción se reparte proporcionalmente entre los SKU de diciembre. El mismo
    factor se aplica a unidades para conservar coherencia entre venta y volumen.
    """
    out = forecast.copy()
    out["Prediccion_antes_control_lacteos"] = out["Prediccion_final"]
    out["Unidades_antes_control_lacteos"] = out["Unidades_pred"]
    out["Factor_control_diciembre_lacteos"] = 1.0
    out["Control_diciembre_lacteos_aplicado"] = False
    if out.empty or not APLICAR_CONTROL_BASE_DIC_LACTEOS:
        return out
    cat = out["CATEGORIA"].map(normalizar_categoria_calibracion)
    lacteos = cat.eq("LACTEOS CULINARIOS")
    nov = lacteos & out["Periodo"].eq(202611)
    dic = lacteos & out["Periodo"].eq(202612)
    total_nov = float(out.loc[nov, "Prediccion_final"].sum())
    total_dic = float(out.loc[dic, "Prediccion_final"].sum())
    objetivo_dic = total_nov * FACTOR_DIC_VS_NOV_LACTEOS
    if total_nov > 0 and total_dic > objetivo_dic:
        factor = float(objetivo_dic / total_dic)
        out.loc[dic, "Prediccion_final"] *= factor
        out.loc[dic, "Unidades_pred"] = np.where(
            out.loc[dic, "Precio_pred"] > 0,
            out.loc[dic, "Prediccion_final"] / out.loc[dic, "Precio_pred"],
            0.0,
        )
        out.loc[dic, "Venta_estructural"] = out.loc[dic, "Unidades_pred"] * out.loc[dic, "Precio_pred"]
        out.loc[dic, "Factor_control_diciembre_lacteos"] = factor
        out.loc[dic, "Control_diciembre_lacteos_aplicado"] = True
    return out


PRECIOS_WEB_VERIFICADOS = {
    # Solo coincidencias exactas de UPC y presentacion. No se usan coincidencias por nombre.
    "7501059233980": {"precio": 65.10, "fuente": "WEB - Superdulces", "url": "https://superdulces.com/products/apolv", "fecha_consulta": "2026-11-24"},
    "7501059235441": {"precio": 14.00, "fuente": "WEB - Farmacia San Pablo", "url": "https://www.farmaciasanpablo.com.mx/alimentos-y-bebidas/bebidas/aguas-purificadas/sta-maria-agua-pura-de-manantial/p/000000000007790029", "fecha_consulta": "2026-11-24"},
    "7501059200050": {"precio": 14.00, "fuente": "WEB - Veggicano", "url": "https://veggicano.com.mx/products/agua-embotellada-santa-maria-1", "fecha_consulta": "2026-11-24"},
}


def cargar_sabana_precios(archivo):
    """Lee UPC y $ Precio Venta Sugerido de la hoja SABANA DE PRECIOS."""
    if archivo is None:
        return pd.DataFrame(columns=["CODIGO", "Precio_sugerido_sabana", "Descripcion_sabana"])
    contenido = archivo.getvalue()
    tabla = pd.read_excel(io.BytesIO(contenido), sheet_name="SABANA DE PRECIOS", header=2, engine="openpyxl")
    col_upc = _buscar_columna(tabla.columns, ["UPC"])
    col_precio = _buscar_columna(tabla.columns, ["$ PRECIO VENTA SUGERIDO", "PRECIO VENTA SUGERIDO"])
    col_desc = _buscar_columna(tabla.columns, ["DESCRIPCION", "DESCRIPCIÓN"])
    if col_upc is None or col_precio is None:
        raise ValueError("La sábana debe contener las columnas UPC y $ Precio Venta Sugerido en la fila 3.")
    out = pd.DataFrame({
        "CODIGO": normalizar_codigo(tabla[col_upc]),
        "Precio_sugerido_sabana": pd.to_numeric(tabla[col_precio], errors="coerce"),
        "Descripcion_sabana": tabla[col_desc].astype(str).str.strip() if col_desc is not None else "",
    })
    out = out.loc[out["CODIGO"].ne("") & out["Precio_sugerido_sabana"].gt(0)].copy()
    # Si un UPC se repite, se conserva la mediana para evitar duplicar el forecast.
    return out.groupby("CODIGO", as_index=False).agg(
        Precio_sugerido_sabana=("Precio_sugerido_sabana", "median"),
        Descripcion_sabana=("Descripcion_sabana", "first"),
    )


def construir_referencias_precio(df, sabana, corte=202610, usar_web=True):
    """Prioridad: sábana por UPC, web exacta por UPC, último mes válido del mismo UPC."""
    h = df.loc[df["Periodo"].le(corte), ["CODIGO", "Periodo", "VENTA", "UNIDADES"]].copy()
    h["CODIGO"] = normalizar_codigo(h["CODIGO"])
    h["VENTA"] = pd.to_numeric(h["VENTA"], errors="coerce")
    h["UNIDADES"] = pd.to_numeric(h["UNIDADES"], errors="coerce")
    h = h.loc[h["VENTA"].gt(0) & h["UNIDADES"].gt(0)].copy()
    h = h.groupby(["CODIGO", "Periodo"], as_index=False)[["VENTA", "UNIDADES"]].sum()
    h["Precio_ultimo_mes"] = h["VENTA"] / h["UNIDADES"]
    ultimo = h.sort_values(["CODIGO", "Periodo"]).groupby("CODIGO", as_index=False).tail(1)
    ultimo = ultimo[["CODIGO", "Periodo", "Precio_ultimo_mes"]].rename(columns={"Periodo": "Periodo_ultimo_precio"})
    base = pd.DataFrame({"CODIGO": normalizar_codigo(df["CODIGO"]).drop_duplicates()})
    base = base.merge(ultimo, on="CODIGO", how="left", validate="one_to_one")
    if sabana is not None and not sabana.empty:
        s = sabana.copy(); s["CODIGO"] = normalizar_codigo(s["CODIGO"])
        base = base.merge(s, on="CODIGO", how="left", validate="one_to_one")
    else:
        base["Precio_sugerido_sabana"] = np.nan; base["Descripcion_sabana"] = np.nan
    web = pd.DataFrame([{"CODIGO": k, "Precio_web": v["precio"], "Fuente_web": v["fuente"],
                         "URL_precio_web": v["url"], "Fecha_precio_web": v["fecha_consulta"]}
                        for k, v in PRECIOS_WEB_VERIFICADOS.items()])
    base = base.merge(web if usar_web else web.iloc[:0], on="CODIGO", how="left", validate="one_to_one")
    base["Precio_referencia_SKU"] = base["Precio_sugerido_sabana"].combine_first(base["Precio_web"]).combine_first(base["Precio_ultimo_mes"])
    base["Fuente_precio"] = np.select(
        [base["Precio_sugerido_sabana"].notna(), base["Precio_web"].notna(), base["Precio_ultimo_mes"].notna()],
        ["SABANA - PRECIO VENTA SUGERIDO", "WEB - UPC EXACTO", "HISTORICO - ULTIMO MES"], default="SIN PRECIO")
    base["Coincidencia_codigo_sabana"] = base["Precio_sugerido_sabana"].notna()
    base["Precio_min_auditoria"] = base["Precio_referencia_SKU"] * 0.80
    base["Precio_max_auditoria"] = base["Precio_referencia_SKU"] * 1.20
    base["Calidad_referencia_precio"] = np.select(
        [base["Fuente_precio"].eq("SABANA - PRECIO VENTA SUGERIDO"), base["Fuente_precio"].eq("WEB - UPC EXACTO"),
         base["Fuente_precio"].eq("HISTORICO - ULTIMO MES")],
        ["PRECIO SUGERIDO POR UPC", "PRECIO WEB VERIFICADO POR UPC", "ULTIMO MES DEL SKU"], default="SIN REFERENCIA SKU")
    return base.set_index("CODIGO").to_dict("index")


def referencias_precio_observado(df, corte):
    """Precios por SKU a partir de pares observados, sin precios imputados."""
    h = df.loc[df["Periodo"].le(corte), ["CODIGO", "Periodo", "VENTA", "UNIDADES"]].copy()
    for col in ["VENTA", "UNIDADES"]:
        h[col] = pd.to_numeric(h[col], errors="coerce")
    h = h.loc[np.isfinite(h["VENTA"]) & np.isfinite(h["UNIDADES"])]
    h = h.loc[(h["VENTA"] > 0) & (h["UNIDADES"] > 0)].copy()
    h = h.groupby(["CODIGO", "Periodo"], as_index=False)[["VENTA", "UNIDADES"]].sum()
    h["Precio"] = h["VENTA"] / h["UNIDADES"]
    h = h.loc[np.isfinite(h["Precio"]) & h["Precio"].gt(0)].sort_values("Periodo")
    corte_mes = (corte // 100) * 12 + corte % 100
    referencias = {}
    for codigo, serie in h.groupby("CODIGO", sort=False):
        edad = corte_mes - (serie["Periodo"] // 100 * 12 + serie["Periodo"] % 100)
        ventana = serie.loc[edad.lt(12)]
        antigua = ventana.empty
        if antigua:
            ventana = serie.tail(12)
        precios = ventana["Precio"].to_numpy()
        logp = np.log(precios)
        centro = np.median(logp)
        mad = np.median(np.abs(logp - centro))
        # Tolerancia de deteccion, no conversion de piezas/cajas ni precio de mercado.
        filtro = np.abs(logp - centro) <= max(np.log(1.5), 3.5 * 1.4826 * mad)
        limpios = ventana.loc[filtro, "Precio"]
        precio = float(limpios.tail(6).median())
        inf, sup = limpios.quantile([0.10, 0.90]).tolist()
        referencias[str(codigo)] = {
            "Precio_referencia_SKU": precio,
            "Precio_min_auditoria": min(float(inf), precio / 1.25),
            "Precio_max_auditoria": max(float(sup), precio * 1.25),
            "Meses_precio_validos": len(limpios),
            "Ultimo_periodo_precio": int(ventana.loc[filtro, "Periodo"].iloc[-1]),
            "Meses_precio_atipicos": int((~filtro).sum()),
            "Calidad_referencia_precio": (
                "HISTORICO ANTIGUO" if antigua else
                "POCOS MESES" if len(limpios) < 3 else "HISTORICO RECIENTE"
            ),
        }
    return referencias


def auditar_y_reconciliar_unidades(forecast, referencias, columna_unidades):
    """Conserva el importe; deja sin estimar el volumen sin referencia del SKU."""
    out = forecast.copy()
    if out.empty:
        return out
    ref = pd.DataFrame([referencias.get(str(codigo), {}) for codigo in out["CODIGO"]], index=out.index)
    for col in ["Precio_referencia_SKU", "Precio_min_auditoria", "Precio_max_auditoria",
                "Precio_sugerido_sabana", "Precio_web", "Precio_ultimo_mes", "Periodo_ultimo_precio",
                "Fuente_precio", "Coincidencia_codigo_sabana", "Fuente_web", "URL_precio_web", "Fecha_precio_web"]:
        out[col] = ref.get(col, np.nan)
    out["Calidad_referencia_precio"] = ref.get("Calidad_referencia_precio", pd.Series(index=out.index, dtype=object)).fillna("SIN REFERENCIA SKU")
    out["Meses_precio_validos"] = np.where(out["Precio_ultimo_mes"].notna(), 1, 0)
    out["Ultimo_periodo_precio"] = out["Periodo_ultimo_precio"]
    out["Meses_precio_atipicos"] = 0
    venta = out["Prediccion_final"]
    anterior = pd.to_numeric(out[columna_unidades], errors="coerce")
    anterior = anterior.where(np.isfinite(anterior))
    precio = out["Precio_referencia_SKU"]
    implicito = venta.div(anterior.where(anterior.gt(0)))
    out["Unidades_antes_validacion"] = anterior
    out["Precio_implicito_anterior"] = implicito
    out["Alerta_precio_anterior"] = np.select(
        [venta.eq(0) & anterior.gt(0), venta.gt(0) & precio.isna(),
         venta.gt(0) & (anterior.isna() | anterior.le(0)),
         venta.gt(0) & implicito.lt(out["Precio_min_auditoria"]),
         venta.gt(0) & implicito.gt(out["Precio_max_auditoria"])],
        ["UNIDADES SIN VENTA", "SIN REFERENCIA SKU", "VENTA SIN UNIDADES VALIDAS",
         "PRECIO IMPLICITO BAJO", "PRECIO IMPLICITO ALTO"], default="SIN ALERTA")
    relacion_precio = implicito.div(precio.where(precio.gt(0)))
    out["Factor_inconsistencia_precio"] = np.maximum(
        relacion_precio, 1.0 / relacion_precio.where(relacion_precio.gt(0))
    ).replace([np.inf, -np.inf], np.nan)
    out["Prioridad_correccion_precio"] = np.select(
        [out["Alerta_precio_anterior"].isin(["UNIDADES SIN VENTA", "VENTA SIN UNIDADES VALIDAS", "SIN REFERENCIA SKU"]),
         out["Factor_inconsistencia_precio"].ge(3.0),
         out["Factor_inconsistencia_precio"].ge(1.5),
         out["Alerta_precio_anterior"].ne("SIN ALERTA")],
        ["P1 - CRITICA", "P1 - CRITICA", "P2 - ALTA", "P3 - MEDIA"],
        default="P4 - SIN ALERTA"
    )
    out["Precio_pred"] = precio
    out["Unidades_pred"] = venta.div(precio.where(precio.gt(0))).mask(venta.eq(0), 0.0)
    out["Venta_estructural"] = (out["Unidades_pred"] * precio).mask(venta.eq(0), 0.0)
    out["Diferencia_unidades"] = out["Unidades_pred"] - anterior
    return out


@st.cache_data(show_spinner=False)
def generar_forecast(df, sabana_precios, ano_objetivo=2026, metas_sucursales=None, version_modelo="sucursales-v9"):
    """Genera forecast para todos los SKU presentes en ventas 2026.

    Los SKU activos desde agosto 2026 forman el universo base. Los SKU presentes
    en 2026, pero sin actividad desde agosto, se pronostican y se etiquetan para
    que el usuario decida mediante segmentador si afectan o no los totales.
    """
    corte = ano_objetivo * 100 + 10
    df = df.loc[df.Periodo.le(corte)].copy()
    original = df.copy()
    escalas = [1.0, 1.0]
    if metas_sucursales is not None:
        factores, escalas, _ = preparar_exposicion(df, corte, metas_sucursales)
        for col in ["VENTA", "UNIDADES"]:
            df[col] = df[col].div(df["Periodo"].map(factores))
    modelos = entrenar_modelos(df, 500 if ano_objetivo == 2026 else 120)
    filas = []
    periodos = [corte + 1, corte + 2]
    codigos_2026 = set(
        df.loc[df["AÑO"] == ano_objetivo, "CODIGO"].astype(str).unique().tolist()
    )

    for codigo, historial in df.groupby("CODIGO", sort=False):
        if str(codigo) not in codigos_2026:
            continue
        hist_original = historial.sort_values("Periodo").copy()
        if hist_original.empty:
            continue

        ultimo_periodo_disponible = int(hist_original["Periodo"].max())
        incluido_base = ultimo_periodo_disponible >= ano_objetivo * 100 + 8
        estado_forecast = "INCLUIDO_BASE" if incluido_base else "NO_CONTADO_BASE"
        meses_sin_actividad = meses_entre_periodos(ultimo_periodo_disponible, corte)

        # Para SKU no contados en el forecast base, se completan con ceros hasta
        # octubre. Se mantienen separados y no afectan el total salvo elección.
        hist = (
            hist_original
            if incluido_base and ano_objetivo == 2026
            else extender_historial_hasta_octubre_2026(hist_original, corte)
        )
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

        for periodo, vd, unid, precio_prom, escala in zip(
            periodos, venta_directa, unidades, precio, escalas
        ):
            estructural = unid * precio_prom
            base = PESO_VENTA_DIRECTA * vd + PESO_ESTRUCTURAL * estructural
            final_antes_calibracion, diag = aplicar_ajuste_estacional(
                base, hist, periodo
            )
            final, factor_categoria, calibracion_aplicada, motivo_calibracion = (
                aplicar_calibracion_categoria_v2(
                    final_antes_calibracion,
                    meta.get("CATEGORIA", "SIN_DATO"),
                    meta.get("COMBOS", 0),
                )
            )
            if ano_objetivo != 2026:
                final, factor_categoria, calibracion_aplicada, motivo_calibracion = (
                    final_antes_calibracion, 1.0, False, "Sin calibración ajustada al periodo evaluado")
            final_antes_calibracion *= escala
            final *= escala
            vd *= escala
            unid *= escala
            estructural *= escala
            base *= escala
            for col in ("Referencia_ultimo_ano", "Prediccion_estacional", "Prediccion_combinada",
                        "Piso_parcial", "Correccion_piso"):
                diag[col] *= escala
            # La salida independiente se guarda para auditar antes de reconciliar.
            unidades_modelo = float(unid)
            filas.append({
                "CODIGO": str(codigo),
                "DESCRIPCION": str(meta.get("DESCRIPCION", "")),
                "CATEGORIA": str(meta.get("CATEGORIA", "SIN_DATO")).strip(),
                "COMBOS": int(meta.get("COMBOS", 0)),
                "Periodo": periodo,
                "Estado_forecast": estado_forecast,
                "Incluido_forecast_base": bool(incluido_base),
                "Ultimo_periodo_disponible": ultimo_periodo_disponible,
                "Meses_sin_actividad_a_octubre": int(meses_sin_actividad),
                "Venta_directa": float(vd),
                "Unidades_modelo_sin_reconciliar": unidades_modelo,
                "Unidades_pred": unidades_modelo,
                "Precio_modelo": float(precio_prom),
                "Precio_pred": float(precio_prom),
                "Venta_estructural_modelo": float(estructural),
                "Venta_estructural": float(estructural),
                "Prediccion_base": float(base),
                "Referencia_ultimo_ano": diag["Referencia_ultimo_ano"],
                "Crecimiento_YTD": diag["Crecimiento_YTD"],
                "Prediccion_estacional": diag["Prediccion_estacional"],
                "Prediccion_combinada": diag["Prediccion_combinada"],
                "Piso_parcial": diag["Piso_parcial"],
                "Porcentaje_recuperacion": diag["Porcentaje_recuperacion"],
                "Correccion_piso": diag["Correccion_piso"],
                "Indice_estacional": diag["Indice_estacional"],
                "Ajuste_aplicado": diag["Ajuste_aplicado"],
                "Prediccion_antes_calibracion": float(final_antes_calibracion),
                "Factor_calibracion_categoria": float(factor_categoria),
                "Calibracion_categoria_aplicada": bool(calibracion_aplicada),
                "Motivo_calibracion": motivo_calibracion,
                "Prediccion_final": float(final),
            })
    resultado = pd.DataFrame(filas)
    if resultado.empty:
        return resultado
    resultado = ajustar_noviembre_chocolate_mesa(resultado)
    resultado = ajustar_diciembre_lacteos_18(resultado)
    return auditar_y_reconciliar_unidades(
        resultado, construir_referencias_precio(original, sabana_precios if ano_objetivo == 2026 else None,
                                                  corte, usar_web=ano_objetivo == 2026), "Unidades_modelo_sin_reconciliar"
    )


def cargar_venta_semanal_noviembre(archivo):
    """Lee la tabla dinámica por categoría de la hoja PREDICCIONES NOVIEMBRE.

    Devuelve solo categorías válidas. La fila Total general se conserva aparte
    mediante la suma de categorías para evitar duplicar importes.
    """
    if archivo is None:
        return pd.DataFrame(columns=["CATEGORIA", "Venta_Semana_1", "Unidades_Semana_1"])
    contenido = archivo.getvalue()
    tabla = pd.read_excel(
        io.BytesIO(contenido), sheet_name="PREDICCIONES NOVIEMBRE",
        usecols="A:D", engine="openpyxl",
    )
    renombres = {}
    for col in tabla.columns:
        nombre = _normalizar_encabezado(col)
        if nombre == "CATEGORIA": renombres[col] = "CATEGORIA"
        elif nombre == "VENTA SEMANAL 1": renombres[col] = "Venta_Semana_1"
        elif nombre == "UNIDADES VENDIDAS": renombres[col] = "Unidades_Semana_1"
    tabla = tabla.rename(columns=renombres)
    requeridas = {"CATEGORIA", "Venta_Semana_1"}
    if not requeridas.issubset(tabla.columns):
        raise ValueError(
            "La hoja PREDICCIONES NOVIEMBRE debe contener CATEGORIA y Venta Semanal 1."
        )
    if "Unidades_Semana_1" not in tabla.columns:
        tabla["Unidades_Semana_1"] = 0.0
    tabla["CATEGORIA"] = tabla["CATEGORIA"].astype(str).str.strip()
    tabla["Categoria_normalizada"] = tabla["CATEGORIA"].map(normalizar_categoria_calibracion)
    tabla["Venta_Semana_1"] = pd.to_numeric(tabla["Venta_Semana_1"], errors="coerce")
    tabla["Unidades_Semana_1"] = pd.to_numeric(tabla["Unidades_Semana_1"], errors="coerce")
    tabla = tabla[
        tabla["Venta_Semana_1"].notna()
        & ~tabla["Categoria_normalizada"].isin(["TOTAL GENERAL", "NAN", "NONE", ""])
    ].copy()
    return (tabla.groupby(["CATEGORIA", "Categoria_normalizada"], as_index=False)
            .agg(Venta_Semana_1=("Venta_Semana_1", "sum"),
                 Unidades_Semana_1=("Unidades_Semana_1", "sum")))


def construir_comparativo_semanal(forecast_filtrado, venta_semanal, peso_real=0.30):
    """Compara forecast original, ritmo lineal y nowcast ponderado de noviembre."""
    nov = forecast_filtrado[forecast_filtrado["Periodo"] == 202611].copy()
    if nov.empty or venta_semanal.empty:
        return pd.DataFrame()
    base = (nov.assign(Categoria_normalizada=nov["CATEGORIA"].map(normalizar_categoria_calibracion))
              .groupby("Categoria_normalizada", as_index=False)
              .agg(Prediccion_Original=("Prediccion_final", "sum")))
    comp = venta_semanal.merge(base, on="Categoria_normalizada", how="inner")
    dias_transcurridos = 7.0
    dias_mes = 30.0
    comp["Avance_Calendario"] = dias_transcurridos / dias_mes
    comp["Cumplimiento_Forecast"] = np.where(
        comp["Prediccion_Original"] > 0,
        comp["Venta_Semana_1"] / comp["Prediccion_Original"], np.nan,
    )
    comp["Proyeccion_Ritmo"] = comp["Venta_Semana_1"] * dias_mes / dias_transcurridos

    # Conservamos el valor del modelo solo para auditoría. La columna visible
    # Prediccion_Original se corrige directamente, sin crear una barra adicional.
    comp["Prediccion_Modelo_Sin_Ajuste"] = comp["Prediccion_Original"]
    comp["Peso_Correccion_Ritmo"] = 0.0
    comp["Tope_Correccion"] = np.nan

    lacteos = comp["Categoria_normalizada"].eq("LACTEOS CULINARIOS")
    cereales = comp["Categoria_normalizada"].eq("CEREALES")

    # Lácteos: forecast y ritmo están próximos. Se recupera 90% de la brecha.
    comp.loc[lacteos, "Peso_Correccion_Ritmo"] = 0.90

    # Cereales: la semana 1 está muy por encima del modelo. Se recupera 50%
    # de la brecha, pero con tope de 1.60x para no extrapolar una posible promoción.
    comp.loc[cereales, "Peso_Correccion_Ritmo"] = 0.50
    comp.loc[cereales, "Tope_Correccion"] = 1.60

    comp["Prediccion_Original"] = (
        comp["Prediccion_Modelo_Sin_Ajuste"]
        + comp["Peso_Correccion_Ritmo"]
        * (comp["Proyeccion_Ritmo"] - comp["Prediccion_Modelo_Sin_Ajuste"])
    )
    comp.loc[cereales, "Prediccion_Original"] = np.minimum(
        comp.loc[cereales, "Prediccion_Original"],
        comp.loc[cereales, "Prediccion_Modelo_Sin_Ajuste"] * 1.60,
    )
    comp["Factor_Correccion_Aplicado"] = np.where(
        comp["Prediccion_Modelo_Sin_Ajuste"] > 0,
        comp["Prediccion_Original"] / comp["Prediccion_Modelo_Sin_Ajuste"],
        1.0,
    )
    comp["Correccion_Aplicada"] = np.select(
        [lacteos, cereales],
        ["Lácteos: 90% de brecha", "Cereales: 50% de brecha; tope 1.60x"],
        default="Sin corrección",
    )

    # El nowcast se recalcula desde el forecast ya corregido.
    comp["Cumplimiento_Forecast"] = np.where(
        comp["Prediccion_Original"] > 0,
        comp["Venta_Semana_1"] / comp["Prediccion_Original"], np.nan,
    )
    comp["Nowcast_Ponderado"] = (
        (1.0 - peso_real) * comp["Prediccion_Original"]
        + peso_real * comp["Proyeccion_Ritmo"]
    )
    comp["Brecha_Ritmo_pp"] = (
        comp["Cumplimiento_Forecast"] - comp["Avance_Calendario"]
    ) * 100
    comp["Estado"] = np.select(
        [comp["Brecha_Ritmo_pp"] >= 5, comp["Brecha_Ritmo_pp"] <= -5],
        ["ADELANTADA", "REZAGADA"], default="EN RANGO",
    )
    return comp


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


def tabla_sucursales(df):
    serie = pd.to_numeric(df["CANTIDAD_SUCURSALES"], errors="coerce")
    validas = serie.where(serie.gt(0) & np.isfinite(serie) & serie.mod(1).eq(0))
    g = validas.groupby(df["Periodo"])
    conflictos = g.nunique().loc[lambda s: s.gt(1)].index.tolist()
    if conflictos:
        raise ValueError(f"Hay cantidades de sucursales contradictorias en: {conflictos}")
    return g.first().sort_index()


def preparar_exposicion(df, corte, metas):
    serie = tabla_sucursales(df.loc[df["Periodo"].le(corte)])
    if serie.dropna().empty:
        raise ValueError("No hay cantidades de sucursales válidas antes del corte.")
    imputada = serie.ffill()
    if imputada.isna().any():
        raise ValueError("Falta un conteo al inicio del histórico. No se usarán meses futuros para completarlo.")
    base = float(imputada.iloc[-1])
    metas = np.asarray(metas, dtype=float)
    if not (np.isfinite(metas) & (metas > 0) & (metas % 1 == 0)).all():
        raise ValueError("Indica un número entero positivo de sucursales para cada mes futuro.")
    return imputada.div(base), metas / base, serie.index[serie.isna()].tolist()


def metricas_bt(datos, real, pred):
    y = datos[real].to_numpy(dtype=float)
    p = datos[pred].to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    e = p - y
    s = np.abs(y).sum()
    var = np.square(y - y.mean()).sum() if y.size else 0
    return {"N": y.size, "WAPE (%)": 100 * np.abs(e).sum() / s if s else np.nan,
            "MAE": np.abs(e).mean() if y.size else np.nan,
            "RMSE": np.sqrt(np.square(e).mean()) if y.size else np.nan,
            "R²": 1 - np.square(e).sum() / var if y.size > 1 and var > 0 else np.nan}


@st.cache_data(show_spinner=False)
def backtest_sucursales_2025(df, nov, dic):
    corte = 202510
    periodos = [202511, 202512]
    crudo = df.loc[df["REGISTRO_ORIGINAL"].eq(True)].copy()
    entrenamiento = crudo.loc[crudo.Periodo.le(corte)].copy()
    if not set([202411, 202412, corte]).issubset(entrenamiento.Periodo):
        raise ValueError("Se requieren noviembre y diciembre de 2024 y octubre de 2025.")
    reales = crudo.loc[crudo.Periodo.isin(periodos)].copy()
    if set(reales.Periodo) != set(periodos):
        raise ValueError("Faltan las ventas reales de noviembre o diciembre de 2025.")
    conteos = tabla_sucursales(entrenamiento)
    entrenamiento = completar_calendario(entrenamiento)
    entrenamiento["CANTIDAD_SUCURSALES"] = entrenamiento.Periodo.map(conteos)
    resultados = []
    for metas in [None, (nov, dic)]:
        f = generar_forecast(entrenamiento, None, ano_objetivo=2025, metas_sucursales=metas)
        resultados.append(f[["CODIGO", "Periodo", "DESCRIPCION", "CATEGORIA", "COMBOS",
                             "Prediccion_final", "Unidades_pred"]])
    a, b = resultados
    llaves = ["CODIGO", "Periodo"]
    z = a.merge(b[llaves + ["Prediccion_final", "Unidades_pred"]], on=llaves,
                validate="one_to_one", suffixes=("_sin", "_con"))
    z = z.merge(reales[llaves + ["VENTA", "UNIDADES"]], on=llaves, how="left", validate="one_to_one")
    z["Registro_real_en_archivo"] = z.VENTA.notna()
    z[["VENTA", "UNIDADES"]] = z[["VENTA", "UNIDADES"]].fillna(0)
    fuera = reales.loc[~reales.CODIGO.isin(a.CODIGO)]
    _, _, imputados = preparar_exposicion(entrenamiento, corte, (nov, dic))
    return z, fuera.groupby("Periodo")["VENTA"].sum().to_dict(), imputados


def mostrar_backtest_2025(df):
    with st.expander("Backtest Nov-Dic 2025: con y sin sucursales", expanded=True):
        st.caption("Corte: octubre 2025. Se proyectan ambos meses sin usar ventas ni precios de 2026. "
                   "Diciembre usa rezagos pronosticados, no las ventas reales de noviembre. "
                   "Ambos brazos usan 120 iteraciones y las mismas reglas estacionales: prueba retrospectiva, no validación independiente.")
        try:
            historia = tabla_sucursales(df.loc[df.Periodo.le(202510)]).dropna()
        except ValueError as exc:
            st.warning(str(exc)); return
        if historia.empty:
            st.info("Carga los Excel con CANTIDAD_SUCURSALES para probar esta variante."); return
        base = int(historia.iloc[-1])
        a, b = st.columns(2)
        nov = int(a.number_input("Sucursales supuestas Nov 2025", min_value=1, value=base))
        dic = int(b.number_input("Sucursales supuestas Dic 2025", min_value=1, value=base))
        st.caption("Se propone el último dato a octubre para no usar el cierre real posterior. "
                   "Cualquier cambio manual se evalúa como escenario. Los meses históricos sin cifra se completan con la última disponible.")
        if st.checkbox("Mostrar backtest 2025", key="mostrar_bt_2025"):
            try:
                with st.spinner("Comparando dos modelos CatBoost..."):
                    z, fuera, imputados = backtest_sucursales_2025(df, nov, dic)
            except (ValueError, ImportError) as exc:
                st.warning(str(exc)); return
            tipo = st.selectbox("Combos (backtest)", ["SIN COMBOS", "TODOS", "COMBOS"])
            if tipo != "TODOS":
                z = z.loc[z.COMBOS.eq(int(tipo == "COMBOS"))]
            cat = st.selectbox("Categoría (backtest)", ["TODAS"] + sorted(z.CATEGORIA.unique()))
            if cat != "TODAS":
                z = z.loc[z.CATEGORIA.eq(cat)]
            if z.empty:
                st.info("No hay SKU en esta selección."); return
            filas = []
            for nombre, t in [("Nov 2025", z.loc[z.Periodo.eq(202511)]),
                              ("Dic 2025", z.loc[z.Periodo.eq(202512)]), ("Ambos meses", z)]:
                for objetivo, real, pred in [("Venta (MXN)", "VENTA", "Prediccion_final"),
                                              ("Unidades", "UNIDADES", "Unidades_pred")]:
                    comun = t.loc[t[pred + "_sin"].notna() & t[pred + "_con"].notna()]
                    for suf, modo in [("_sin", "Sin sucursales"), ("_con", "Con sucursales")]:
                        filas.append({"Periodo": nombre, "Objetivo": objetivo, "Modelo": modo,
                                      **metricas_bt(comun, real, pred + suf)})
            metricas = pd.DataFrame(filas)
            st.dataframe(metricas.style.format({"WAPE (%)": "{:.2f}", "MAE": "{:,.2f}",
                                               "RMSE": "{:,.2f}", "R²": "{:.4f}"}, na_rep="N/D"),
                         hide_index=True, use_container_width=True)
            resumen = z.groupby("Periodo")[["VENTA", "Prediccion_final_sin", "Prediccion_final_con"]].sum()
            st.bar_chart(resumen)
            st.dataframe(resumen.style.format("${:,.2f}"), use_container_width=True)
            st.caption(f"Métricas por SKU-mes del catálogo conocido al corte; ausencia de fila real se evalúa como cero. "
                       f"Venta real total de SKU fuera de ese catálogo (sin filtros): ${sum(fuera.values()):,.0f}. "
                       f"Meses con sucursales arrastradas: {imputados or 'ninguno'}. "
                       "Unidades: solo filas con estimación en ambos brazos, con precio del último mes válido antes del corte.")
            st.download_button("Descargar detalle backtest 2025", z.to_csv(index=False).encode("utf-8-sig"),
                               "backtest_2025_sku.csv", "text/csv")
            st.download_button("Descargar métricas backtest 2025", metricas.to_csv(index=False).encode("utf-8-sig"),
                               "backtest_2025_metricas.csv", "text/csv")


archivos = st.file_uploader(
    "Carga los archivos de ventas 2024, 2025 y 2026",
    type=["xlsx"],
    accept_multiple_files=True,
)
archivo_semanal = st.file_uploader(
    "Carga PREDICCIONES NOVIEMBRE.xlsx para comparar la venta real de la semana 1",
    type=["xlsx"],
    accept_multiple_files=False,
    key="archivo_venta_semanal_noviembre",
    help="Se lee la tabla por categoría de la hoja PREDICCIONES NOVIEMBRE.",
)
archivo_sabana = st.file_uploader(
    "Carga la SÁBANA DE PRECIOS para conciliar unidades por UPC",
    type=["xlsx"], accept_multiple_files=False, key="archivo_sabana_precios",
    help="Prioridad: precio sugerido de la sábana por UPC; después web verificada por UPC; finalmente último mes válido del SKU.",
)

if archivos:
    try:
        # La carga no usa caché: cada ejecución procesa exactamente los archivos visibles.
        df, diagnostico_carga = cargar_archivos(archivos)
        resumen_anual = (
            df.groupby("AÑO", as_index=False)
            .agg(SKU=("CODIGO", "nunique"), Registros=("Periodo", "size"),
                 Desde=("Periodo", "min"), Hasta=("Periodo", "max"))
        )
        anos_cargados = set(resumen_anual["AÑO"].astype(int).tolist())
        st.success("Los archivos fueron leídos nuevamente sin utilizar caché.")
        st.dataframe(resumen_anual, use_container_width=True, hide_index=True)
        with st.expander("Diagnóstico detallado de cada hoja"):
            st.dataframe(diagnostico_carga, use_container_width=True, hide_index=True)
        mostrar_backtest_2025(df)
        sabana_precios = cargar_sabana_precios(archivo_sabana)
        faltantes = sorted({2024, 2025, 2026} - anos_cargados)
        if faltantes:
            st.error(
                f"No se cargaron datos válidos de {faltantes}. Revisa en el diagnóstico "
                "qué columna falta en las hojas correspondientes."
            )
            st.stop()

        metas_2026 = None
        if st.checkbox("Usar sucursales en el forecast de 2026", value=False):
            historico_locales = tabla_sucursales(df.loc[df.Periodo.le(202610)]).dropna()
            if historico_locales.empty:
                st.warning("Falta CANTIDAD_SUCURSALES; el forecast de 2026 conservará el modelo base.")
            else:
                ultimo = int(historico_locales.iloc[-1])
                izq, der = st.columns(2)
                metas_2026 = (int(izq.number_input("Sucursales Nov 2026", min_value=1, value=ultimo)),
                              int(der.number_input("Sucursales Dic 2026", min_value=1, value=ultimo)))
                st.caption("Cantidades del escenario, no cierres confirmados. Los huecos históricos se completan con el último conteo disponible.")
        import hashlib
        huella = hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
            + (archivo_sabana.getvalue() if archivo_sabana is not None else b"")
        ).hexdigest()
        clave_forecast = (huella, metas_2026)
        if st.session_state.get("clave_forecast_v9") != clave_forecast:
            st.session_state.pop("forecast_2026_upc_sabana_web_v8", None)
        col_boton, _ = st.columns([1, 2])
        with col_boton:
            if st.button("Limpiar forecast anterior"):
                st.session_state.pop("forecast_2026_upc_sabana_web_v8", None)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()

        if "forecast_2026_upc_sabana_web_v8" not in st.session_state:
            st.info("Genera el forecast para habilitar la gráfica de noviembre-diciembre de 2026.")

        if st.button("Generar forecast final Nov-Dic 2026", type="primary"):
            with st.spinner("Entrenando el modelo y generando el forecast..."):
                st.session_state["forecast_2026_upc_sabana_web_v8"] = generar_forecast(df, sabana_precios, metas_sucursales=metas_2026)
                st.session_state["clave_forecast_v9"] = clave_forecast
            st.rerun()

        if "forecast_2026_upc_sabana_web_v8" in st.session_state:
            forecast = st.session_state["forecast_2026_upc_sabana_web_v8"]
            if forecast.empty:
                st.error("No se encontraron SKU activos con datos recientes de 2026.")
                st.stop()

            st.markdown("### Filtros")
            st.markdown("#### Universo de SKU 2026")
            universo_sku = st.segmented_control(
                "¿Qué SKU deben afectar la gráfica y los totales?",
                options=[
                    "SOLO INCLUIDOS BASE",
                    "INCLUIR NO CONTADOS",
                    "SOLO NO CONTADOS",
                ],
                default="SOLO INCLUIDOS BASE",
                help=(
                    "Incluidos base: SKU con datos desde agosto 2026. "
                    "Incluir no contados: suma también SKU presentes en 2026 sin actividad reciente. "
                    "Solo no contados: permite revisarlos por separado sin mezclar el resto."
                ),
            )
            if universo_sku is None:
                universo_sku = "SOLO INCLUIDOS BASE"

            if universo_sku == "SOLO INCLUIDOS BASE":
                forecast_universo = forecast[forecast["Incluido_forecast_base"]].copy()
            elif universo_sku == "SOLO NO CONTADOS":
                forecast_universo = forecast[~forecast["Incluido_forecast_base"]].copy()
            else:
                forecast_universo = forecast.copy()

            conteo_base = forecast.loc[forecast["Incluido_forecast_base"], "CODIGO"].nunique()
            conteo_no_contado = forecast.loc[~forecast["Incluido_forecast_base"], "CODIGO"].nunique()
            u1, u2, u3 = st.columns(3)
            u1.metric("SKU incluidos base", f"{conteo_base:,}")
            u2.metric("SKU 2026 no contados", f"{conteo_no_contado:,}")
            u3.metric("SKU en universo seleccionado", f"{forecast_universo['CODIGO'].nunique():,}")

            if universo_sku != "SOLO INCLUIDOS BASE":
                st.warning(
                    "Los SKU no contados tienen menor evidencia reciente. Sus meses faltantes hasta "
                    "octubre se completan con venta/unidades cero. Úsalos como escenario de sensibilidad."
                )

            categorias = ["TODAS"] + sorted(
                forecast_universo["CATEGORIA"].dropna().astype(str).str.strip().unique().tolist()
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

            universo = forecast_universo.copy()
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
                df, forecast_universo, categoria, combo, producto
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
            sku_mostrados = (
                universo["CODIGO"].astype(str).nunique()
                if producto == "TODOS" else 1
            )
            k4.metric("SKU incluidos", f"{sku_mostrados:,}")

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

            # --------------------------------------------------------------
            # Seguimiento de noviembre con venta real de la primera semana
            # --------------------------------------------------------------
            # Valor predeterminado disponible también para la tabla SKU.
            peso_real = 0.30
            st.markdown("### Seguimiento semanal de noviembre")
            if archivo_semanal is None:
                st.info(
                    "Carga PREDICCIONES NOVIEMBRE.xlsx arriba para comparar el forecast "
                    "con la venta real acumulada de la semana 1."
                )
            else:
                venta_semanal = cargar_venta_semanal_noviembre(archivo_semanal)
                peso_real = st.slider(
                    "Peso de la venta real semanal en el nowcast",
                    min_value=0, max_value=100, value=30, step=5,
                    format="%d%%",
                    help=(
                        "30% conserva estabilidad del forecast y permite que el ritmo real "
                        "actualice gradualmente la proyección mensual."
                    ),
                ) / 100.0
                comparativo = construir_comparativo_semanal(
                    forecast_universo, venta_semanal, peso_real
                )
                st.info(
                    "La barra naranja ya incluye la corrección selectiva: Lácteos recupera 90% "
                    "de la brecha hacia ritmo; Cereales recupera 50% con máximo de 1.60x. "
                    "Las demás categorías conservan el forecast del modelo."
                )
                if categoria != "TODAS":
                    cat_norm = normalizar_categoria_calibracion(categoria)
                    comparativo = comparativo[
                        comparativo["Categoria_normalizada"] == cat_norm
                    ].copy()

                categorias_disponibles = sorted(comparativo["CATEGORIA"].unique().tolist())
                categorias_seleccionadas = st.multiselect(
                    "Segmentador de categorías para comparar proyecciones",
                    options=categorias_disponibles,
                    default=categorias_disponibles,
                    help="Lácteos Culinarios y Cereales ya incorporan la corrección selectiva en la barra naranja.",
                )
                comparativo = comparativo[
                    comparativo["CATEGORIA"].isin(categorias_seleccionadas)
                ].copy()

                escenarios = st.segmented_control(
                    "Escenarios visibles",
                    options=["TODOS", "FORECAST VS NOWCAST", "FORECAST VS RITMO"],
                    default="TODOS",
                ) or "TODOS"

                if comparativo.empty:
                    st.warning("No hay categorías coincidentes entre el forecast y la tabla semanal.")
                else:
                    total_semana = comparativo["Venta_Semana_1"].sum()
                    total_original = comparativo["Prediccion_Original"].sum()
                    total_ritmo = comparativo["Proyeccion_Ritmo"].sum()
                    total_nowcast = comparativo["Nowcast_Ponderado"].sum()
                    avance = total_semana / total_original if total_original > 0 else np.nan
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Venta real semana 1", f"${total_semana:,.0f}")
                    s2.metric("Forecast original Nov", f"${total_original:,.0f}")
                    s3.metric("Proyección por ritmo", f"${total_ritmo:,.0f}",
                              f"{(total_ritmo/total_original-1)*100:+.1f}% vs forecast" if total_original else None)
                    s4.metric("Nowcast ponderado", f"${total_nowcast:,.0f}",
                              f"{(total_nowcast/total_original-1)*100:+.1f}% vs forecast" if total_original else None)
                    st.caption(
                        f"Avance acumulado: {avance:.1%}. Avance calendario de referencia: 23.3% (7 de 30 días)."
                    )

                    columnas_escenario = {
                        "TODOS": ["Prediccion_Original", "Proyeccion_Ritmo", "Nowcast_Ponderado"],
                        "FORECAST VS NOWCAST": ["Prediccion_Original", "Nowcast_Ponderado"],
                        "FORECAST VS RITMO": ["Prediccion_Original", "Proyeccion_Ritmo"],
                    }[escenarios]
                    etiquetas = {
                        "Prediccion_Original": "Forecast original corregido",
                        "Proyeccion_Ritmo": "Proyección por ritmo",
                        "Nowcast_Ponderado": "Nowcast ponderado",
                    }
                    grafica_comp = comparativo.melt(
                        id_vars=["CATEGORIA"], value_vars=columnas_escenario,
                        var_name="Escenario", value_name="Valor",
                    )
                    grafica_comp["Escenario"] = grafica_comp["Escenario"].map(etiquetas)
                    barras = alt.Chart(grafica_comp).mark_bar().encode(
                        x=alt.X("CATEGORIA:N", title=None, sort="-y",
                                axis=alt.Axis(labelAngle=-35, labelLimit=180)),
                        xOffset="Escenario:N",
                        y=alt.Y("Valor:Q", title="Venta proyectada", axis=alt.Axis(format="$,.2s")),
                        color=alt.Color(
                            "Escenario:N", title="Escenario",
                            scale=alt.Scale(
                                domain=["Forecast original corregido", "Proyección por ritmo", "Nowcast ponderado"],
                                range=["#F59E0B", "#EF4444", "#10B981"],
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("CATEGORIA:N", title="Categoría"),
                            alt.Tooltip("Escenario:N", title="Escenario"),
                            alt.Tooltip("Valor:Q", title="Valor", format="$,.2f"),
                        ],
                    ).properties(height=430, title="Comparación de proyecciones de noviembre por categoría")
                    # Dos gráficas adicionales con el mismo histórico mensual.
                    # Noviembre se actualiza con la venta semanal. Para evitar una caída
                    # artificial en diciembre, se propaga parcialmente el ajuste relativo
                    # de noviembre. Configuración prudente para semana 1: 40%, con límite ±15%.
                    st.markdown("#### Ajuste de continuidad hacia diciembre")
                    col_prop, col_lim = st.columns(2)
                    with col_prop:
                        propagacion_dic = st.slider(
                            "Propagación del ajuste de noviembre a diciembre",
                            min_value=0, max_value=100, value=40, step=5,
                            format="%d%%",
                            help=(
                                "Semana 1 recomendada: 40%. Traslada solo una parte de la "
                                "desviación relativa de noviembre hacia diciembre."
                            ),
                        ) / 100.0
                    with col_lim:
                        limite_ajuste_dic = st.slider(
                            "Límite máximo del ajuste de diciembre",
                            min_value=5, max_value=30, value=15, step=5,
                            format="±%d%%",
                            help="Protege diciembre contra una primera semana atípica.",
                        ) / 100.0

                    def relacion_historica_dic_nov_lacteos():
                        """Mediana robusta Dic/Nov 2024-2025 para Lacteos Culinarios."""
                        cat_obj = "LACTEOS CULINARIOS"
                        hist_cat = df[
                            df["CATEGORIA"].map(normalizar_categoria_calibracion).eq(cat_obj)
                            & df["AÑO"].isin([2024, 2025])
                            & df["MES_NUM"].isin([11, 12])
                        ].groupby(["AÑO", "MES_NUM"], as_index=False)["VENTA"].sum()
                        ratios = []
                        for ano in [2024, 2025]:
                            nov_h = hist_cat.loc[
                                (hist_cat["AÑO"] == ano) & (hist_cat["MES_NUM"] == 11), "VENTA"
                            ].sum()
                            dic_h = hist_cat.loc[
                                (hist_cat["AÑO"] == ano) & (hist_cat["MES_NUM"] == 12), "VENTA"
                            ].sum()
                            if nov_h > 0 and dic_h > 0:
                                ratios.append(dic_h / nov_h)
                        if not ratios:
                            return 1.0 + CRECIMIENTO_BASE_DIC_LACTEOS, []
                        return float(np.median(ratios)), ratios

                    categoria_actual_norm = (
                        normalizar_categoria_calibracion(categoria)
                        if categoria != "TODAS" else "TODAS"
                    )
                    seleccion_exclusiva_lacteos = categoria_actual_norm == "LACTEOS CULINARIOS"
                    relacion_hist_lacteos, ratios_hist_lacteos = relacion_historica_dic_nov_lacteos()

                    def ajustar_diciembre(valor_noviembre_escenario):
                        if total_original <= 0 or dic <= 0:
                            return float(dic), 1.0, 0.0, "Sin ajuste"
                        factor_noviembre = float(valor_noviembre_escenario) / float(total_original)

                        # Regla especial solo cuando el filtro muestra exclusivamente Lacteos Culinarios.
                        # Usa noviembre actualizado como ancla y limita diciembre a -5% / +10%.
                        if APLICAR_TOPE_DICIEMBRE_LACTEOS and seleccion_exclusiva_lacteos:
                            ratio_base = (
                                relacion_hist_lacteos
                                if USAR_RELACION_HISTORICA_LACTEOS
                                else 1.0 + CRECIMIENTO_BASE_DIC_LACTEOS
                            )
                            ratio_controlado = float(np.clip(
                                ratio_base,
                                PISO_DIC_VS_NOV_LACTEOS,
                                TOPE_DIC_VS_NOV_LACTEOS,
                            ))
                            dic_controlado = float(valor_noviembre_escenario) * ratio_controlado
                            ajuste_vs_dic_original = (
                                dic_controlado / float(dic) - 1.0 if dic > 0 else 0.0
                            )
                            metodo = (
                                f"Lácteos: Nov escenario × {ratio_controlado:.3f} "
                                f"(relación histórica limitada entre "
                                f"{PISO_DIC_VS_NOV_LACTEOS:.2f} y {TOPE_DIC_VS_NOV_LACTEOS:.2f})"
                            )
                            return dic_controlado, factor_noviembre, ajuste_vs_dic_original, metodo

                        # Para el resto de categorías conserva la propagación parcial existente.
                        ajuste_sin_limite = (factor_noviembre - 1.0) * propagacion_dic
                        ajuste_limitado = float(np.clip(
                            ajuste_sin_limite, -limite_ajuste_dic, limite_ajuste_dic
                        ))
                        metodo = "Propagación parcial sobre diciembre original"
                        return (
                            float(dic) * (1.0 + ajuste_limitado),
                            factor_noviembre,
                            ajuste_limitado,
                            metodo,
                        )

                    dic_ritmo, factor_nov_ritmo, ajuste_dic_ritmo, metodo_dic_ritmo = ajustar_diciembre(total_ritmo)
                    dic_nowcast, factor_nov_nowcast, ajuste_dic_nowcast, metodo_dic_nowcast = ajustar_diciembre(total_nowcast)

                    d1, d2 = st.columns(2)
                    d1.metric(
                        "Diciembre ajustado por ritmo", f"${dic_ritmo:,.0f}",
                        f"{ajuste_dic_ritmo:+.1%} vs diciembre original",
                    )
                    d2.metric(
                        "Diciembre ajustado por nowcast", f"${dic_nowcast:,.0f}",
                        f"{ajuste_dic_nowcast:+.1%} vs diciembre original",
                    )
                    if seleccion_exclusiva_lacteos:
                        ratios_texto = ", ".join(f"{r:.3f}" for r in ratios_hist_lacteos) or "sin historial válido"
                        st.success(
                            "Control especial activo para Lácteos Culinarios: diciembre se calcula "
                            "desde noviembre actualizado y queda limitado entre -5% y +18%."
                        )
                        st.caption(
                            f"Relaciones históricas Dic/Nov disponibles: {ratios_texto}. "
                            f"Método nowcast: {metodo_dic_nowcast}."
                        )
                    else:
                        st.caption(
                            "Fórmula general: Dic. ajustado = Dic. original × [1 + límite((Nov. escenario / "
                            "Nov. original − 1) × propagación)]."
                        )

                    octubre_puente = historico[
                        historico["Fecha"] == pd.Timestamp("2026-10-01")
                    ][["Fecha", "Valor"]].copy()
                    if octubre_puente.empty:
                        octubre_puente = historico.sort_values("Fecha").tail(1)[["Fecha", "Valor"]].copy()

                    def crear_grafica_escenario(valor_noviembre, valor_diciembre, nombre, color, subtitulo):
                        puntos = pd.DataFrame({
                            "Fecha": [pd.Timestamp("2026-11-01"), pd.Timestamp("2026-12-01")],
                            "Valor": [float(valor_noviembre), float(valor_diciembre)],
                            "Serie": [nombre, nombre],
                        })
                        puente = octubre_puente.copy()
                        puente["Serie"] = nombre
                        linea = pd.concat(
                            [puente[["Fecha", "Valor", "Serie"]], puntos],
                            ignore_index=True,
                        ).sort_values("Fecha")
                        tooltip_escenario = [
                            alt.Tooltip("Fecha:T", title="Mes", format="%B %Y"),
                            alt.Tooltip("Valor:Q", title="Venta", format="$,.2f"),
                            alt.Tooltip("Serie:N", title="Escenario"),
                        ]
                        linea_real_esc = alt.Chart(historico).mark_line(
                            color="#3B82F6", strokeWidth=3.2, interpolate="monotone"
                        ).encode(x=eje_x, y=eje_y, tooltip=tooltip)
                        puntos_reales_esc = alt.Chart(historico).mark_circle(
                            color="#3B82F6", size=44, opacity=0.9
                        ).encode(x=eje_x, y=eje_y, tooltip=tooltip)
                        linea_esc = alt.Chart(linea).mark_line(
                            color=color, strokeWidth=3.5, strokeDash=[8, 5],
                            interpolate="monotone",
                        ).encode(x=eje_x, y=eje_y, tooltip=tooltip_escenario)
                        puntos_esc = alt.Chart(puntos).mark_circle(
                            color=color, size=125, stroke="white", strokeWidth=1.5
                        ).encode(x=eje_x, y=eje_y, tooltip=tooltip_escenario)
                        return alt.layer(
                            linea_real_esc, puntos_reales_esc, linea_esc, puntos_esc,
                            regla, etiqueta,
                        ).properties(
                            height=520,
                            title=alt.TitleParams(
                                text=f"Evolución mensual y {nombre.lower()} Nov-Dic 2026",
                                subtitle=subtitulo,
                                anchor="start", fontSize=18, subtitleFontSize=12,
                                offset=16,
                            ),
                        ).configure_view(strokeWidth=0).configure_axis(
                            labelFontSize=12, titleFontSize=13
                        )

                    st.markdown("### Evolución mensual con proyección por ritmo")
                    chart_ritmo = crear_grafica_escenario(
                        total_ritmo,
                        dic_ritmo,
                        "Proyección por ritmo",
                        "#EF4444",
                        (f"Azul: ventas reales | Rojo: ritmo; diciembre anclado a noviembre con tope +18%" if seleccion_exclusiva_lacteos else f"Azul: ventas reales | Rojo: ritmo; diciembre propaga {propagacion_dic:.0%} del ajuste, limitado a ±{limite_ajuste_dic:.0%}"),
                    )
                    st.altair_chart(chart_ritmo, use_container_width=True)
                    st.caption(
                        "Noviembre extrapola la venta de la semana 1 a 30 días. Diciembre recibe solo "
                        f"{propagacion_dic:.0%} del cambio relativo, con una protección máxima de ±{limite_ajuste_dic:.0%}."
                    )

                    st.markdown("### Evolución mensual con nowcast ponderado")
                    chart_nowcast = crear_grafica_escenario(
                        total_nowcast,
                        dic_nowcast,
                        "Nowcast ponderado",
                        "#10B981",
                        (f"Azul: ventas reales | Verde: nowcast con {peso_real:.0%}; diciembre anclado a noviembre con tope +18%" if seleccion_exclusiva_lacteos else f"Azul: ventas reales | Verde: nowcast con {peso_real:.0%} de peso semanal; diciembre ajustado con propagación parcial"),
                    )
                    st.altair_chart(chart_nowcast, use_container_width=True)
                    st.caption(
                        "El nowcast combina el forecast original de noviembre con la proyección por ritmo. "
                        "Diciembre mantiene la estructura del forecast, pero incorpora una fracción controlada del cambio de noviembre."
                    )

                    st.markdown("### Comparación de proyecciones por categoría")
                    st.altair_chart(barras, use_container_width=True)

                    resumen_escenarios_dic = pd.DataFrame({
                        "Escenario": ["Forecast original", "Proyección por ritmo", "Nowcast ponderado"],
                        "Noviembre": [total_original, total_ritmo, total_nowcast],
                        "Diciembre": [dic, dic_ritmo, dic_nowcast],
                        "Factor_Noviembre_vs_original": [1.0, factor_nov_ritmo, factor_nov_nowcast],
                        "Ajuste_propagado_Diciembre": [0.0, ajuste_dic_ritmo, ajuste_dic_nowcast],
                        "Metodo_Diciembre": ["Forecast original", metodo_dic_ritmo, metodo_dic_nowcast],
                    })
                    st.dataframe(
                        resumen_escenarios_dic.style.format({
                            "Noviembre": "${:,.2f}", "Diciembre": "${:,.2f}",
                            "Factor_Noviembre_vs_original": "{:.3f}",
                            "Ajuste_propagado_Diciembre": "{:+.1%}",
                        }), use_container_width=True, hide_index=True,
                    )

                    tabla_comp = comparativo[[
                        "CATEGORIA", "Venta_Semana_1", "Prediccion_Modelo_Sin_Ajuste",
                        "Prediccion_Original", "Factor_Correccion_Aplicado", "Correccion_Aplicada",
                        "Cumplimiento_Forecast", "Proyeccion_Ritmo", "Nowcast_Ponderado",
                        "Brecha_Ritmo_pp", "Estado",
                    ]].sort_values("Brecha_Ritmo_pp")
                    st.dataframe(
                        tabla_comp.style.format({
                            "Venta_Semana_1": "${:,.2f}",
                            "Prediccion_Modelo_Sin_Ajuste": "${:,.2f}",
                            "Prediccion_Original": "${:,.2f}",
                            "Factor_Correccion_Aplicado": "{:.3f}",
                            "Cumplimiento_Forecast": "{:.1%}",
                            "Proyeccion_Ritmo": "${:,.2f}",
                            "Nowcast_Ponderado": "${:,.2f}",
                            "Brecha_Ritmo_pp": "{:+.1f} pp",
                        }),
                        use_container_width=True, hide_index=True,
                    )
                    st.download_button(
                        "Descargar comparativo semanal CSV",
                        tabla_comp.to_csv(index=False).encode("utf-8-sig"),
                        "comparativo_semana_1_vs_forecast_noviembre.csv",
                        "text/csv",
                    )

            # ------------------------------------------------------------------
            # Tabla resumen: Categoría, Código, Descripción, Nov 2026, Dic 2026
            # ------------------------------------------------------------------
            universo_tabla = universo if producto == "TODOS" else universo.loc[universo["CODIGO"].astype(str).eq(producto)]
            with st.expander("Auditoría de equivalencia: ventas, unidades y precio"):
                st.caption("Precio implícito anterior = venta final / unidades originales de CatBoost. "
                           "La referencia usa precio sugerido de la sábana por UPC; si falta, web verificada por UPC; y como último respaldo, el precio del último mes válido del mismo código. "
                           "Las referencias antiguas o con pocos meses requieren revisión.")
                cols_aud = ["CATEGORIA", "CODIGO", "DESCRIPCION", "Periodo", "Prediccion_final",
                            "Unidades_antes_validacion", "Precio_implicito_anterior", "Precio_referencia_SKU",
                            "Precio_sugerido_sabana", "Precio_web", "Precio_ultimo_mes", "Fuente_precio",
                            "Coincidencia_codigo_sabana", "Fuente_web", "URL_precio_web",
                            "Unidades_pred", "Diferencia_unidades", "Factor_inconsistencia_precio",
                            "Prioridad_correccion_precio", "Alerta_precio_anterior",
                            "Calidad_referencia_precio", "Meses_precio_validos", "Ultimo_periodo_precio",
                            "Meses_precio_atipicos"]
                aud = universo_tabla[cols_aud].copy()
                prioridad_orden = {"P1 - CRITICA": 1, "P2 - ALTA": 2, "P3 - MEDIA": 3, "P4 - SIN ALERTA": 4}
                aud["Orden_prioridad"] = aud["Prioridad_correccion_precio"].map(prioridad_orden).fillna(5)
                aud = aud.sort_values(["Orden_prioridad", "Factor_inconsistencia_precio"], ascending=[True, False])
                resumen_cat = aud.groupby(["CATEGORIA", "Periodo"], as_index=False).agg(
                    Venta_MXN=("Prediccion_final", "sum"),
                    Unidades_antes=("Unidades_antes_validacion", "sum"),
                    Unidades_corregidas=("Unidades_pred", "sum"),
                    Precio_implicito_antes=("Precio_implicito_anterior", "median"),
                    Precio_referencia_mediana=("Precio_referencia_SKU", "median"),
                    Factor_inconsistencia_mediano=("Factor_inconsistencia_precio", "median"),
                    Factor_inconsistencia_maximo=("Factor_inconsistencia_precio", "max"),
                    SKU_analizados=("CODIGO", "nunique"),
                    Casos_P1=("Prioridad_correccion_precio", lambda s: int((s == "P1 - CRITICA").sum())),
                    Casos_P2=("Prioridad_correccion_precio", lambda s: int((s == "P2 - ALTA").sum())),
                )
                resumen_cat["Precio_categoria_antes"] = resumen_cat["Venta_MXN"].div(
                    resumen_cat["Unidades_antes"].where(resumen_cat["Unidades_antes"].gt(0))
                )
                resumen_cat["Precio_categoria_corregido"] = resumen_cat["Venta_MXN"].div(
                    resumen_cat["Unidades_corregidas"].where(resumen_cat["Unidades_corregidas"].gt(0))
                )
                resumen_cat["Prioridad_categoria"] = np.select(
                    [resumen_cat["Casos_P1"].gt(0) | resumen_cat["Factor_inconsistencia_mediano"].ge(3.0),
                     resumen_cat["Casos_P2"].gt(0) | resumen_cat["Factor_inconsistencia_mediano"].ge(1.5)],
                    ["P1 - CRITICA", "P2 - ALTA"], default="P3 - MONITOREO"
                )
                resumen_cat["Orden"] = resumen_cat["Prioridad_categoria"].map(
                    {"P1 - CRITICA": 1, "P2 - ALTA": 2, "P3 - MONITOREO": 3}
                )
                resumen_cat = resumen_cat.sort_values(
                    ["Orden", "Casos_P1", "Factor_inconsistencia_mediano", "Venta_MXN"],
                    ascending=[True, False, False, False]
                ).drop(columns="Orden")
                st.markdown("#### Prioridad por categoría")
                st.dataframe(resumen_cat, hide_index=True, use_container_width=True)
                st.download_button(
                    "Descargar prioridad de corrección por categoría CSV",
                    resumen_cat.to_csv(index=False).encode("utf-8-sig"),
                    "prioridad_correccion_precio_por_categoria.csv", "text/csv"
                )
                alertas = aud["Alerta_precio_anterior"].ne("SIN ALERTA") | aud["Calidad_referencia_precio"].ne("HISTORICO RECIENTE")
                solo = st.checkbox("Mostrar solo casos para revisión", value=True, key="solo_alertas_equivalencia")
                st.dataframe(aud.loc[alertas] if solo else aud, hide_index=True, use_container_width=True)
                aud_descarga = aud.drop(columns="Orden_prioridad")
                st.download_button("Descargar auditoría de equivalencia CSV", aud_descarga.to_csv(index=False).encode("utf-8-sig"),
                                   "auditoria_equivalencia_forecast.csv", "text/csv")
            if universo_tabla["Unidades_pred"].isna().any():
                st.warning("Hay SKU sin precio histórico válido. El total de unidades del periodo queda sin estimar; el importe sí incluye todos los SKU seleccionados.")
            st.markdown("### Tabla resumen: predicción por SKU (Nov-Dic 2026)")
            st.caption(
                "Unidades = venta monetaria final / precio conciliado por UPC. Prioridad: sábana, web exacta y último mes válido del SKU. "
                "La venta semanal está agregada por categoría; los escenarios por SKU se distribuyen "
                "según la participación monetaria de cada SKU."
            )

            tabla_sku = (
                universo.groupby(
                    ["CATEGORIA", "CODIGO", "DESCRIPCION", "Periodo"],
                    as_index=False,
                )["Prediccion_final"].sum()
                .pivot(
                    index=["CATEGORIA", "CODIGO", "DESCRIPCION"],
                    columns="Periodo",
                    values="Prediccion_final",
                )
                .reset_index()
            )
            tabla_sku.columns.name = None
            tabla_sku = tabla_sku.rename(columns={
                202611: "Prediccion_Nov_2026",
                202612: "Prediccion_Dic_2026",
            })

            # Agrega unidades reconciliadas con precio observado del SKU.
            tabla_unidades_sku = (
                universo.groupby(
                    ["CATEGORIA", "CODIGO", "DESCRIPCION", "Periodo"],
                    as_index=False,
                )["Unidades_pred"].sum(min_count=1)
                .pivot(
                    index=["CATEGORIA", "CODIGO", "DESCRIPCION"],
                    columns="Periodo",
                    values="Unidades_pred",
                )
                .reset_index()
            )
            tabla_unidades_sku.columns.name = None
            tabla_unidades_sku = tabla_unidades_sku.rename(columns={
                202611: "Unidades_Pred_Nov_2026",
                202612: "Unidades_Pred_Dic_2026",
            })
            tabla_sku = tabla_sku.merge(
                tabla_unidades_sku,
                on=["CATEGORIA", "CODIGO", "DESCRIPCION"],
                how="left",
                validate="one_to_one",
            )
            for col in [
                "Prediccion_Nov_2026", "Prediccion_Dic_2026",
                "Unidades_Pred_Nov_2026", "Unidades_Pred_Dic_2026",
            ]:
                if col not in tabla_sku.columns:
                    tabla_sku[col] = np.nan
                tabla_sku[col] = pd.to_numeric(
                    tabla_sku[col], errors="coerce"
                )
                if col.startswith("Prediccion_"):
                    tabla_sku[col] = tabla_sku[col].fillna(0.0)

            tabla_sku["Precio_Implicito_Nov_2026"] = tabla_sku["Prediccion_Nov_2026"].div(
                tabla_sku["Unidades_Pred_Nov_2026"].where(tabla_sku["Unidades_Pred_Nov_2026"].gt(0))
            )
            tabla_sku["Precio_Implicito_Dic_2026"] = tabla_sku["Prediccion_Dic_2026"].div(
                tabla_sku["Unidades_Pred_Dic_2026"].where(tabla_sku["Unidades_Pred_Dic_2026"].gt(0))
            )
            tabla_sku["Total_NovDic"] = (
                tabla_sku["Prediccion_Nov_2026"] + tabla_sku["Prediccion_Dic_2026"]
            )
            tabla_sku["Total_Unidades_NovDic"] = (
                tabla_sku["Unidades_Pred_Nov_2026"]
                + tabla_sku["Unidades_Pred_Dic_2026"]
            )

            # --------------------------------------------------------------
            # Escenarios de noviembre por SKU
            # --------------------------------------------------------------
            # La tabla semanal disponible está agregada por categoría. Por ello,
            # la venta de semana 1 se distribuye entre SKU según la participación
            # de cada SKU en el forecast original de noviembre de su categoría.
            # Es una estimación por SKU, no una venta real observada por código.
            tabla_sku["Categoria_normalizada"] = tabla_sku["CATEGORIA"].map(
                normalizar_categoria_calibracion
            )
            tabla_sku["Participacion_SKU_Categoria"] = np.where(
                tabla_sku.groupby("Categoria_normalizada")["Prediccion_Nov_2026"]
                         .transform("sum") > 0,
                tabla_sku["Prediccion_Nov_2026"] /
                tabla_sku.groupby("Categoria_normalizada")["Prediccion_Nov_2026"]
                         .transform("sum"),
                0.0,
            )
            tabla_sku["Participacion_Unidades_SKU_Categoria"] = np.where(
                tabla_sku.groupby("Categoria_normalizada")["Unidades_Pred_Nov_2026"].transform("sum") > 0,
                tabla_sku["Unidades_Pred_Nov_2026"] / tabla_sku.groupby("Categoria_normalizada")["Unidades_Pred_Nov_2026"].transform("sum"),
                0.0,
            )
            tabla_sku["Monto_Venta_Semana_Prediccion_SKU_Estimado"] = np.nan
            tabla_sku["Unidades_Vendidas_Semana_Prediccion_SKU_Estimadas"] = np.nan
            tabla_sku["Proyeccion_Ritmo_Nov_2026"] = np.nan
            tabla_sku["Nowcast_Ponderado_Nov_2026"] = np.nan

            if archivo_semanal is not None:
                venta_semanal_sku = cargar_venta_semanal_noviembre(archivo_semanal)[
                    ["Categoria_normalizada", "Venta_Semana_1", "Unidades_Semana_1"]
                ].copy()
                tabla_sku = tabla_sku.merge(
                    venta_semanal_sku,
                    on="Categoria_normalizada",
                    how="left",
                )
                tabla_sku["Monto_Venta_Semana_Prediccion_SKU_Estimado"] = (
                    tabla_sku["Venta_Semana_1"] * tabla_sku["Participacion_SKU_Categoria"]
                )
                tabla_sku["Unidades_Vendidas_Semana_Prediccion_SKU_Estimadas"] = (
                    tabla_sku["Unidades_Semana_1"] * tabla_sku["Participacion_Unidades_SKU_Categoria"]
                )
                tabla_sku["Proyeccion_Ritmo_Nov_2026"] = (
                    tabla_sku["Monto_Venta_Semana_Prediccion_SKU_Estimado"] * 30.0 / 7.0
                )
                tabla_sku["Nowcast_Ponderado_Nov_2026"] = (
                    (1.0 - peso_real) * tabla_sku["Prediccion_Nov_2026"]
                    + peso_real * tabla_sku["Proyeccion_Ritmo_Nov_2026"]
                )
                tabla_sku = tabla_sku.drop(columns=["Venta_Semana_1", "Unidades_Semana_1"])
            else:
                st.info(
                    "Para mostrar ritmo y nowcast por SKU, carga PREDICCIONES NOVIEMBRE.xlsx. "
                    "Como la venta semanal está por categoría, el reparto por SKU se estima "
                    "con la participación del forecast original de noviembre."
                )

            tabla_sku = tabla_sku.drop(columns=["Categoria_normalizada"])
            tabla_sku = tabla_sku.sort_values(
                ["CATEGORIA", "CODIGO"]
            ).reset_index(drop=True)

            if producto != "TODOS":
                tabla_sku = tabla_sku.loc[tabla_sku["CODIGO"].astype(str).eq(producto)].copy()

            # Fila de totales al final
            fila_total = pd.DataFrame([{
                "CATEGORIA": "TOTAL",
                "CODIGO": "",
                "DESCRIPCION": "",
                "Prediccion_Nov_2026": tabla_sku["Prediccion_Nov_2026"].sum(),
                "Unidades_Pred_Nov_2026": tabla_sku["Unidades_Pred_Nov_2026"].sum(skipna=False),
                "Monto_Venta_Semana_Prediccion_SKU_Estimado": tabla_sku["Monto_Venta_Semana_Prediccion_SKU_Estimado"].sum(min_count=1),
                "Unidades_Vendidas_Semana_Prediccion_SKU_Estimadas": tabla_sku["Unidades_Vendidas_Semana_Prediccion_SKU_Estimadas"].sum(min_count=1),
                "Proyeccion_Ritmo_Nov_2026": tabla_sku["Proyeccion_Ritmo_Nov_2026"].sum(min_count=1),
                "Nowcast_Ponderado_Nov_2026": tabla_sku["Nowcast_Ponderado_Nov_2026"].sum(min_count=1),
                "Prediccion_Dic_2026": tabla_sku["Prediccion_Dic_2026"].sum(),
                "Unidades_Pred_Dic_2026": tabla_sku["Unidades_Pred_Dic_2026"].sum(skipna=False),
                "Total_NovDic": tabla_sku["Total_NovDic"].sum(),
                "Total_Unidades_NovDic": tabla_sku["Total_Unidades_NovDic"].sum(skipna=False),
                "Participacion_SKU_Categoria": np.nan,
                "Participacion_Unidades_SKU_Categoria": np.nan,
            }])
            for mes in ("Nov", "Dic"):
                unidades_total = fila_total[f"Unidades_Pred_{mes}_2026"]
                fila_total[f"Precio_Implicito_{mes}_2026"] = fila_total[f"Prediccion_{mes}_2026"].div(
                    unidades_total.where(unidades_total.gt(0))
                )
            tabla_sku_mostrar = pd.concat([tabla_sku, fila_total], ignore_index=True)

            st.dataframe(
                tabla_sku_mostrar.style.format({
                    "Prediccion_Nov_2026": "${:,.2f}",
                    "Precio_Implicito_Nov_2026": "${:,.2f}",
                    "Unidades_Pred_Nov_2026": "{:,.0f}",
                    "Monto_Venta_Semana_Prediccion_SKU_Estimado": "${:,.2f}",
                    "Unidades_Vendidas_Semana_Prediccion_SKU_Estimadas": "{:,.0f}",
                    "Proyeccion_Ritmo_Nov_2026": "${:,.2f}",
                    "Nowcast_Ponderado_Nov_2026": "${:,.2f}",
                    "Participacion_SKU_Categoria": "{:.2%}",
                    "Participacion_Unidades_SKU_Categoria": "{:.2%}",
                    "Prediccion_Dic_2026": "${:,.2f}",
                    "Precio_Implicito_Dic_2026": "${:,.2f}",
                    "Unidades_Pred_Dic_2026": "{:,.0f}",
                    "Total_NovDic": "${:,.2f}",
                    "Total_Unidades_NovDic": "{:,.0f}",
                }, na_rep="—"),
                use_container_width=True,
                hide_index=True,
                height=440,
            )
            st.download_button(
                "Descargar tabla SKU con semana de predicción, unidades, ritmo y nowcast CSV",
                tabla_sku.to_csv(index=False).encode("utf-8-sig"),
                "tabla_sku_forecast_ventas_unidades_ritmo_nowcast_nov_dic_2026.csv",
                "text/csv",
            )

            st.download_button(
                "Descargar forecast filtrado CSV",
                universo_tabla.to_csv(index=False).encode("utf-8-sig"),
                "forecast_nov_dic_2026_universo_seleccionado.csv",
                "text/csv",
            )
            with st.expander("Diagnóstico del forecast y universo SKU"):
                st.caption(
                    "Predicción base muestra la salida 85% venta directa + 15% unidades por precio. "
                    "Corrección piso indica cuánto añadió la protección estacional; cero significa que el modelo ya estaba por encima del piso."
                )
                diag = universo[[
                    "CODIGO", "DESCRIPCION", "CATEGORIA", "COMBOS", "Periodo",
                    "Prediccion_base", "Referencia_ultimo_ano", "Crecimiento_YTD",
                    "Prediccion_estacional", "Prediccion_combinada", "Piso_parcial",
                    "Porcentaje_recuperacion", "Correccion_piso",
                    "Ajuste_aplicado", "Prediccion_antes_calibracion",
                    "Factor_calibracion_categoria", "Calibracion_categoria_aplicada",
                    "Motivo_calibracion", "Prediccion_antes_control_lacteos",
                    "Unidades_antes_control_lacteos", "Factor_control_diciembre_lacteos",
                    "Control_diciembre_lacteos_aplicado", "Prediccion_antes_ajuste_chocolate_mesa", "Factor_ajuste_nov_chocolate_mesa", "Ajuste_nov_chocolate_mesa_aplicado", "Prediccion_final", "Unidades_modelo_sin_reconciliar", "Precio_modelo", "Precio_pred", "Unidades_pred",
                ]].copy()
                resumen_categoria_v2 = (
                    diag.groupby("CATEGORIA", as_index=False)
                    .agg(
                        SKU=("CODIGO", "nunique"),
                        Antes=("Prediccion_antes_calibracion", "sum"),
                        Despues=("Prediccion_final", "sum"),
                        Factor=("Factor_calibracion_categoria", "max"),
                    )
                )
                resumen_categoria_v2["Impacto"] = resumen_categoria_v2["Despues"] - resumen_categoria_v2["Antes"]
                st.markdown("#### Resumen por categoría")
                st.dataframe(
                    resumen_categoria_v2.style.format({
                        "Antes": "${:,.2f}", "Despues": "${:,.2f}",
                        "Factor": "{:.2f}", "Impacto": "${:+,.2f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
                no_contados = forecast[~forecast["Incluido_forecast_base"]].copy()
                if not no_contados.empty:
                    resumen_no_contados = (
                        no_contados.groupby(
                            ["CODIGO", "DESCRIPCION", "CATEGORIA", "COMBOS",
                             "Ultimo_periodo_disponible", "Meses_sin_actividad_a_octubre"],
                            as_index=False,
                        )
                        .agg(Forecast_NovDic=("Prediccion_final", "sum"))
                        .sort_values("Forecast_NovDic", ascending=False)
                    )
                    st.markdown("#### SKU presentes en 2026 no contados en el universo base")
                    st.dataframe(
                        resumen_no_contados.style.format({"Forecast_NovDic": "${:,.2f}"}),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "Descargar SKU 2026 no contados",
                        resumen_no_contados.to_csv(index=False).encode("utf-8-sig"),
                        "sku_2026_no_contados_forecast_base.csv",
                        "text/csv",
                    )
                st.markdown("#### Detalle por SKU")
                st.dataframe(diag, use_container_width=True, hide_index=True)


    except Exception as error:
        st.exception(error)
else:
    st.info("Carga los tres archivos Excel para comenzar.")


