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

El forecast combina modelos globales CatBoost con selección mensual automática por categoría y reconciliación jerárquica SKU-categoría. El objetivo es estimar las ventas de noviembre y diciembre de 2026 sin limitarse a copiar el cierre de 2025.

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

La configuración se validó entrenando hasta octubre de 2025 y pronosticando noviembre-diciembre de 2025. Sin combos, los resultados fueron:

| Métrica | Resultado |
|---|---:|
| WAPE | 18.42% |
| MAE | $39,321.84 |
| RMSE | $113,103.17 |
| R² | 0.9457 |
| Sesgo | -6.43% |

El sesgo negativo indica que el modelo sigue siendo conservador. Las métricas son una referencia histórica, no una garantía del error real de 2026.

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
        "REGISTRO_PRESENTE",
        "CAT_VALOR_LAG1", "CAT_UNIDADES_LAG1", "CAT_PRECIO_LAG1",
        "CAT_CREC_VALOR_YOY", "CAT_CREC_VOL_YOY", "CAT_CREC_PRECIO_YOY",
        "CAT_TENDENCIA_VALOR_3M", "CAT_TENDENCIA_VOL_3M",
        "CAT_ACELERACION_VOL", "CAT_PCT_SKU_CAYENDO_VOL",
        "CAT_CREC_VALOR_LFL_YOY", "CAT_CREC_VOL_LFL_YOY",
        "CAT_SCORE_CONTRACCION", "CAT_REGIMEN_NUM",
    ]
)
FEATURES = NUM_FEATURES + CAT_FEATURES

# Configuración seleccionada en el backtest Nov-Dic 2025.
PESO_VENTA_DIRECTA = 0.85
PESO_ESTRUCTURAL = 0.15
PESO_FORECAST_ESTACIONAL = 0.85
UMBRAL_ESTACIONAL = 1.20

# Calibración selectiva V2 validada sobre el backtest Nov-Dic 2025 sin combos.
CALIBRACION_CATEGORIAS_V2 = {}  # Desactivada: selección automática por backtesting mensual.
FACTOR_CALIBRACION_MIN = 0.90
FACTOR_CALIBRACION_MAX = 1.10
APLICAR_CALIBRACION_A_COMBOS = False

# Control comercial de diciembre para LACTEOS CULINARIOS.
# Diciembre se ancla al noviembre actualizado del escenario, no al forecast alto original.
APLICAR_TOPE_DICIEMBRE_LACTEOS = False
PISO_DIC_VS_NOV_LACTEOS = 0.95       # Diciembre no cae más de 5% vs noviembre actualizado.
CRECIMIENTO_BASE_DIC_LACTEOS = 0.05  # Escenario central: +5% vs noviembre actualizado.
TOPE_DIC_VS_NOV_LACTEOS = 1.10       # Máximo: +10% vs noviembre actualizado.
USAR_RELACION_HISTORICA_LACTEOS = True


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
    salida = completar_calendario(agrupado)
    return salida, pd.DataFrame(diagnostico)


def completar_calendario(df):
    salida = []
    for codigo, grupo in df.groupby("CODIGO", sort=False):
        x = grupo.sort_values("Periodo").copy()
        fechas = pd.to_datetime(
            dict(year=x["AÑO"], month=x["MES_NUM"], day=1)
        )
        x.index = fechas
        x["REGISTRO_PRESENTE"] = 1
        x = x.reindex(pd.date_range(fechas.min(), fechas.max(), freq="MS"))
        x["REGISTRO_PRESENTE"] = x["REGISTRO_PRESENTE"].fillna(0).astype(int)
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


def _serie_numerica(df, columna, valor=0.0):
    """Devuelve siempre una Serie numérica alineada al índice de df."""
    if columna not in df.columns:
        return pd.Series(valor, index=df.index, dtype="float64")
    return pd.to_numeric(df[columna], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(valor)


def _division_segura(numerador, denominador, valor_neutro=0.0):
    """División vectorizada que siempre devuelve Series y nunca escalares."""
    num = pd.Series(numerador).astype(float)
    den = pd.Series(denominador, index=num.index).astype(float)
    salida = num.div(den.where(den.abs() > 1e-9))
    return salida.replace([np.inf, -np.inf], np.nan).fillna(valor_neutro)


def crear_features_categoria_causales(df):
    """Señales de categoría disponibles antes del mes a pronosticar."""
    requeridas = {"CATEGORIA", "CODIGO", "Periodo", "VENTA", "UNIDADES"}
    faltantes = requeridas.difference(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas para régimen de categoría: {sorted(faltantes)}")
    z = df.sort_values(["CATEGORIA", "CODIGO", "Periodo"]).copy()
    z["VENTA"] = _serie_numerica(z, "VENTA", 0.0)
    z["UNIDADES"] = _serie_numerica(z, "UNIDADES", 0.0)
    base = (z.groupby(["CATEGORIA", "Periodo"], as_index=False)
              .agg(CAT_VALOR=("VENTA", "sum"), CAT_UNIDADES=("UNIDADES", "sum")))
    base = base.sort_values(["CATEGORIA", "Periodo"]).reset_index(drop=True)
    base["CAT_PRECIO"] = _division_segura(base["CAT_VALOR"], base["CAT_UNIDADES"], 0.0)
    g = base.groupby("CATEGORIA", group_keys=False)
    for col, nom in [("CAT_VALOR","VALOR"),("CAT_UNIDADES","UNIDADES"),("CAT_PRECIO","PRECIO")]:
        base[f"CAT_{nom}_LAG1"] = g[col].shift(1)
        base[f"CAT_{nom}_LAG12"] = g[col].shift(12)
    base["CAT_CREC_VALOR_YOY"] = _division_segura(base["CAT_VALOR_LAG1"],base["CAT_VALOR_LAG12"],1.0)-1
    base["CAT_CREC_VOL_YOY"] = _division_segura(base["CAT_UNIDADES_LAG1"],base["CAT_UNIDADES_LAG12"],1.0)-1
    base["CAT_CREC_PRECIO_YOY"] = _division_segura(base["CAT_PRECIO_LAG1"],base["CAT_PRECIO_LAG12"],1.0)-1
    for col, nom in [("CAT_VALOR","VALOR"),("CAT_UNIDADES","VOL")]:
        ult3=g[col].transform(lambda s:s.shift(1).rolling(3,min_periods=2).mean())
        ant3=g[col].transform(lambda s:s.shift(4).rolling(3,min_periods=2).mean())
        base[f"CAT_TENDENCIA_{nom}_3M"]=_division_segura(ult3,ant3,1.0)-1
    base["CAT_ACELERACION_VOL"]=(base["CAT_TENDENCIA_VOL_3M"]-g["CAT_TENDENCIA_VOL_3M"].shift(3)).fillna(0.0)

    sku=z[["CATEGORIA","CODIGO","Periodo","VENTA","UNIDADES"]].copy()
    gs=sku.groupby(["CATEGORIA","CODIGO"],group_keys=False)
    for col,pref in [("VENTA","V"),("UNIDADES","U")]:
        sku[f"{pref}_LAG1"]=gs[col].shift(1)
        sku[f"{pref}_LAG13"]=gs[col].shift(13)
    sku["COMPARABLE"]=sku["U_LAG1"].notna() & sku["U_LAG13"].gt(0)
    sku["CAYENDO"]=sku["COMPARABLE"] & sku["U_LAG1"].lt(sku["U_LAG13"])
    def suma_comp(s):
        return s.where(sku.loc[s.index,"COMPARABLE"]).sum()
    amp=(sku.groupby(["CATEGORIA","Periodo"],as_index=False)
          .agg(COMPARABLE=("COMPARABLE","sum"),CAYENDO=("CAYENDO","sum"),
               V_ACT=("V_LAG1",suma_comp),V_ANT=("V_LAG13",suma_comp),
               U_ACT=("U_LAG1",suma_comp),U_ANT=("U_LAG13",suma_comp)))
    amp["CAT_PCT_SKU_CAYENDO_VOL"]=_division_segura(amp["CAYENDO"],amp["COMPARABLE"],0.0)
    amp["CAT_CREC_VALOR_LFL_YOY"]=_division_segura(amp["V_ACT"],amp["V_ANT"],1.0)-1
    amp["CAT_CREC_VOL_LFL_YOY"]=_division_segura(amp["U_ACT"],amp["U_ANT"],1.0)-1
    base=base.merge(amp[["CATEGORIA","Periodo","CAT_PCT_SKU_CAYENDO_VOL","CAT_CREC_VALOR_LFL_YOY","CAT_CREC_VOL_LFL_YOY"]],on=["CATEGORIA","Periodo"],how="left",validate="one_to_one")
    for c in ["CAT_CREC_VOL_YOY","CAT_TENDENCIA_VOL_3M","CAT_CREC_VALOR_YOY","CAT_ACELERACION_VOL","CAT_CREC_VOL_LFL_YOY"]:
        base[c]=_serie_numerica(base,c,0.0).clip(-0.50,0.50)
    base["CAT_PCT_SKU_CAYENDO_VOL"]=_serie_numerica(base,"CAT_PCT_SKU_CAYENDO_VOL",0.0).clip(0,1)
    base["CAT_SCORE_CONTRACCION"]=(
        .25*base["CAT_CREC_VOL_LFL_YOY"]+.20*base["CAT_CREC_VOL_YOY"]+
        .20*base["CAT_TENDENCIA_VOL_3M"]+.15*base["CAT_CREC_VALOR_YOY"]+
        .10*base["CAT_ACELERACION_VOL"]-.10*(base["CAT_PCT_SKU_CAYENDO_VOL"]-.50)
    ).clip(-.50,.50)
    base["CAT_SENALES_CONTRACCION"]=(
        (base["CAT_CREC_VOL_LFL_YOY"]<-.05).astype(int)+
        (base["CAT_TENDENCIA_VOL_3M"]<-.05).astype(int)+
        (base["CAT_ACELERACION_VOL"]<-.05).astype(int)+
        (base["CAT_PCT_SKU_CAYENDO_VOL"]>.60).astype(int))
    base["CAT_REGIMEN"]=np.select([
        (base["CAT_SENALES_CONTRACCION"]>=3)&(base["CAT_SCORE_CONTRACCION"]<-.20),
        base["CAT_SENALES_CONTRACCION"]>=3,base["CAT_SCORE_CONTRACCION"]<-.05,
        base["CAT_SCORE_CONTRACCION"]>.05],
        ["CONTRACCION_SEVERA","CONTRACCION","DESACELERACION","EXPANSION"],default="ESTABLE")
    base["CAT_REGIMEN_NUM"]=base["CAT_REGIMEN"].map({"CONTRACCION_SEVERA":-2,"CONTRACCION":-1,"DESACELERACION":-.5,"ESTABLE":0,"EXPANSION":1}).astype(float)
    cols=["CATEGORIA","Periodo","CAT_REGIMEN","CAT_SENALES_CONTRACCION"]+[c for c in NUM_FEATURES if c.startswith("CAT_")]
    return base[list(dict.fromkeys(cols))]


def anexar_features_categoria(df):
    """Idempotente: puede llamarse varias veces sin columnas _x/_y."""
    z=df.copy()
    generadas=[c for c in NUM_FEATURES if c.startswith("CAT_")]+["CAT_REGIMEN","CAT_SENALES_CONTRACCION"]
    borrar=[c for c in z.columns if c in generadas or (c.startswith("CAT_") and (c.endswith("_x") or c.endswith("_y")))]
    z=z.drop(columns=borrar,errors="ignore")
    cat=crear_features_categoria_causales(z)
    z=z.merge(cat,on=["CATEGORIA","Periodo"],how="left",validate="many_to_one")
    for c in [x for x in NUM_FEATURES if x.startswith("CAT_")]:
        z[c]=_serie_numerica(z,c,0.0)
    if "CAT_REGIMEN" not in z.columns:
        z["CAT_REGIMEN"]="ESTABLE"
    else:
        z["CAT_REGIMEN"]=z["CAT_REGIMEN"].fillna("ESTABLE").astype(str)
    if "CAT_SENALES_CONTRACCION" not in z.columns:
        z["CAT_SENALES_CONTRACCION"]=0
    else:
        z["CAT_SENALES_CONTRACCION"]=_serie_numerica(z,"CAT_SENALES_CONTRACCION",0).astype(int)
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
    if "REGISTRO_PRESENTE" not in z.columns:
        z["REGISTRO_PRESENTE"] = 1.0
    else:
        z["REGISTRO_PRESENTE"] = pd.to_numeric(
            z["REGISTRO_PRESENTE"], errors="coerce"
        ).fillna(0.0)
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
    if "REGISTRO_PRESENTE" in f.columns:
        f = f[f["REGISTRO_PRESENTE"].eq(1)].copy()
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
def entrenar_modelos(df_modelo):
    return {
        objetivo: entrenar_objetivo(df_modelo, objetivo)
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
    fila["REGISTRO_PRESENTE"] = 0
    ultimo = hist.iloc[-1]
    for col in [c for c in NUM_FEATURES if c.startswith("CAT_")]:
        valor = pd.to_numeric(pd.Series([ultimo.get(col, 0.0)]), errors="coerce").fillna(0.0).iloc[0]
        fila[col] = float(valor)
    fila["tendencia_3_12"] = (
        fila["mean3"] / fila["mean12"]
        if pd.notna(fila["mean12"]) and fila["mean12"] > 0
        else 1.0
    )

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


def extender_historial_hasta_octubre_2026(historial):
    """Completa meses faltantes hasta octubre 2026 para SKU inactivos.

    Los meses posteriores al último registro se consideran sin venta y sin
    unidades. El precio se conserva como última referencia disponible. Esto
    evita tratar julio como si fuera el mes inmediatamente anterior a noviembre.
    """
    hist = historial.sort_values("Periodo").copy()
    ultimo_periodo = int(hist["Periodo"].max())
    if ultimo_periodo >= 202610:
        return hist

    ultima_fecha = pd.Timestamp(
        year=ultimo_periodo // 100,
        month=ultimo_periodo % 100,
        day=1,
    )
    fechas_faltantes = pd.date_range(
        ultima_fecha + pd.offsets.MonthBegin(1),
        pd.Timestamp("2026-10-01"),
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



def _pronosticos_categoria_historicos(serie):
    """Backtest causal de métodos mensuales; cada fila usa solo meses anteriores."""
    y = pd.Series(serie, dtype=float).reset_index(drop=True)
    out = pd.DataFrame({"Real": y})
    out["ESTACIONAL_12"] = y.shift(12)
    out["MEDIANA_3"] = y.shift(1).rolling(3, min_periods=3).median()
    out["MEDIA_POND_3"] = (
        0.50 * y.shift(1) + 0.30 * y.shift(2) + 0.20 * y.shift(3)
    )
    media3 = y.shift(1).rolling(3, min_periods=3).mean()
    previa3 = y.shift(4).rolling(3, min_periods=3).mean()
    tendencia = (media3 / previa3.replace(0, np.nan)).clip(0.80, 1.25)
    out["TENDENCIA_LIMITADA"] = media3 * tendencia
    return out


def seleccionar_metodo_categoria(df, cortes=12):
    """Selecciona método por categoría con Score = WAPE + 0.5*|sesgo|."""
    mensual = (df[df["REGISTRO_PRESENTE"].eq(1)]
               .groupby(["CATEGORIA", "Periodo"], as_index=False)["VENTA"].sum()
               .sort_values(["CATEGORIA", "Periodo"]))
    filas=[]
    for cat,g in mensual.groupby("CATEGORIA", sort=False):
        bt=_pronosticos_categoria_historicos(g["VENTA"])
        bt=bt.tail(int(cortes))
        for metodo in ["ESTACIONAL_12","MEDIANA_3","MEDIA_POND_3","TENDENCIA_LIMITADA"]:
            z=bt[["Real",metodo]].dropna()
            if len(z)<4 or z["Real"].sum()<=0: continue
            err=z[metodo]-z["Real"]
            wape=err.abs().sum()/z["Real"].abs().sum()
            sesgo=err.sum()/z["Real"].abs().sum()
            filas.append({"CATEGORIA":cat,"Metodo":metodo,"WAPE":wape,
                          "Sesgo":sesgo,"Score":wape+0.50*abs(sesgo),"Cortes":len(z)})
    metricas=pd.DataFrame(filas)
    if metricas.empty: return {}, metricas
    mejor=(metricas.sort_values(["CATEGORIA","Score"])
           .groupby("CATEGORIA",as_index=False).first())
    return dict(zip(mejor["CATEGORIA"],mejor["Metodo"])), metricas


def forecast_categoria_mensual(df, categoria, periodo, metodo):
    g=(df[(df["CATEGORIA"]==categoria)&df["REGISTRO_PRESENTE"].eq(1)]
       .groupby("Periodo",as_index=False)["VENTA"].sum().sort_values("Periodo"))
    valores=g["VENTA"].astype(float).tolist()
    if not valores: return np.nan
    if metodo=="ESTACIONAL_12" and len(valores)>=12: return valores[-12]
    if metodo=="MEDIANA_3" and len(valores)>=3: return float(np.median(valores[-3:]))
    if metodo=="MEDIA_POND_3" and len(valores)>=3:
        return 0.50*valores[-1]+0.30*valores[-2]+0.20*valores[-3]
    if metodo=="TENDENCIA_LIMITADA" and len(valores)>=6:
        m3=np.mean(valores[-3:]); p3=np.mean(valores[-6:-3])
        fac=np.clip(m3/p3,0.80,1.25) if p3>0 else 1.0
        return float(m3*fac)
    return float(np.median(valores[-3:])) if len(valores)>=3 else float(valores[-1])


def aplicar_control_regimen_categoria(forecast, df):
    """Modera excesos solo cuando varias señales confirman contracción."""
    out=forecast.copy()
    cat=crear_features_categoria_causales(df)
    cols=["CATEGORIA","CAT_REGIMEN","CAT_SCORE_CONTRACCION","CAT_SENALES_CONTRACCION","CAT_CREC_VOL_LFL_YOY","CAT_CREC_VALOR_LFL_YOY","CAT_PCT_SKU_CAYENDO_VOL","CAT_VALOR_LAG1"]
    ultimo=cat.sort_values("Periodo").groupby("CATEGORIA",as_index=False).tail(1)[cols]
    out=out.merge(ultimo,on="CATEGORIA",how="left",validate="many_to_one")
    out["Prediccion_antes_control_regimen"]=out["Prediccion_final"]
    out["Factor_control_regimen"]=1.0
    out["Control_regimen_aplicado"]=False
    for (_, _),idx in out.groupby(["CATEGORIA","Periodo"]).groups.items():
        reg=str(out.loc[idx,"CAT_REGIMEN"].fillna("ESTABLE").iloc[0])
        sen=int(_serie_numerica(out.loc[idx],"CAT_SENALES_CONTRACCION",0).iloc[0])
        if reg not in {"DESACELERACION","CONTRACCION","CONTRACCION_SEVERA"} or sen<2: continue
        total=float(out.loc[idx,"Prediccion_final"].sum())
        ref=float(_serie_numerica(out.loc[idx],"CAT_VALOR_LAG1",0).iloc[0])
        score=float(_serie_numerica(out.loc[idx],"CAT_SCORE_CONTRACCION",0).iloc[0])
        if total<=0 or ref<=0: continue
        margen={"DESACELERACION":.08,"CONTRACCION":.03,"CONTRACCION_SEVERA":0}[reg]
        limite=ref*max(.75,1+margen+float(np.clip(.35*score,-.12,.03)))
        if total>limite:
            factor=float(np.clip(limite/total,.75,1.0))
            out.loc[idx,"Prediccion_final"]*=factor
            out.loc[idx,"Factor_control_regimen"]=factor
            out.loc[idx,"Control_regimen_aplicado"]=True
    return out


def reconciliar_categoria_sku(forecast, df, peso_categoria=0.60):
    """Combina total SKU CatBoost con ganador mensual y reconcilia a SKU."""
    seleccion, metricas = seleccionar_metodo_categoria(df, cortes=12)
    out=forecast.copy()
    out["Prediccion_SKU_antes_reconciliacion"]=out["Prediccion_final"]
    out["Metodo_categoria"]="CATBOOST_SKU"
    out["Forecast_categoria_modelo"]=np.nan
    out["Factor_reconciliacion_categoria"]=1.0
    for (cat,periodo),idx in out.groupby(["CATEGORIA","Periodo"]).groups.items():
        metodo=seleccion.get(cat)
        if not metodo: continue
        objetivo=forecast_categoria_mensual(df,cat,int(periodo),metodo)
        total_sku=out.loc[idx,"Prediccion_final"].sum()
        if not np.isfinite(objetivo) or objetivo<0 or total_sku<=0: continue
        # 60% señal categoría, 40% CatBoost SKU; límite histórico prudente ±25%.
        total_final=(1-peso_categoria)*total_sku+peso_categoria*objetivo
        factor=float(np.clip(total_final/total_sku,0.75,1.25))
        out.loc[idx,"Prediccion_final"]*=factor
        out.loc[idx,"Metodo_categoria"]=metodo
        out.loc[idx,"Forecast_categoria_modelo"]=objetivo
        out.loc[idx,"Factor_reconciliacion_categoria"]=factor
    return out, metricas

@st.cache_data(show_spinner=False)
def _generar_forecast_base(df, version_modelo="mensual-jerarquico-v1"):
    """Genera forecast para todos los SKU presentes en ventas 2026.

    Los SKU activos desde agosto 2026 forman el universo base. Los SKU presentes
    en 2026, pero sin actividad desde agosto, se pronostican y se etiquetan para
    que el usuario decida mediante segmentador si afectan o no los totales.
    """
    modelos = entrenar_modelos(df)
    filas = []
    periodos = [202611, 202612]
    codigos_2026 = set(
        df.loc[df["AÑO"] == 2026, "CODIGO"].astype(str).unique().tolist()
    )

    for codigo, historial in df.groupby("CODIGO", sort=False):
        if str(codigo) not in codigos_2026:
            continue
        hist_original = historial.sort_values("Periodo").copy()
        if hist_original.empty:
            continue

        ultimo_periodo_disponible = int(hist_original["Periodo"].max())
        incluido_base = ultimo_periodo_disponible >= 202608
        estado_forecast = "INCLUIDO_BASE" if incluido_base else "NO_CONTADO_BASE"
        meses_sin_actividad = meses_entre_periodos(ultimo_periodo_disponible, 202610)

        # Para SKU no contados en el forecast base, se completan con ceros hasta
        # octubre. Se mantienen separados y no afectan el total salvo elección.
        hist = (
            hist_original
            if incluido_base
            else extender_historial_hasta_octubre_2026(hist_original)
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

        for periodo, vd, unid, precio_prom in zip(
            periodos, venta_directa, unidades, precio
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
                "Unidades_pred": float(unid),
                "Precio_pred": float(precio_prom),
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
    return pd.DataFrame(filas)

@st.cache_data(show_spinner=False)
def generar_forecast(df, version_modelo="mensual-jerarquico-v2-regimen"):
    df_modelo = anexar_features_categoria(df)
    base = _generar_forecast_base(df_modelo, version_modelo)
    reconciliado, metricas = reconciliar_categoria_sku(base, df_modelo, peso_categoria=0.60)
    final = aplicar_control_regimen_categoria(reconciliado, df_modelo)
    final.attrs["metricas_backtest_categoria"] = metricas
    final.attrs["diagnostico_regimen_categoria"] = crear_features_categoria_causales(df_modelo)
    return final

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
        faltantes = sorted({2024, 2025, 2026} - anos_cargados)
        if faltantes:
            st.error(
                f"No se cargaron datos válidos de {faltantes}. Revisa en el diagnóstico "
                "qué columna falta en las hojas correspondientes."
            )
            st.stop()

        col_boton, col_estado = st.columns([1, 2])
        with col_boton:
            if st.button("Limpiar forecast anterior"):
                st.session_state.pop("forecast_2026_mensual_jerarquico_v2_regimen", None)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
        with col_estado:
            st.info("Modelo mensual jerárquico activo: backtesting por categoría, reconciliación SKU y sin calibraciones manuales.")

        if "forecast_2026_mensual_jerarquico_v2_regimen" not in st.session_state:
            st.info("Genera el forecast para habilitar la gráfica de noviembre-diciembre de 2026.")

        if st.button("Generar forecast mensual jerárquico Nov-Dic 2026", type="primary"):
            with st.spinner("Entrenando el modelo y generando el forecast..."):
                st.session_state["forecast_2026_mensual_jerarquico_v2_regimen"] = generar_forecast(df)
            st.rerun()

        if "forecast_2026_mensual_jerarquico_v2_regimen" in st.session_state:
            forecast = st.session_state["forecast_2026_mensual_jerarquico_v2_regimen"]
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
                    help="Permite comparar una o varias categorías sin alterar el forecast original.",
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
                        "Prediccion_Original": "Forecast original",
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
                                domain=["Forecast original", "Proyección por ritmo", "Nowcast ponderado"],
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
                            "desde noviembre actualizado y queda limitado entre -5% y +10%."
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
                        (f"Azul: ventas reales | Rojo: ritmo; diciembre anclado a noviembre con tope +10%" if seleccion_exclusiva_lacteos else f"Azul: ventas reales | Rojo: ritmo; diciembre propaga {propagacion_dic:.0%} del ajuste, limitado a ±{limite_ajuste_dic:.0%}"),
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
                        (f"Azul: ventas reales | Verde: nowcast con {peso_real:.0%}; diciembre anclado a noviembre con tope +10%" if seleccion_exclusiva_lacteos else f"Azul: ventas reales | Verde: nowcast con {peso_real:.0%} de peso semanal; diciembre ajustado con propagación parcial"),
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
                        "CATEGORIA", "Venta_Semana_1", "Prediccion_Original",
                        "Cumplimiento_Forecast", "Proyeccion_Ritmo", "Nowcast_Ponderado",
                        "Brecha_Ritmo_pp", "Estado",
                    ]].sort_values("Brecha_Ritmo_pp")
                    st.dataframe(
                        tabla_comp.style.format({
                            "Venta_Semana_1": "${:,.2f}",
                            "Prediccion_Original": "${:,.2f}",
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
            st.markdown("### Tabla resumen: predicción por SKU (Nov-Dic 2026)")
            st.caption(
                "Incluye forecast original, proyección por ritmo y nowcast ponderado de noviembre. "
                "Como la venta semanal está agregada por categoría, los escenarios por SKU se estiman "
                "distribuyendo la venta semanal según la participación de cada SKU en el forecast original."
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
            for col in ["Prediccion_Nov_2026", "Prediccion_Dic_2026"]:
                if col not in tabla_sku.columns:
                    tabla_sku[col] = 0.0
                tabla_sku[col] = tabla_sku[col].fillna(0.0)

            tabla_sku["Total_NovDic"] = (
                tabla_sku["Prediccion_Nov_2026"] + tabla_sku["Prediccion_Dic_2026"]
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
            tabla_sku["Venta_Semana_1_SKU_Estimada"] = np.nan
            tabla_sku["Proyeccion_Ritmo_Nov_2026"] = np.nan
            tabla_sku["Nowcast_Ponderado_Nov_2026"] = np.nan

            if archivo_semanal is not None:
                venta_semanal_sku = cargar_venta_semanal_noviembre(archivo_semanal)[
                    ["Categoria_normalizada", "Venta_Semana_1"]
                ].copy()
                tabla_sku = tabla_sku.merge(
                    venta_semanal_sku,
                    on="Categoria_normalizada",
                    how="left",
                )
                tabla_sku["Venta_Semana_1_SKU_Estimada"] = (
                    tabla_sku["Venta_Semana_1"]
                    * tabla_sku["Participacion_SKU_Categoria"]
                )
                tabla_sku["Proyeccion_Ritmo_Nov_2026"] = (
                    tabla_sku["Venta_Semana_1_SKU_Estimada"] * 30.0 / 7.0
                )
                tabla_sku["Nowcast_Ponderado_Nov_2026"] = (
                    (1.0 - peso_real) * tabla_sku["Prediccion_Nov_2026"]
                    + peso_real * tabla_sku["Proyeccion_Ritmo_Nov_2026"]
                )
                tabla_sku = tabla_sku.drop(columns=["Venta_Semana_1"])
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

            # Fila de totales al final
            fila_total = pd.DataFrame([{
                "CATEGORIA": "TOTAL",
                "CODIGO": "",
                "DESCRIPCION": "",
                "Prediccion_Nov_2026": tabla_sku["Prediccion_Nov_2026"].sum(),
                "Venta_Semana_1_SKU_Estimada": tabla_sku["Venta_Semana_1_SKU_Estimada"].sum(min_count=1),
                "Proyeccion_Ritmo_Nov_2026": tabla_sku["Proyeccion_Ritmo_Nov_2026"].sum(min_count=1),
                "Nowcast_Ponderado_Nov_2026": tabla_sku["Nowcast_Ponderado_Nov_2026"].sum(min_count=1),
                "Prediccion_Dic_2026": tabla_sku["Prediccion_Dic_2026"].sum(),
                "Total_NovDic": tabla_sku["Total_NovDic"].sum(),
                "Participacion_SKU_Categoria": np.nan,
            }])
            tabla_sku_mostrar = pd.concat([tabla_sku, fila_total], ignore_index=True)

            st.dataframe(
                tabla_sku_mostrar.style.format({
                    "Prediccion_Nov_2026": "${:,.2f}",
                    "Venta_Semana_1_SKU_Estimada": "${:,.2f}",
                    "Proyeccion_Ritmo_Nov_2026": "${:,.2f}",
                    "Nowcast_Ponderado_Nov_2026": "${:,.2f}",
                    "Participacion_SKU_Categoria": "{:.2%}",
                    "Prediccion_Dic_2026": "${:,.2f}",
                    "Total_NovDic": "${:,.2f}",
                }),
                use_container_width=True,
                hide_index=True,
                height=440,
            )
            st.download_button(
                "Descargar tabla SKU con ritmo y nowcast CSV",
                tabla_sku.to_csv(index=False).encode("utf-8-sig"),
                "tabla_sku_forecast_ritmo_nowcast_nov_dic_2026.csv",
                "text/csv",
            )

            st.download_button(
                "Descargar forecast filtrado CSV",
                universo.to_csv(index=False).encode("utf-8-sig"),
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
                    "Motivo_calibracion", "Prediccion_final",
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
                st.markdown("#### Backtesting mensual por categoría")
                metricas_bt = forecast.attrs.get("metricas_backtest_categoria", pd.DataFrame())
                if not metricas_bt.empty:
                    st.dataframe(metricas_bt.style.format({"WAPE":"{:.1%}","Sesgo":"{:+.1%}","Score":"{:.3f}"}),
                                 use_container_width=True, hide_index=True)
                st.markdown("#### Detalle por SKU")
                columnas_extra=[c for c in ["Prediccion_SKU_antes_reconciliacion","Metodo_categoria",
                    "Forecast_categoria_modelo","Factor_reconciliacion_categoria"] if c in universo.columns]
                st.dataframe(universo[["CODIGO","CATEGORIA","Periodo","Prediccion_final"]+columnas_extra],
                             use_container_width=True, hide_index=True)


    except Exception as error:
        st.exception(error)
else:
    st.info("Carga los tres archivos Excel para comenzar.")



