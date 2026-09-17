import streamlit as st
import pandas as pd
import numpy as np
import io

# Configuración inicial de la página
st.set_page_config(page_title="Modelo de Reparto Inteligente", layout="wide", page_icon="📦")

def estandarizar_columnas(df):
    """Limpia y estandariza los nombres de las columnas."""
    df.columns = df.columns.str.strip()
    mapa_columnas = {
        'nombre': 'Sucursal', 'sucursal': 'Sucursal',
        'categoría': 'Categoria', 'categoria': 'Categoria',
        'unidades vendidas': 'Unidades Vendidas', 'unidadesvendidas': 'Unidades Vendidas',
        'stocksv': 'StockSV', 'stock sv': 'StockSV',
        'faltante 0-3': 'Faltantes 0-3', 'faltantes 0-3': 'Faltantes 0-3',
        'dias de inventario': 'Dias de inventario', 'días de inventario': 'Dias de inventario',
        'total existencias': 'Total Existencias', 'instock in': 'Instock in',
        'idproducto': 'IDProducto', 'id producto': 'IDProducto',
        'descripción': 'Descripción', 'descripcion': 'Descripción',
        'cedis': 'CEDIS', 'universo': 'Universo'
    }
    nuevas_columnas = {col: mapa_columnas.get(col.lower().strip(), col) for col in df.columns}
    return df.rename(columns=nuevas_columnas)

@st.cache_data
def generar_modelo_reparto_sucursal(df):
    """Procesa el DataFrame y genera las prioridades de resurtido."""
    df_proc = estandarizar_columnas(df.copy())

    cols_numericas = [
        'CEDIS', 'Total Existencias', 'Unidades Vendidas', 
        'Universo', 'Faltantes 0-3', 'StockSV', 'Instock in', 'Dias de inventario'
    ]

    for col in cols_numericas:
        if col in df_proc.columns:
            if df_proc[col].dtype == object:
                df_proc[col] = df_proc[col].astype(str).str.replace(',', '').str.replace('%', '')
            df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce').fillna(0)

    # Reglas de negocio
    condiciones = [
        (df_proc['Faltantes 0-3'] >= 1) & (df_proc['CEDIS'] > 0) & (df_proc['Dias de inventario'] <= 7),
        (df_proc['Dias de inventario'] <= 7) & (df_proc['CEDIS'] <= 0) & (df_proc['Unidades Vendidas'] > 0),
        (df_proc['StockSV'] >= 1) | ((df_proc['Total Existencias'] > 0) & (df_proc['Unidades Vendidas'] == 0)),
        (df_proc['Dias de inventario'] > 60) & (df_proc['Unidades Vendidas'] > 0),
        (df_proc['Dias de inventario'] > 7) & (df_proc['Dias de inventario'] <= 60)
    ]

    prioridades = [
        '1 - Resurtir Urgente (CEDIS -> Sucursal)',
        '2 - Alerta Quiebre (Sin Stock en CEDIS)',
        '3 - Acción Comercial Local (Stock sin Venta)',
        '4 - Sobrestock Local (Frenar Envíos)',
        '5 - Inventario Sano'
    ]

    acciones = [
        'Generar orden de reparto desde CEDIS prioritariamente a esta sucursal.',
        'Sin stock en CEDIS; evaluar traspaso o compra.',
        'Revisar exhibición/frenteo en piso de venta o promover.',
        'Pausar despachos a esta tienda.',
        'Mantener flujo normal de abastecimiento.'
    ]

    df_proc['Nivel_Prioridad'] = np.select(condiciones, prioridades, default='6 - Revisión Manual')
    df_proc['Accion_Recomendada'] = np.select(condiciones, acciones, default='Validar datos de origen.')

    cols_orden = [c for c in ['Nivel_Prioridad', 'Sucursal', 'Dias de inventario'] if c in df_proc.columns]
    df_proc = df_proc.sort_values(by=cols_orden, ascending=[True, True, True])

    return df_proc

# ==========================================
# INTERFAZ DE STREAMLIT
# ==========================================

st.title("📦 Modelo Predictivo de Reparto por Sucursal")
st.markdown("Sube tu tabla semanal de detalles (CSV o Excel) para clasificar las acciones prioritarias de inventario.")

# 1. Componente para subir el archivo
archivo_subido = st.file_uploader("Arrastra aquí tu archivo detallado (.csv, .xlsx)", type=['csv', 'xlsx', 'xls'])

if archivo_subido is not None:
    # Cargar el archivo según su extensión
    if archivo_subido.name.lower().endswith('.csv'):
        try:
            df_bruto = pd.read_csv(archivo_subido, encoding='utf-8')
        except:
            df_bruto = pd.read_csv(archivo_subido, encoding='latin1')
    else:
        df_bruto = pd.read_excel(archivo_subido)

    # Procesar los datos
    with st.spinner("Procesando reglas de negocio e inventario..."):
        df_resultado = generar_modelo_reparto_sucursal(df_bruto)

    st.success("¡Datos procesados con éxito!")

    # 2. Métricas rápidas (KPIs)
    st.markdown("### 📊 Resumen de Prioridades")
    prioridad_counts = df_resultado['Nivel_Prioridad'].value_counts()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Resurtir Urgente (P1)", prioridad_counts.get('1 - Resurtir Urgente (CEDIS -> Sucursal)', 0))
    col2.metric("Alerta Quiebre (P2)", prioridad_counts.get('2 - Alerta Quiebre (Sin Stock en CEDIS)', 0))
    col3.metric("Acción Comercial (P3)", prioridad_counts.get('3 - Acción Comercial Local (Stock sin Venta)', 0))
    col4.metric("Sobrestock (P4)", prioridad_counts.get('4 - Sobrestock Local (Frenar Envíos)', 0))

    # 3. Filtros Interactivos
    st.markdown("### 🔍 Explorador de Recomendaciones")
    col_filtro1, col_filtro2 = st.columns(2)
    
    with col_filtro1:
        sucursales_unicas = ['Todas'] + list(df_resultado['Sucursal'].dropna().unique())
        sucursal_sel = st.selectbox("Filtrar por Sucursal", sucursales_unicas)
        
    with col_filtro2:
        prioridades_unicas = list(df_resultado['Nivel_Prioridad'].unique())
        prioridades_sel = st.multiselect("Filtrar por Prioridad", prioridades_unicas, default=prioridades_unicas)

    # Aplicar filtros
    df_mostrar = df_resultado[df_resultado['Nivel_Prioridad'].isin(prioridades_sel)]
    if sucursal_sel != 'Todas':
        df_mostrar = df_mostrar[df_mostrar['Sucursal'] == sucursal_sel]

    # Mostrar la tabla en Streamlit
    st.dataframe(df_mostrar, use_container_width=True, hide_index=True)

    # 4. Botón para Exportar a Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_resultado.to_excel(writer, index=False, sheet_name='Recomendaciones')
    
    excel_data = output.getvalue()

    st.markdown("### 📥 Exportar Resultados")
    st.download_button(
        label="Descargar Reporte en Excel",
        data=excel_data,
        file_name="Reporte_Recomendaciones_Reparto.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )