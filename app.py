import streamlit as st
import pandas as pd
import numpy as np
import io
import requests

# Configuración principal de la aplicación web
st.set_page_config(
    page_title="Modelo Predictivo de Reparto",
    layout="wide",
    page_icon="📦"
)

def estandarizar_columnas(df):
    """Limpia y estandariza los nombres de las columnas del archivo subido."""
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
    """Procesa el DataFrame y evalúa prioridades con base en el inventario físico local."""
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

    # REGLAS DE NEGOCIO (Priorizando el Total Existencias físico local)
    condiciones = [
        # PRIORIDAD 1: Sucursal en quiebre (<=0) o crítica (Faltante 0-3) Y SÍ HAY stock en CEDIS.
        ((df_proc['Total Existencias'] <= 0) | (df_proc['Faltantes 0-3'] >= 1)) & (df_proc['CEDIS'] > 0),
        
        # PRIORIDAD 2: Sucursal en quiebre (<=0) o crítica, pero CEDIS TAMBIÉN está en ceros.
        ((df_proc['Total Existencias'] <= 0) | (df_proc['Faltantes 0-3'] >= 1)) & (df_proc['CEDIS'] <= 0),
        
        # PRIORIDAD 3: Hay inventario físico en sucursal, pero NO HAY VENTAS (Estancado)
        (df_proc['Total Existencias'] > 0) & (df_proc['Unidades Vendidas'] <= 0),
        
        # PRIORIDAD 4: Inventario local sano (>3), pero la CADENA está sobrestockeada (>60 días). Frenar envíos.
        (df_proc['Total Existencias'] > 3) & (df_proc['Dias de inventario'] > 60),
        
        # PRIORIDAD 5: Inventario local sano (>3) y la cobertura global es aceptable (<=60 días).
        (df_proc['Total Existencias'] > 3) & (df_proc['Dias de inventario'] <= 60)
    ]

    prioridades = [
        '1 - Resurtir Urgente (CEDIS -> Sucursal)',
        '2 - Alerta Quiebre (Sin Stock en CEDIS)',
        '3 - Acción Comercial Local (Stock sin Venta)',
        '4 - Sobrestock (Frenar Envíos)',
        '5 - Inventario Sano'
    ]

    acciones = [
        'Sucursal en quiebre. Generar orden de reparto desde CEDIS.',
        'Sucursal en quiebre y sin stock en CEDIS; evaluar traspaso o compra.',
        'Revisar exhibición en piso de venta; mercancía estancada.',
        'Pausar despachos a esta tienda.',
        'Mantener flujo normal de abastecimiento.'
    ]

    df_proc['Nivel_Prioridad'] = np.select(condiciones, prioridades, default='6 - Revisión Manual')
    df_proc['Accion_Recomendada'] = np.select(condiciones, acciones, default='Validar datos de origen.')

    cols_orden = [c for c in ['Nivel_Prioridad', 'Sucursal', 'CEDIS'] if c in df_proc.columns]
    df_proc = df_proc.sort_values(by=cols_orden, ascending=[True, True, False])

    return df_proc

def enviar_mensaje_evolution_api(server_url, api_key, instance_name, numero_destino, mensaje):
    """Envía un mensaje de WhatsApp a través de una instancia auto-hospedada de Evolution API."""
    server_url = server_url.rstrip('/')
    endpoint = f"{server_url}/message/sendText/{instance_name}"
    
    headers = {
        "apikey": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "number": numero_destino,
        "text": mensaje,
        "options": {
            "delay": 1200,
            "presence": "composing"
        }
    }
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=15)
        if response.status_code in [200, 201]:
            return True, "Mensaje enviado exitosamente."
        else:
            return False, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return False, f"Error de conexión: {str(e)}"

# ==========================================
# INTERFAZ DE STREAMLIT
# ==========================================

st.title("📦 Modelo Predictivo de Reparto por Sucursal")
st.markdown("Sube tu tabla semanal de detalles para clasificar las acciones prioritarias de inventario.")

archivo_subido = st.file_uploader("Arrastra aquí tu archivo detallado (.csv, .xlsx)", type=['csv', 'xlsx', 'xls'])

if archivo_subido is not None:
    # Carga de datos dinámica
    if archivo_subido.name.lower().endswith('.csv'):
        try:
            df_bruto = pd.read_csv(archivo_subido, encoding='utf-8')
        except:
            df_bruto = pd.read_csv(archivo_subido, encoding='latin1')
    else:
        df_bruto = pd.read_excel(archivo_subido)

    with st.spinner("Procesando reglas de negocio e inventario..."):
        df_resultado = generar_modelo_reparto_sucursal(df_bruto)

    st.success("¡Datos procesados con éxito!")

    # 1. Resumen de Métricas
    st.markdown("### 📊 Resumen de Prioridades")
    prioridad_counts = df_resultado['Nivel_Prioridad'].value_counts()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Resurtir Urgente (P1)", prioridad_counts.get('1 - Resurtir Urgente (CEDIS -> Sucursal)', 0))
    col2.metric("Alerta Quiebre (P2)", prioridad_counts.get('2 - Alerta Quiebre (Sin Stock en CEDIS)', 0))
    col3.metric("Acción Comercial (P3)", prioridad_counts.get('3 - Acción Comercial Local (Stock sin Venta)', 0))
    col4.metric("Sobrestock (P4)", prioridad_counts.get('4 - Sobrestock (Frenar Envíos)', 0))

    # 2. Explorador con Filtros
    st.markdown("### 🔍 Explorador de Recomendaciones")
    col_filtro1, col_filtro2 = st.columns(2)
    
    with col_filtro1:
        sucursales_unicas = ['Todas'] + list(df_resultado['Sucursal'].dropna().unique())
        sucursal_sel = st.selectbox("Filtrar por Sucursal", sucursales_unicas)
        
    with col_filtro2:
        prioridades_unicas = list(df_resultado['Nivel_Prioridad'].unique())
        prioridades_sel = st.multiselect("Filtrar por Prioridad", prioridades_unicas, default=prioridades_unicas)

    df_mostrar = df_resultado[df_resultado['Nivel_Prioridad'].isin(prioridades_sel)]
    if sucursal_sel != 'Todas':
        df_mostrar = df_mostrar[df_mostrar['Sucursal'] == sucursal_sel]

    st.dataframe(df_mostrar, use_container_width=True, hide_index=True)

    # 3. Exportar a Excel
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

    # 4. Módulo de Notificaciones por WhatsApp (Evolution API)
    st.markdown("---")
    st.markdown("### 📱 Enviar Alerta Automatizada por WhatsApp (Servidor Gratuito)")

    with st.expander("⚙️ Configurar envío vía Evolution API"):
        col_ev1, col_ev2 = st.columns(2)
        
        with col_ev1:
            url_servidor = st.text_input("URL del Servidor", value="https://tu-instancia.koyeb.app")
            instance_name = st.text_input("Nombre de la Instancia", value="reparto_florido")
            
        with col_ev2:
            api_key_server = st.text_input("API Key del Servidor", type="password")
            telefono_destino = st.text_input("Número de WhatsApp Destino", value="5216641234567")

        if st.button("🚀 Enviar Resumen de Alertas Críticas (P1 y P2)", type="primary"):
            if not url_servidor or not api_key_server or not instance_name or len(telefono_destino) < 10:
                st.warning("Completa los datos del servidor y el número telefónico de destino.")
            else:
                df_criticos = df_resultado[df_resultado['Nivel_Prioridad'].str.startswith(('1', '2'))]
                
                if df_criticos.empty:
                    st.info("No hay alertas críticas (P1 o P2) para notificar.")
                else:
                    mensaje_texto = f"🚨 REPORTE DE ALERTAS - REPARTO SECTORIAL\n"
                    mensaje_texto += f"Se detectaron {len(df_criticos)} SKU(s) en estado crítico:\n\n"
                    
                    for idx, row in df_criticos.head(8).iterrows():
                        mensaje_texto += f"• {row['Sucursal']} | {row['Descripción']}\n"
                        mensaje_texto += f"  - ID: {row['IDProducto']}\n"
                        mensaje_texto += f"  - Prioridad: {row['Nivel_Prioridad']}\n\n"
                    
                    if len(df_criticos) > 8:
                        mensaje_texto += f"⚠️ ...y {len(df_criticos) - 8} productos más en el reporte completo.\n\n"
                        
                    mensaje_texto += "📌 Acción: Revisar la plataforma web para procesar órdenes."

                    with st.spinner("Enviando mensaje desde tu servidor de WhatsApp..."):
                        exito, respuesta = enviar_mensaje_evolution_api(
                            url_servidor, 
                            api_key_server, 
                            instance_name, 
                            telefono_destino, 
                            mensaje_texto
                        )
                        
                    if exito:
                        st.success("¡Alerta enviada correctamente por WhatsApp!")
                    else:
                        st.error(f"Fallo en el envío: {respuesta}")
