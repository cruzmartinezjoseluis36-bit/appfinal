import streamlit as st
import pandas as pd
import re
import math

st.set_page_config(page_title="Detección de Registros Basura", layout="wide")
st.title("📊 Sistema de Detección de Registros Basura (UPIICSA)")
st.write("Versión Optimizada de Alto Rendimiento (Sin librerías pesadas)")

# --- 1. LÓGICA MATEMÁTICA DEL ÁRBOL DE DECISIÓN (HECHO A MANO) ---
def calcular_entropia(y):
    total = len(y)
    if total == 0: return 0
    conteo_1 = sum(y)
    conteo_0 = total - conteo_1
    p0 = conteo_0 / total
    p1 = conteo_1 / total
    
    entropia = 0
    if p0 > 0: entropia -= p0 * math.log2(p0)
    if p1 > 0: entropia -= p1 * math.log2(p1)
    return entropia

def entrenar_arbol_reglas(df, caracteristicas):
    # Evaluamos cuál característica reduce más la entropía (Ganancia de Información)
    mejor_caract = None
    mejor_ganancia = -1
    entropia_global = calcular_entropia(df['target_basura'].tolist())
    
    for col in caracteristicas:
        # Dividir los datos en base a la bandera (0 o 1)
        hijo_0 = df[df[col] == 0]['target_basura'].tolist()
        hijo_1 = df[df[col] == 1]['target_basura'].tolist()
        
        peso_0 = len(hijo_0) / len(df)
        peso_1 = len(hijo_1) / len(df)
        
        entropia_condicional = (peso_0 * calcular_entropia(hijo_0)) + (peso_1 * calcular_entropia(hijo_1))
        ganancia = entropia_global - entropia_condicional
        
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_caract = col
            
    return mejor_caract, mejor_ganancia

# --- 2. INTERFAZ DE USUARIO ---
uploaded_file = st.file_uploader("Selecciona un archivo CSV o Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success("¡Archivo cargado con éxito!")
        
        # FEATURE ENGINEERING
        df['f_curp_inv'] = df['curp'].apply(lambda x: 1 if pd.isna(x) or str(x).strip().upper() == 'ND' or len(str(x).strip()) != 18 else 0)
        
        col_clues = 'clues2' if 'clues2' in df.columns else 'clues'
        def validar_clues(x):
            patron = r'^[A-Z]{2}IMS[0-9]{6}$'
            if pd.isna(x): return 1
            return 0 if re.match(patron, str(x).strip()) else 1
        df['f_clues_inv'] = df[col_clues].apply(validar_clues)
        
        def validar_fechas(row):
            try:
                if 'fecsis' not in row or pd.isna(row['fecsis']) or pd.isna(row['fecha_derivacion']): return 0
                fsis = pd.to_datetime(row['fecsis'], dayfirst=True, errors='coerce')
                fder = pd.to_datetime(row['fecha_derivacion'], dayfirst=True, errors='coerce')
                if pd.isnat(fsis) or pd.isnat(fder): return 1
                return 1 if fder < fsis else 0
            except: return 1
        df['f_fecha_incoherente'] = df.apply(validar_fechas, axis=1)
        
        df['f_caso_invalido'] = df['caso'].apply(lambda x: 1 if pd.isna(x) or str(x).strip().upper() == 'ND' else 0)
        
        caracteristicas = ['f_curp_inv', 'f_clues_inv', 'f_fecha_incoherente', 'f_caso_invalido']
        df['target_basura'] = df[caracteristicas].max(axis=1)
        
        # Entrenar árbol matemático para encontrar la raíz principal
        raiz, ganancia = entrenar_arbol_reglas(df, caracteristicas)
        
        # MÁSTRICAS EN PANTALLA
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Total de Registros Analizados", value=len(df))
        col2.metric(label="Registros Basura Detectados", value=int(df['target_basura'].sum()))
        col3.metric(label="Nodo Raíz del Árbol (Mayor Entropía)", value=raiz)
        
        # DIAGRAMA DEL ÁRBOL EN TEXTO ENRIQUECIDO
        st.subheader("🌳 Estructura Jerárquica del Árbol de Decisión (Matemático)")
        st.markdown(f"""
        ```text
        [Nodo Raíz: {raiz}] (Ganancia de Información: {ganancia:.4f})
           ├── SI == 1 ──> [Clasificación: REGISTRO BASURA] (Cumple criterio de anomalía)
           └── NO == 0 ──> [Evaluar resto de banderas]
                ├── f_curp_inv == 1 --------> [BASURA]
                ├── f_clues_inv == 1 -------> [BASURA]
                ├── f_fecha_incoherente == 1 -> [BASURA]
                └── Todas en 0 -------------> [REGISTRO REAL / LIMPIO]
        ```
        """)
        
        # DESCARGA
        st.subheader("📥 Descargar Diagnóstico")
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Descargar CSV Analizado", data=csv_data, file_name="resultados_asf_covid.csv", mime="text/csv")
        
    except Exception as e:
        st.error(f"Error al procesar: {e}")
