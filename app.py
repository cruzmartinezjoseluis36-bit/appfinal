import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Configuración de la página web
st.set_page_config(page_title="Detección de Registros Basura", layout="wide")

st.title("📊 Sistema de Detección de Registros Basura (UPIICSA)")
st.write("Carga tu archivo de datos (CSV o Excel) basado en el formato de registro de pacientes COVID-19.")

# 1. Selector de archivos en la interfaz web
uploaded_file = st.file_uploader("Selecciona un archivo CSV o Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Leer el archivo según su extensión
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success("¡Archivo cargado con éxito!")
        
        # Mostrar vista previa de los datos cargados
        st.subheader("👀 Vista previa de los datos originales")
        st.dataframe(df.head(5))
        
        # 2. Ingeniería de Características (Feature Engineering)
        st.info("Generando indicadores de anomalías...")
        
        # Regla 1: CURP inválida (No tiene 18 caracteres, es nula o dice 'ND')
        df['f_curp_inv'] = df['curp'].apply(lambda x: 1 if pd.isna(x) or str(x).strip().upper() == 'ND' or len(str(x).strip()) != 18 else 0)
        
        # Regla 2: CLUES no sigue el patrón (ej. DFIMS000230)
        def validar_clues(x):
            patron = r'^[A-Z]{2}IMS[0-9]{6}$'
            if pd.isna(x): return 1
            return 0 if re.match(patron, str(x).strip()) else 1

        # En la imagen la columna se llama 'clues2'
        col_clues = 'clues2' if 'clues2' in df.columns else 'clues'
        df['f_clues_inv'] = df[col_clues].apply(validar_clues)
        
        # Regla 3: Fechas incoherentes (fecsis vs fecha_derivacion)
        def validar_fechas(row):
            try:
                # Si 'fecsis' no existe en tu archivo de prueba, puedes simularla o ignorar esta regla
                if 'fecsis' not in row or pd.isna(row['fecsis']): 
                    return 0 
                
                fsis = pd.to_datetime(row['fecsis'], dayfirst=True, errors='coerce')
                fder = pd.to_datetime(row['fecha_derivacion'], dayfirst=True, errors='coerce')
                
                if pd.isnat(fsis) or pd.isnat(fder): return 1
                return 1 if fder < fsis else 0
            except:
                return 1

        df['f_fecha_incoherente'] = df.apply(validar_fechas, axis=1)
        
        # Regla 4: Caso No Determinado o Faltante
        # (Adaptado: como en la imagen no vemos pdf5, evaluamos si el 'caso' es 'ND' o nulo)
        df['f_caso_invalido'] = df['caso'].apply(lambda x: 1 if pd.isna(x) or str(x).strip().upper() == 'ND' else 0)
        
        # 3. Definir el TARGET (1 si tiene cualquier anomalía, 0 si es limpio)
        caracteristicas = ['f_curp_inv', 'f_clues_inv', 'f_fecha_incoherente', 'f_caso_invalido']
        df['target_basura'] = df[caracteristicas].max(axis=1)
        
        # Mostrar el dataset procesado con las banderas creadas
        st.subheader("🧠 Datos con Indicadores de Anomalía Calculados")
        st.dataframe(df[['id', 'curp', col_clues, 'caso'] + caracteristicas + ['target_basura']].head(10))
        
        # 4. Entrenamiento del Árbol de Decisión
        X = df[caracteristicas]
        y = df['target_basura']
        
        # Validar que tengamos ambas clases para entrenar
        if len(y.unique()) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
            clf.fit(X_train, y_train)
            
            # Mostrar métricas en la interfaz web
            accuracy = clf.score(X_test, y_test) * 100
            
            col1, col2 = st.columns(2)
            col1.metric(label="Precisión del Modelo (Accuracy)", value=f"{accuracy:.2f}%")
            col2.metric(label="Total de Registros Analizados", value=len(df))
            
            # 5. Graficar el Árbol en la Web
            st.subheader("🌳 Estructura del Árbol de Decisión")
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_tree(clf, 
                      feature_names=['CURP Inv.', 'CLUES Inv.', 'Fecha Inc.', 'Caso Inv.'], 
                      class_names=['Real/Limpio', 'Basura'], 
                      filled=True, 
                      rounded=True,
                      ax=ax)
            st.pyplot(fig)
            
            # 6. Botón para descargar resultados
            st.subheader("📥 Descargar Resultados")
            df_anomalias = df[df['target_basura'] == 1]
            st.write(f"Se encontraron **{len(df_anomalias)}** registros sospechosos o basura.")
            
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar CSV con Diagnóstico",
                data=csv_data,
                file_name="resultados_diagnostico_covid.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ No se puede entrenar el árbol porque todos los registros pertenecen a la misma categoría (todos limpios o todos basura). Intenta con un archivo que tenga datos mixtos.")
            
    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
        st.info("Asegúrate de que el archivo contenga las columnas: 'id', 'curp', 'clues2' (o 'clues'), 'fecha_derivacion' y 'caso'.")
