import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Título del dashboard
st.title("Plataforma de Salud Mental para Empleados - Interbank")
st.markdown("""
Bienvenido a la Plataforma de Salud Mental para Empleados de Interbank.  
📊 Esta herramienta analiza información clave del personal para detectar riesgos de salud mental.

**Instrucciones:**
1. Sube un archivo Excel con los datos de empleados (con columnas como `riesgo`, `nivel_burnout`, etc).
2. Visualiza métricas clave en el dashboard.
3. Usa los controles para predecir el riesgo de un nuevo empleado.

---
""")

# Cargar datos desde archivo
archivo = st.file_uploader("📎 Sube el archivo Excel con los datos de empleados", type=["xlsx"])

if archivo is not None:
    df = pd.read_excel(archivo)

    # Asegurar que la columna de fecha esté en formato datetime
    df["fecha_registro"] = pd.to_datetime(df["fecha_registro"], errors='coerce')

    # Filtro de fechas
    st.subheader("📆 Filtro por rango de fechas")
    min_fecha = df["fecha_registro"].min()
    max_fecha = df["fecha_registro"].max()

    fecha_inicio, fecha_fin = st.date_input(
        "Selecciona el rango de fechas",
        [min_fecha, max_fecha],
        min_value=min_fecha,
        max_value=max_fecha
    )

    # Filtrar los datos según la fecha
    filtro_fecha = (df["fecha_registro"] >= pd.to_datetime(fecha_inicio)) & (df["fecha_registro"] <= pd.to_datetime(fecha_fin))
    df_filtrado = df[filtro_fecha]

    # Mostrar tabla
    st.subheader("📋 Vista previa de los datos filtrados")
    st.dataframe(df_filtrado.head())

    # Entrenar modelo IA con columnas clave
    columnas_utiles = [
        "horas_trabajadas", "dias_ausencia", "estres_encuesta",
        "evaluacion_desempeno", "tiempo_respuesta_emails",
        "feedback_negativo", "feedback_positivo", "nivel_burnout",
        "satisfaccion_laboral", "encuesta_motivacion",
        "uso_licencia_mental", "apoyo_psicologico_empresa",
        "resiliencia_autoevaluada", "presion_objetivos"
    ]
    X = df_filtrado[columnas_utiles]
    y = df_filtrado["riesgo"]

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    st.subheader("📊 Distribución de riesgos")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df_filtrado, x="riesgo", ax=ax1, palette="Set2")
    st.pyplot(fig1)

    st.subheader("🔥 Nivel promedio de burnout por riesgo")
    fig2, ax2 = plt.subplots()
    sns.barplot(data=df_filtrado, x="riesgo", y="nivel_burnout", ax=ax2, palette="coolwarm")
    st.pyplot(fig2)

    st.subheader("📈 Evolución del riesgo en el tiempo")
    df_filtrado['mes'] = df_filtrado['fecha_registro'].dt.to_period('M').astype(str)
    riesgo_tiempo = df_filtrado.groupby(['mes', 'riesgo']).size().unstack(fill_value=0)
    fig3, ax3 = plt.subplots(figsize=(10,5))
    riesgo_tiempo.plot(kind='line', marker='o', ax=ax3)
    ax3.set_title("Tendencia del riesgo mental a lo largo del tiempo")
    ax3.set_xlabel("Mes")
    ax3.set_ylabel("Número de empleados")
    ax3.legend(title="Riesgo")
    ax3.grid(True)
    st.pyplot(fig3)

    st.subheader("🧠 Predicción de riesgo de salud mental para nuevo empleado")
    entrada = {}
    for col in columnas_utiles:
        valor = st.slider(f"{col.replace('_', ' ').capitalize()}", 0, 10, 5)
        entrada[col] = valor

    if st.button("Predecir riesgo"):
        nuevo = pd.DataFrame([entrada])
        riesgo_pred = modelo.predict(nuevo)[0]

        colores = {"bajo": "#C8E6C9", "medio": "#FFF9C4", "alto": "#FFCDD2"}

        st.markdown(
            f"""
            <div style='padding:20px; background-color:{colores[riesgo_pred]}; border-radius:10px; font-size:20px; 
            font-weight:bold; text-align:center;'>
            Riesgo estimado: {riesgo_pred.upper()}
            </div>
            """,
            unsafe_allow_html=True
        )

        if riesgo_pred == "alto":
            st.warning("⚠️ Se recomienda contacto con bienestar y apoyo inmediato.")
        elif riesgo_pred == "medio":
            st.info("🔶 Evaluar carga laboral y seguimiento cercano.")
        else:
            st.success("🟢 El empleado no presenta señales de riesgo actuales.")

else:
    st.info("🔄 Por favor, sube un archivo Excel para visualizar y analizar los datos.")
