import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Título del dashboard
st.title("Plataforma de Salud Mental para Empleados - Interbank")

# Cargar datos desde archivo
archivo = st.file_uploader("Sube el archivo Excel con datos de empleados", type=["xlsx"])

if archivo:
    df = pd.read_excel(archivo)

    st.subheader("Vista previa de los datos cargados")
    st.dataframe(df.head())

    # Entrenar modelo IA con columnas clave
    columnas_utiles = [
        "horas_trabajadas", "dias_ausencia", "estres_encuesta",
        "evaluacion_desempeno", "tiempo_respuesta_emails",
        "feedback_negativo", "feedback_positivo", "nivel_burnout",
        "satisfaccion_laboral", "encuesta_motivacion",
        "uso_licencia_mental", "apoyo_psicologico_empresa",
        "resiliencia_autoevaluada", "presion_objetivos"
    ]
    X = df[columnas_utiles]
    y = df["riesgo"]

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    st.subheader("Distribución de riesgos")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="riesgo", ax=ax1, palette="Set2")
    st.pyplot(fig1)

    st.subheader("Nivel promedio de burnout por riesgo")
    fig2, ax2 = plt.subplots()
    sns.barplot(data=df, x="riesgo", y="nivel_burnout", ax=ax2, palette="coolwarm")
    st.pyplot(fig2)

    st.subheader("Predicción de riesgo de salud mental para nuevo empleado")
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
