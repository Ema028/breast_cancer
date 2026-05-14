import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "melhor_modelo.pkl"

modelo = joblib.load(MODEL_PATH)

st.title("Modelo de Triagem Oncológica")
st.write("Insira os dados da biópsia para classificação:")

mean_concave_points = st.number_input("Média de pontos côncavos do tumor(severidade dos contornos do núcleo)", min_value=0.0)
worst_radius = st.number_input("Média dos três maiores raios celulares medidos(piores casos)", min_value=0.0)
worst_concavity = st.number_input("Média das três maiores concavidades(profundidade das depressões no contorno)", min_value=0.0)
mean_texture = st.number_input("Textura média celular(variação na escala de cinza da imagem)", min_value=0.0)
worst_perimeter = st.number_input("Média dos três maiores perímetros celulares(piores casos)", min_value=0.0)
mean_fractal_dimension = st.number_input("Dimensão fractal média(complexidade da borda celular)", min_value=0.0)
area_error = st.number_input("Erro padrão da área celular", min_value=0.0)

if st.button("Gerar Diagnóstico"):
    try:
        features_reduzidas = ['mean concave points', 'worst radius', 'worst concavity', 'mean texture', 'worst perimeter', 'mean fractal dimension', 'area error']
        dados_paciente = pd.DataFrame([[mean_concave_points, worst_radius, worst_concavity, mean_texture, worst_perimeter,
                                        mean_fractal_dimension, area_error]], columns=features_reduzidas)

        previsao = modelo.predict(dados_paciente)
        probabilidade = modelo.predict_proba(dados_paciente)[0][0] * 100 #prob da classe0(com câncer)

        if previsao[0] == 0: st.error(f"Modelo previu tumor Maligno com {probabilidade:.2f}% de chance de câncer")
        else: st.success(f"Modelo previu tumor Benigno com {probabilidade:.2f}% de chance de câncer")
    except Exception as e: st.error(f"Erro: {e}")