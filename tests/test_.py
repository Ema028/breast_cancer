import pytest
import pandas as pd
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "melhor_modelo.pkl"

features = ['mean concave points', 'worst radius', 'worst concavity',
            'mean texture', 'worst perimeter', 'mean fractal dimension', 'area error']

@pytest.fixture
def modelo_carregado():
    assert MODEL_PATH.exists(), "modelo não foi encontrado"
    return joblib.load(MODEL_PATH)

def test_previsao(modelo_carregado):
    #paciente fictício
    dados_falsos = pd.DataFrame([[0.1, 15.0, 0.2, 20.0, 100.0, 0.06, 20.0]], columns=features)
    
    previsao = modelo_carregado.predict(dados_falsos)
    
    #precisa ter 1 resposta e deve ser 0 ou 1
    assert len(previsao) == 1
    assert previsao[0] in [0, 1], "previu uma classe inválida"

def test_probabilidades(modelo_carregado):
    dados_falsos = pd.DataFrame([[0.1, 15.0, 0.2, 20.0, 100.0, 0.06, 20.0]], columns=features)
    
    probabilidades = modelo_carregado.predict_proba(dados_falsos)

    assert probabilidades.shape == (1, 2), "modelo deve retornar probabilidades para 2 classes"
    assert 0.0 <= probabilidades[0][0] <= 1.0, "probabilidade classe 0(doente) fora de limite"
    assert 0.0 <= probabilidades[0][1] <= 1.0, "Probabilidade classe 1(saudável) fora de limite"