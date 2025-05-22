# Plataforma de Manutenção com Machine Learning - Projeto ViaBilidade - FIAP 

Este projeto consiste em uma solução inteligente para predição do **tempo real de manutenção** de trens, utilizando **Machine Learning**, com integração entre **Flask (API)** e uma **interface web**.

## Objetivo

Treinar e disponibilizar via API um modelo preditivo que, a partir de dados de manutenções anteriores, estime o **tempo real de uma nova manutenção**. Isso contribui para otimizar o planejamento técnico e os recursos da empresa.

---

## Tecnologias Utilizadas

- Python 3.13
- Pandas, Numpy
- Scikit-learn
- Flask
- HTML + CSS (frontend simples)
- Postman (para testes da API)
- VSCode / Jupyter Notebook

---

## Etapas do Projeto

1. **Coleta e limpeza dos dados**
2. **Análise exploratória**
3. **Treinamento e comparação de 3 modelos**
4. **Escolha do melhor modelo (com melhor score na validação)**
5. **Serialização com `joblib`**
6. **Criação de uma API Flask**
7. **Integração com página HTML para simulação de predições**
8. **Testes com Postman e web**

---

## 📊 Dados Utilizados

O dataset contém colunas como:
- Data
- Tipo de manutenção (`Corretiva` ou `Preventiva`)
- Tempo estimado
- Técnico responsável
- POP (procedimento operacional padrão)
- Necessidade de peças (`Sim` ou `Não`)
- Problema reincidente (`Sim` ou `Não`)
- Tempo real (variável a ser predita)
- Status da manutenção (`Concluído` ou `Pendente`)

---

## odelos Testados

Foram comparados 3 modelos:
- Regressão Linear
- Random Forest Regressor
- Gradient Boosting Regressor

✔️ O modelo com melhor desempenho na partição de teste foi **`RandomForestRegressor`**.

---

##  Execução Local

### Requisitos
Instale as dependências com:

```bash
pip install -r requirements.txt
cd app
python app.py

Acesse no navegador: http://127.0.0.1:5000/