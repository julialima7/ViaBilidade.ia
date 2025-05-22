
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Carregar base
df = pd.read_csv("base_viabilidade.csv", delimiter=";")
df = df.drop(columns=["dt_manut"])

# Tratar valores ausentes
df = df.dropna()

# Transformar variáveis categóricas
cat_cols = ["tipo_manut", "tec_resp", "POP_utiliz", "problema_recorr", "falha_crit", "status_manut"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Variáveis de entrada e saída
X = df.drop(columns=["tempo_manut"])
y = df["tempo_manut"]

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelos
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

melhor_modelo = None
melhor_r2 = -float("inf")

for nome, modelo in models.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{nome}: R² = {r2:.3f}, MAE = {mae:.2f}")
    if r2 > melhor_r2:
        melhor_modelo = modelo
        melhor_r2 = r2

# Salvar modelo e scaler
joblib.dump(melhor_modelo, "modelo.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Modelo salvo com sucesso.")
