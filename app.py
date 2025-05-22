from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Carregar modelo e scaler
modelo = joblib.load("modelo.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
        
        features = [
            int(data["tipo_manut"]),
            int(data["tec_resp"]),
            int(data["POP_utiliz"]),
            int(data["problema_recorr"]),
            int(data["falha_crit"]),
            float(data["tempo_resolucao_padr"]),
            int(data["status_manut"])
        ]
        
        input_scaled = scaler.transform([features])
        prediction = modelo.predict(input_scaled)
        
        # Se veio do navegador (form), retorna o template com resultado
        if not request.is_json:
            return render_template("index.html", prediction_text=f"Tempo previsto: {prediction[0]:.2f} minutos")
        
        # Se veio JSON (ex: Postman), retorna JSON
        return jsonify({"tempo_previsto": round(prediction[0], 2)})
    
    except Exception as e:
        print(f"Erro na rota /predict: {e}")
        return jsonify({"erro": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
