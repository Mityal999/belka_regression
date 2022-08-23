from flask import Flask, jsonify, request
import pickle
import numpy as np
import json
import joblib

app = Flask(__name__)


@app.post('/predict')
def predict():

    rquest_json = request.json
    try:
        input_text = rquest_json['data']
    except KeyError:
        return jsonify({'error': 'No text sent'})

    input_parsed = json.loads(input_text)
    input = np.array(input_parsed).reshape(1,-1)
    pred_scaled = model.predict(input)
    pred = y_scaler.inverse_transform(pred_scaled.reshape(1,-1) * 3)
    pred_float = np.squeeze(pred)

    try:
        result = jsonify({'prediction': str(pred_float)})
    except TypeError as e:
        result = jsonify({'error': str(e)})

    return result


if __name__ == '__main__':

    # Paths
    SCALER_FILEPATH = 'scaler.save'
    XGB_FILEPATH = 'xgb_r_model.pkl'

    # Load scaler
    y_scaler = joblib.load(SCALER_FILEPATH)

    # Load XGBoost
    model = pickle.load(open(XGB_FILEPATH, "rb"))

    app.run(host='0.0.0.0', debug=True)