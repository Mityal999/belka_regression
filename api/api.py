from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import pickle
import numpy as np
import pandas as pd
import json
import joblib
import xgboost as xgb


app = Flask(__name__)
api = Api(app)


# Define the method
class ApartmentRegressor(Resource):
    def put(self, data):
        # Get input
        parsed_list = json.loads(data)
        input = np.array(parsed_list)
        # Predict
        pred_in_shape = model.predict(input).reshape(1,-1)
        pred_unscaled = (y_scaler.inverse_transform(pred_in_shape * 3))
        # Export pred
        pred_unscaled_df = pd.DataFrame(pred_unscaled, columns=['Прогноз цены, р.'])
        result = pred_unscaled_df.iloc[0].to_json(force_ascii=False) 
        return {'data': result}
api.add_resource(ApartmentRegressor, '/apartments/<string:data>')


if __name__ == '__main__':

    # Paths
    SCALER_FILEPATH = 'scaler.save'
    XGB_FILEPATH = 'xgb_r_model.pkl'

    # Load scaler
    y_scaler = joblib.load(SCALER_FILEPATH)

    # Load XGBoost
    model = pickle.load(open(XGB_FILEPATH, "rb"))

    app.run(debug=True)