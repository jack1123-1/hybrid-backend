import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def load_json_data():
    with open('hybrid_constrained_realistic.json', 'r') as f:
        data = json.load(f)
        df = pd.DataFrame(data)
    return df
data = load_json_data()


class IrradiancePredictor:
    def __init__(self):
        self.irradiance_model = LinearRegression()
        self.is_trained = False

    def train(self, test_size=0.3, random_state=70):
        df = data

        irradiance_features = df[['solar_voltage_v', 'wind_voltage_v']]
        irradiance_target = df['solar_irradiance_wm2']

        if test_size > 0:
            irradiance_X_train, irradiance_X_test, irradiance_y_train, irradiance_y_test = train_test_split(
                irradiance_features, irradiance_target, test_size=test_size, random_state=random_state)
            self.irradiance_model.fit(irradiance_X_train, irradiance_y_train)
            irradiance_pred = self.irradiance_model.predict(irradiance_X_test)
        self.is_trained = True

    def predict(self, conditions):
        if not self.is_trained:
            raise ValueError("Model not trained")

        irradiance_input = pd.DataFrame(
            [[conditions['overall_voltage'], conditions['wind_voltage']]],
            columns=['overall_voltage_v', 'wind_voltage_v'])
        irradiance_value = self.irradiance_model.predict(irradiance_input)[0]

        return max(0, irradiance_value)
    
    def get_model_info(self):
        return {
            'is_trained': self.is_trained,
            'metrics': self.model_metrics,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }