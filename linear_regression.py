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


class PowerPredictor:
    def __init__(self):
        self.solar_model = LinearRegression()
        self.wind_model = LinearRegression()
        self.power_model = LinearRegression()
        self.is_trained = False

    def train(self, test_size=0.3, random_state=70):
        df = data

        power_features = df[['solar_voltage_v', 'wind_voltage_v', 'solar_irradiance_wm2', 'cloud_cover_percent', 'wind_speed_kmh']]
        total_power_watts = df['combined_voltage_output_v'] * df['combined_current_output_a']
        power_target = total_power_watts

        if test_size > 0:
            power_X_train, power_X_test, power_y_train, power_y_test = train_test_split(
                power_features, power_target, test_size=test_size, random_state=random_state)

            self.power_model.fit(power_X_train, power_y_train)

            power_pred = self.power_model.predict(power_X_test)

        self.is_trained = True

    def predict(self, conditions):
        if not self.is_trained:
            raise ValueError("Model not trained")

        power_input = pd.DataFrame([[conditions['solar_irradiance'], conditions['solar_voltage'], conditions['wind_voltage'], conditions['cloud_cover'], conditions['wind_speed']]],
                                   columns=['solar_voltage_v', 'wind_voltage_v', 'solar_irradiance_wm2', 'cloud_cover_percent', 'wind_speed_kmh'])
        total_power = self.power_model.predict(power_input)[0]

        return max(0, total_power)

    def recommend_devices(self, predicted_power):
        if predicted_power > 200:
            return "High power: {Recommendations: washing machine, heater, multiple devices}"
        elif predicted_power > 100:
            return "Medium power: {Recommendations: TV, laptop, lights}"
        else:
            return "Low power: {Recommendations: Only essential devices, ie phone charging}"

"""
if __name__ == "__main__":
    predictor = PowerPredictor()
    predictor.train()

    test_conditions = {
        'solar_irradiance': 800,
        'cloud_cover': 20,
        'temperature': 22,
        'wind_speed': 8,
    }
    prediction = predictor.predict(test_conditions)

    print("Prediction for new conditions:")
    print(f"Solar irradiance: {test_conditions['solar_irradiance']} W/m²")
    print(f"Cloud cover: {test_conditions['cloud_cover']}%")
    print(f"Temperature: {test_conditions['temperature']}°C")
    print(f"Wind speed: {test_conditions['wind_speed']} km/h")
    print()
    print(f"Predicted solar voltage: {prediction['solar_voltage']:.2f}V")
    print(f"Predicted wind voltage: {prediction['wind_voltage']:.2f}V")
    print(f"Predicted total power: {prediction['total_power']:.1f}W")
    print()
    print("Recommendation:", predictor.recommend_devices(prediction['total_power']))
"""