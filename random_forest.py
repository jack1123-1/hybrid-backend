import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np

def load_json_data():
    with open('hybrid_constrained_realistic.json', 'r') as f:
        data = json.load(f)
        df = pd.DataFrame(data)
    return df


data = load_json_data()


class SolarPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.is_solar_trained = False

    def train(self):
        df = pd.DataFrame(data)

        data_features = ['temperature_c', 'humidity_percent', 'cloud_cover_percent',
                         'wind_speed_kmh', 'solar_irradiance_wm2', 'wind_voltage_v']
        solar_target = 'solar_voltage_v'

        solar_X = df[data_features]
        solar_y = df[solar_target]

        self.model.fit(solar_X, solar_y)
        self.is_solar_trained = True

        solar_predictions = self.model.predict(solar_X)
        accuracy = r2_score(solar_y, solar_predictions)
        print(f"Solar model R_squared Score: {accuracy:.3f}")

        importance = self.model.feature_importances_
        print("\nSolar Model Feature Importance:")
        for feature, imp in zip(data_features, importance):
            print(f"  {feature}: {imp:.3f}")

    def predict(self, temperature, humidity, cloud_cover, wind_speed, solar_irradiance, wind_voltage=0):
        if not self.is_solar_trained:
            raise ValueError("Model must be trained first")

        X = [[temperature, humidity, cloud_cover, wind_speed, solar_irradiance, wind_voltage]]
        solar_voltage = self.model.predict(X)[0]

        return max(0, solar_voltage)


class WindPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.is_wind_trained = False

    def train(self):
        df = pd.DataFrame(data)

        data_features = ['temperature_c', 'humidity_percent', 'cloud_cover_percent',
                         'wind_speed_kmh', 'solar_irradiance_wm2', 'solar_voltage_v']
        wind_target = 'wind_voltage_v'

        wind_X = df[data_features]
        wind_y = df[wind_target]

        self.model.fit(wind_X, wind_y)
        self.is_wind_trained = True

        wind_predictions = self.model.predict(wind_X)
        accuracy = r2_score(wind_y, wind_predictions)
        print(f"Wind model R_squared Score: {accuracy:.3f}")

        importance = self.model.feature_importances_
        print("\nWind Model Feature Importance:")
        for feature, imp in zip(data_features, importance):
            print(f"  {feature}: {imp:.3f}")

    def predict(self, temperature, humidity, cloud_cover, wind_speed, solar_irradiance, solar_voltage=0):
        if not self.is_wind_trained:
            raise ValueError("Model must be trained first")

        X = [[temperature, humidity, cloud_cover, wind_speed, solar_irradiance, solar_voltage]]
        wind_voltage = self.model.predict(X)[0]

        return max(0, wind_voltage)


class PowerPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.is_trained = False

    def train(self):
        df = pd.DataFrame(data)

        features = ['temperature_c', 'humidity_percent', 'cloud_cover_percent',
                    'wind_speed_kmh', 'solar_irradiance_wm2', 'wind_voltage_v', 'solar_voltage_v']

        power_X = df[features]

        if 'combined_voltage_output_v' in df.columns and 'combined_current_output_a' in df.columns:
            total_power_watts = df['combined_voltage_output_v'] * df['combined_current_output_a']
        else:
            total_power_watts = (df['wind_voltage_v'] + df['solar_voltage_v']) * 2

        power_y = total_power_watts

        self.model.fit(power_X, power_y)
        self.is_trained = True

        predictions = self.model.predict(power_X)
        accuracy = r2_score(power_y, predictions)
        print(f"Power model R_squared Score: {accuracy:.3f}")

        importance = self.model.feature_importances_
        print("\nPower Model Feature Importance:")
        for feature, imp in zip(features, importance):
            print(f"  {feature}: {imp:.3f}")

    def predict(self, temperature, humidity, cloud_cover, wind_speed, solar_irradiance, wind_voltage=0,
                solar_voltage=0):
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        X = [[temperature, humidity, cloud_cover, wind_speed, solar_irradiance, wind_voltage, solar_voltage]]
        power = self.model.predict(X)[0]

        return max(0, power)

    def recommend(self, predicted_power):
        if predicted_power > 200:
            return "High power: Run washing machine, heater, multiple devices"
        elif predicted_power > 100:
            return "Medium power: Good for TV, laptop, lights"
        else:
            return "Low power: Only essential devices"


if __name__ == "__main__":
    solar_predictor = SolarPredictor()
    wind_predictor = WindPredictor()
    power_predictor = PowerPredictor()

    print("Training Solar Predictor...")
    solar_predictor.train()
    print("\nTraining Wind Predictor...")
    wind_predictor.train()
    print("\nTraining Power Predictor...")
    power_predictor.train()

    test_conditions = {
        'temperature': 22,
        'humidity': 40,
        'cloud_cover': 20,
        'wind_speed': 8,
        'solar_irradiance': 800
    }

    solar_voltage = solar_predictor.predict(
        test_conditions['temperature'],
        test_conditions['humidity'],
        test_conditions['cloud_cover'],
        test_conditions['wind_speed'],
        test_conditions['solar_irradiance']
    )

    wind_voltage = wind_predictor.predict(
        test_conditions['temperature'],
        test_conditions['humidity'],
        test_conditions['cloud_cover'],
        test_conditions['wind_speed'],
        test_conditions['solar_irradiance']
    )

    power = power_predictor.predict(
        test_conditions['temperature'],
        test_conditions['humidity'],
        test_conditions['cloud_cover'],
        test_conditions['wind_speed'],
        test_conditions['solar_irradiance'],
        wind_voltage,
        solar_voltage
    )

    print(f"\n=== PREDICTIONS ===")
    print(f"Solar voltage: {solar_voltage:.1f} V")
    print(f"Wind voltage: {wind_voltage:.1f} V")
    print(f"Predicted power output: {power:.1f} watts")
    print(f"Recommendation: {power_predictor.recommend(power)}")
