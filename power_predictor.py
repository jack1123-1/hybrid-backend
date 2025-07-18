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
        self.power_model = LinearRegression()
        self.is_trained = False
        self.model_metrics = {}
        self.feature_columns = []
    
    def train(self, test_size=0.3, random_state=70):
        df = data
        
        self.feature_columns = ['solar_voltage_v', 'wind_voltage_v', 'solar_irradiance_wm2', 'cloud_cover_percent', 'wind_speed_kmh']
        power_features = df[self.feature_columns]
        
        total_power_watts = df['combined_voltage_output_v'] * df['combined_current_output_a']
        power_target = total_power_watts
        
        if test_size > 0:
            power_X_train, power_X_test, power_y_train, power_y_test = train_test_split(
                power_features, power_target, test_size=test_size, random_state=random_state)
            
            self.power_model.fit(power_X_train, power_y_train)
            
            power_pred = self.power_model.predict(power_X_test)
            
            self.model_metrics = {
                'mse': mean_squared_error(power_y_test, power_pred),
                'r2': r2_score(power_y_test, power_pred),
                'rmse': np.sqrt(mean_squared_error(power_y_test, power_pred))
            }
        else:
            self.power_model.fit(power_features, power_target)
            self.model_metrics = {'note': 'Trained on full dataset, no test metrics available'}
        
        self.is_trained = True
        return self.power_model
    
    def predict(self, conditions):
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        solar_voltage = conditions.get('overall_voltage', 0) 
        wind_voltage = conditions.get('wind_voltage', 0)
        solar_irradiance = conditions.get('solar_irradiance', 0)
        cloud_cover = conditions.get('cloud_cover', 0)
        wind_speed = conditions.get('wind_speed', 0)
        
        power_input = pd.DataFrame([[solar_voltage, wind_voltage, solar_irradiance, cloud_cover, wind_speed]],
                                   columns=self.feature_columns)
        
        total_power = self.power_model.predict(power_input)[0]
        
        return max(0, total_power)
    
    def get_model_info(self):
        return {
            'is_trained': self.is_trained,
            'metrics': self.model_metrics,
            'feature_columns': self.feature_columns
        }