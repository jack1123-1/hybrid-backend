from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime, timedelta
from urllib.parse import quote_plus
from Irradiance import IrradiancePredictor
from power_predictor import PowerPredictor

app = Flask(__name__)
CORS(app)

username = quote_plus("site")
password = quote_plus("@sitesignup515@")
uri = f"mongodb+srv://{username}:{password}@weather.crydxka.mongodb.net/?retryWrites=true&w=majority&appName=weather"
client = MongoClient(uri, server_api=ServerApi('1'))

API_KEY = '53b98a37f1c419b5fd53e30f5c4c0bde'

db = client["weather_database"]
readings_collection = db["winter_readings"]

irradiance_predictor = IrradiancePredictor()
power_predictor = PowerPredictor()

try:
    irradiance_model = irradiance_predictor.train()
    power_model = power_predictor.train()
    print("Models trained successfully")
except Exception as e:
    print(f"Error training models: {e}")
    irradiance_model = None
    power_model = None

@app.route("/esp32/get-readings", methods=["GET"])
def weather_live_prediction():
    try:
        lat = float(request.args.get("lat", -17.83))
        lon = float(request.args.get("lon", 31.05))
        
        url = f'https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={API_KEY}'
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return jsonify({"error": f"Weather API error: {response.status_code}"}), 500
        
        data = response.json()
        
        if not data.get("list"):
            return jsonify({"error": "No forecast data returned"}), 500
        
        forecast = data["list"][0]
        temperature = forecast["main"]["temp"]
        wind_speed_mps = forecast["wind"]["speed"]
        cloud_cover = forecast["clouds"]["all"]
        
        latest_reading = readings_collection.find_one(sort=[("_id", -1)])
        
        if not latest_reading:
            return jsonify({"error": "No live ESP32 readings found"}), 404
        
        overall_voltage = float(latest_reading.get("overall_voltage", 0))
        overall_current = float(latest_reading.get("overall_current", 0))
        
        voltage_data = {
            "overall_voltage": overall_voltage,
            "overall_current": overall_current,
        }
        
        try:
            irradiance_prediction = irradiance_predictor.predict(voltage_data)
            if isinstance(irradiance_prediction, dict):
                solar_irradiance = irradiance_prediction.get("solar_irradiance_wm2", 750)
                irradiance_details = irradiance_prediction
            else:
                solar_irradiance = float(irradiance_prediction)
                irradiance_details = {"solar_irradiance_wm2": solar_irradiance}
        except Exception as e:
            print(f"Irradiance prediction error: {e}")
            solar_irradiance = 750
            irradiance_details = {"solar_irradiance_wm2": solar_irradiance, "error": str(e)}
        
        input_data = {
            "overall_voltage": overall_voltage,
            "solar_irradiance": solar_irradiance,
            "wind_speed": wind_speed_mps,
            "cloud_cover": cloud_cover,
            "temperature": temperature,
        }
        
        try:
            power_prediction = power_predictor.predict(input_data)
            if isinstance(power_prediction, dict):
                predicted_power = power_prediction
            else:
                predicted_power = float(power_prediction)
        except Exception as e:
            print(f"Power prediction error: {e}")
            predicted_power = {"power_watts": 0, "error": str(e)}
        
        return jsonify({
                "voltage": overall_voltage,
                "current": overall_current,
                "powerPredicted": predicted_power
        }), 200
        
    except ValueError as e:
        return jsonify({"error": f"Invalid parameter: {e}"}), 400
    except requests.exceptions.Timeout:
        return jsonify({"error": "Weather API request timed out"}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Weather API request failed: {e}"}), 502
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": f"Internal server error: {e}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "irradiance_model": irradiance_model is not None,
            "power_model": power_model is not None
        }
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)