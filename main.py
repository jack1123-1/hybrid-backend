from flask import Flask, request, jsonify
import random
import json
from urllib.parse import quote_plus
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from flask_cors import CORS

username = quote_plus("site")
password = quote_plus("@sitesignup515@")

app = Flask(__name__)
CORS(app)
uri = f"mongodb+srv://{username}:{password}@weather.crydxka.mongodb.net/?retryWrites=true&w=majority&appName=weather"
client = MongoClient(uri, server_api=ServerApi('1'))    

db = client["weather_database"]
summary_collection = db["winter_summary"]
hourly_collection = db["winter_hour"]
readings_collection = db["winter_readings"]
prediction_collection = db["winter_prediction_data"]

def return_last_update(iterable):
    last = len(iterable) - 1
    index = random.randint(0, last)
    if not iterable:
        return None
    return iterable[index]

def return_previous_update(iterable):
    last = len(iterable) - 1
    index = random.randint(1, last)
    if len(iterable) < 2:
        return None
    return iterable[index]

@app.route("/weather/add-prediction-data", methods=["GET"])
def add_prediction():
    with open("hybrid_constrained_realistic.json", 'r') as f:
        data = json.load(f)
    result = prediction_collection.insert_many(data)
    return f"Success", 201  

@app.route("/weather/get-prediction-data", methods=["GET"])
def get_prediction():
    try:
        prediction_data = list(prediction_collection.find({}, {"_id": 0}))
    except Exception as e:
        return f"Error, {e}"
    return jsonify(prediction_data), 200

@app.route("/weather/get-summary", methods=["GET"])
def get_weather():
    try:
        weather_data = list(summary_collection.find({}, {"_id": 0}))
    except Exception as e:
        return f"Error, {e}"
    return jsonify(weather_data), 200

@app.route("/weather/get-hour", methods=["GET"])
def get_hour():
    try:
        hour_data = list(hourly_collection.find({}, {"_id": 0}))
    except Exception as e:
        return f"Error, {e}"
    return jsonify(hour_data), 200

@app.route("/esp32/post-readings", methods=["GET"])
def post_readings():
    data = request.get_json()
    readings_collection.insert_many(data)
    return jsonify({"status": "success", "message": data}), 200

@app.route("/weather/get-readings", methods=["GET"])
def get_readings():
    try:
        readings_data = list(readings_collection.find({}, {"_id": 0}))
    except Exception as e:
        return f"Erro, {e}"
    return jsonify(readings_data), 200

@app.route("/weather/get-changes", methods=["GET"])
def get_changes():
    for _ in range(20):
        try:
            wind_current_values = list(readings_collection.find({"wind_current": {"$exists": True}}))
            wind_voltage_values = list(readings_collection.find({"wind_voltage": {"$exists": True}}))
            solar_current_values = list(readings_collection.find({"solar_current": {"$exists": True}}))
            solar_voltage_values = list(readings_collection.find({"solar_voltage": {"$exists": True}}))
            current_wind_current_value = return_last_update(wind_current_values)
            previous_wind_current_value = return_previous_update(wind_current_values)
            current_wind_voltage_value = return_last_update(wind_voltage_values)
            previous_wind_voltage_value = return_previous_update(wind_voltage_values)       
            current_solar_voltage_value = return_last_update(solar_voltage_values)
            previous_solar_voltage_value = return_previous_update(solar_voltage_values)
            current_solar_current_value = return_last_update(solar_current_values)
            previous_solar_current_value = return_previous_update(solar_current_values)
            wind_current_change = current_wind_current_value["wind_current"] - previous_wind_current_value["wind_current"]
            wind_voltage_change = current_wind_voltage_value["wind_voltage"] - previous_wind_voltage_value["wind_voltage"]
            solar_current_change = current_solar_current_value["solar_current"] - previous_solar_current_value["solar_current"]
            solar_voltage_change = current_solar_voltage_value["solar_current"] - previous_solar_voltage_value["solar_current"]
            data = {'wind_current_change': wind_current_change, 'wind_voltage_change': wind_voltage_change,
                    'solar_current_change': solar_current_change, 'solar_voltage_change': solar_voltage_change
                    }
            return data
        except Exception as e:
            return f"Error, {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
