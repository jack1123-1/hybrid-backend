from flask import Flask, request, jsonify
import random
import json
from urllib.parse import quote_plus
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from flask_cors import CORS
from scrapper import Scrapper
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from Irradiance import IrradiancePredictor
from linear_regression import PowerPredictor

username = quote_plus("site")
password = quote_plus("@sitesignup515@")

app = Flask(__name__)
CORS(app)
scrapper = Scrapper()
uri = f"mongodb+srv://{username}:{password}@weather.crydxka.mongodb.net/?retryWrites=true&w=majority&appName=weather"
client = MongoClient(uri, server_api=ServerApi('1'))    
scheduler = BackgroundScheduler()
irradiance_predictor = IrradiancePredictor()
power_predictor = PowerPredictor()
power_model = power_predictor.train()
irradiance_model = irradiance_predictor.train()

db = client["weather_database"]
summary_collection = db["winter_summary"]
hourly_collection = db["winter_hour"]
readings_collection = db["winter_readings"]
prediction_collection = db["winter_prediction_data"]

solar_range = random.randint(12, 24)
wind_range = random.randint(1, 7)

def hourly_scrape():
    try:
        data = scrapper.get_hour_data()
        voltage_data = {
            'solar_voltage': solar_range,
            'wind_voltage': wind_range,
        }
        irradiance_data = irradiance_predictor.predict(voltage_data)
        data.update(irradiance_data)
    except Exception as e:
        return f"Error, {e}"
    print(f"Data to be appended: {data}")
    hourly_collection.insert(data)
    print("hourly weather data updated")

def summary_scrape():
    try:
        data = scrapper.get_summary()
    except Exception as e:
        return f"Error, {e}"
    print(f"Data to be appended: {data}")
    summary_collection.insert(data)
    print("summary data updated")

scheduler.add_job(func=summary_scrape, trigger=CronTrigger(hour=8, minute=0))
scheduler.add_job(func=summary_scrape, trigger=IntervalTrigger(minutes=60))
scheduler.start()

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
        
        results = []
        for record in hour_data:
#            solar_iradiance = record.get('solar_irradiance', 0)
            cloud_cover = record.get('cloud_cover', 0)
            wind_speed = record.get('wind', 0)
            temperature = record.get('temperature', 0)
            data = {
                'solar_irradiance': 787,
                'solar_voltage': 24,
                'wind_voltage': 3,
                'cloud_cover': cloud_cover,
                'temperature': temperature,
                'wind_speed': wind_speed,
            }
            power = power_predictor.predict(data)
            record['predicted_power'] = power
            results.append(record)
        return jsonify(results), 200
    except Exception as e:
        return f"Error, {e}", 500


@app.route("/esp32/post-readings", methods=["GET"])
def post_readings():
    data = request.get_json()
    overall_voltage = data['output_voltage']
    overall_current = data['output_current']
    wind_voltage = data['wind_voltage']
    readings = {'overall_voltage': overall_voltage, 'overall_current': overall_current, 'wind_voltage': wind_voltage}
    readings_collection.insert_many(readings)
    return jsonify({"status": "success"}), 200

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
