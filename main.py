from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime, timedelta
from urllib.parse import quote_plus
import json
import random

# External imports (assumed to exist)
from scrapper import Scrapper
from Irradiance import IrradiancePredictor
from linear_regression import PowerPredictor
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

app = Flask(__name__)
CORS(app)

username = quote_plus("site")
password = quote_plus("@sitesignup515@")
uri = f"mongodb+srv://{username}:{password}@weather.crydxka.mongodb.net/?retryWrites=true&w=majority&appName=weather"
client = MongoClient(uri, server_api=ServerApi('1'))

db = client["weather_database"]
summary_collection = db["winter_summary"]
hourly_collection = db["winter_hour"]
readings_collection = db["winter_readings"]
prediction_collection = db["winter_prediction_data"]

scrapper = Scrapper()
irradiance_predictor = IrradiancePredictor()
power_predictor = PowerPredictor()
power_model = power_predictor.train()
irradiance_model = irradiance_predictor.train()

HARD_CODED_SOLAR_IRRADIANCE = 782
scheduler = BackgroundScheduler()
solar_range = random.randint(12, 24)
wind_range = random.randint(1, 7)

def hourly_scrape():
    try:
        data = scrapper.get_hour_data()
        voltage_data = {'solar_voltage': solar_range, 'wind_voltage': wind_range}
        irradiance_data = irradiance_predictor.predict(voltage_data)
        data.update(irradiance_data)
        hourly_collection.insert(data)
    except Exception as e:
        print(f"Hourly scrape error: {e}")

def summary_scrape():
    try:
        data = scrapper.get_summary()
        summary_collection.insert(data)
    except Exception as e:
        print(f"Summary scrape error: {e}")

scheduler.add_job(func=summary_scrape, trigger=CronTrigger(hour=8, minute=0))
scheduler.add_job(func=summary_scrape, trigger=IntervalTrigger(minutes=60))
scheduler.start()

def return_last_update(data):
    return random.choice(data) if data else None

def return_previous_update(data):
    return random.choice(data[:-1]) if len(data) > 1 else None

@app.route("/weather/add-hourly", methods=["POST"])
def add_hourly_bulk():
    try:
        with open("hourly_seven_day.json", "r") as f:
            new_data = json.load(f)
        hourly_collection.insert_many(new_data)
        return jsonify({"status": "success", "inserted": len(new_data)}), 201
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/weather/add-prediction-data", methods=["GET"])
def add_prediction():
    with open("hybrid_constrained_realistic.json", 'r') as f:
        data = json.load(f)
    prediction_collection.insert_many(data)
    return "Success", 201

@app.route("/weather/get-prediction-data", methods=["GET"])
def get_prediction():
    data = list(prediction_collection.find({}, {"_id": 0}))
    return jsonify(data), 200

@app.route("/weather/get-summary", methods=["GET"])
def get_weather():
    data = list(summary_collection.find({}, {"_id": 0}))
    return jsonify(data), 200

@app.route("/weather/get-hour", methods=["GET"])
def get_hour():
    try:
        data = list(hourly_collection.find({}, {"_id": 0}))
        results = []
        for record in data:
            input_data = {
                'solar_irradiance': record.get('solar_irradiance_wm2', 0),
                'solar_voltage': record.get('solar_voltage', 0),
                'wind_voltage': record.get('wind_voltage', 0),
                'cloud_cover': record.get('cloud_cover_percent', 0),
                'temperature': record.get('temperature_c', 0),
                'wind_speed': round(record.get('wind_speed_kmh', 0) / 3.6, 2),
            }
            power = power_predictor.predict(input_data)
            record['predicted_power'] = power
            results.append(record)
        return jsonify(results), 200
    except Exception as e:
        return f"Error, {e}", 500

@app.route("/data/predicted", methods=["GET"])
def get_predicted_data():
    try:
        latest = hourly_collection.find_one(sort=[("_id", -1)])
        if not latest:
            return jsonify({"message": "No predicted data available"}), 404

        input_data = {
            "solar_irradiance": latest.get("solar_irradiance_wm2", 0),
            "wind_speed": round(latest.get("wind_speed_kmh", 0) / 3.6, 2),
            "cloud_cover": latest.get("cloud_cover_percent", 0),
            "temperature": latest.get("temperature_c", 0),
            "solar_voltage": 24,
            "wind_voltage": 3,
        }
        power = power_predictor.predict(input_data)
        prediction = {
            "solar-irradiance": input_data["solar_irradiance"],
            "windspeed": input_data["wind_speed"],
            "powergenerated": power
        }
        return jsonify({"predicted": prediction}), 200
    except Exception as e:
        return (f"Error: {e}", 500)

@app.route("/data/live", methods=["GET"])
def get_live_data():
    try:
        latest = readings_collection.find_one(sort=[("_id", -1)])
        if not latest:
            return jsonify({"message": "No data available"}), 404

        voltage = latest.get("overall_voltage", 0)
        current = latest.get("overall_current", 0)
        power = round(voltage * current, 2)

        return jsonify({
            "live": {
                "powergenerated": power,
                "voltage": voltage,
                "current": current
            }
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/data/historical", methods=["GET"])
def get_historical_data():
    try:
        compare_data = []
        
        recent_hourly = hourly_collection.find().sort("_id", -1).limit(24)

        for record in recent_hourly:
            input_data = {
                "solar_irradiance": record.get("solar_irradiance_wm2", 0),
                "wind_speed": round(record.get("wind_speed_kmh", 0) / 3.6, 2),
                "cloud_cover": record.get("cloud_cover_percent", 0),
                "temperature": record.get("temperature_c", 0),
                "solar_voltage": 24,
                "wind_voltage": 3,
            }
            compare_data.append({
                "powergenerated": power_predictor.predict(input_data),
                "windspeed": input_data["wind_speed"],
                "solarIrr": input_data["solar_irradiance"]
            })

        historical_data = {
            "powergeneratedPerHour": compare_data
        }

        return jsonify(historical_data), 200

    except Exception as e:
        return (f"Error: {e}", 500)



@app.route("/data/previous", methods=["GET"])
def get_previous_data():
    try:
        readings = list(readings_collection.find().sort("_id", -1).limit(2))
        if len(readings) < 2:
            return jsonify({"message": "Not enough data"}), 404

        prev = readings[1]
        voltage = prev.get("overall_voltage", 0)
        current = prev.get("overall_current", 0)
        power = round(voltage * current, 2)

        return jsonify({
            "previous": {
                "powergenerated": power
            }
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/esp32/post-readings", methods=["POST"])
def post_readings():
    try:
        data = request.get_json(force=True)
        readings = {
            'overall_voltage': data.get('overall_voltage'),
            'overall_current': data.get('overall_current'),
            'wind_voltage': data.get('wind_voltage'),
            'timestamp': datetime.utcnow().isoformat()
        }
        readings_collection.insert_one(readings)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/weather/get-readings", methods=["GET"])
def get_readings():
    data = list(readings_collection.find({}, {"_id": 0}))
    return jsonify(data), 200

@app.route("/weather/get-changes", methods=["GET"])
def get_changes():
    try:
        wind_currents = list(readings_collection.find({"wind_current": {"$exists": True}}))
        wind_voltages = list(readings_collection.find({"wind_voltage": {"$exists": True}}))
        solar_currents = list(readings_collection.find({"solar_current": {"$exists": True}}))
        solar_voltages = list(readings_collection.find({"solar_voltage": {"$exists": True}}))

        changes = {
            'wind_current_change': return_last_update(wind_currents)["wind_current"] - return_previous_update(wind_currents)["wind_current"],
            'wind_voltage_change': return_last_update(wind_voltages)["wind_voltage"] - return_previous_update(wind_voltages)["wind_voltage"],
            'solar_current_change': return_last_update(solar_currents)["solar_current"] - return_previous_update(solar_currents)["solar_current"],
            'solar_voltage_change': return_last_update(solar_voltages)["solar_voltage"] - return_previous_update(solar_voltages)["solar_voltage"]
        }
        return jsonify(changes), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
