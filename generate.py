import json
import random
from datetime import datetime, timedelta, timezone

def generate_time(hour: dict, minute: dict):
    hour = random.randint(hour[0], hour[1])
    minute = random.randint(minute[0], minute[1])
    am_pm = "am" if hour < 12 else "pm"
    return f"{hour}: {minute:02d}{am_pm}"

def generate_sun(rise_set:str, hour: dict, minute: dict):
    hour = random.randint(hour[0], hour[1])
    minute = random.randint(minute[0], minute[1])
    if rise_set == 'rise':
        am_pm = "am"
    am_pm = "pm"
    return f"{hour:02d}:{minute:02d}{am_pm}"

def generate_time_stamp():
    return (datetime.now(timezone.utc)).isoformat()

def generate_uv_index():
    value = random.randint(1, 11)
    return f"{value} of 11"

def generate_entry():
    high_value = random.randint(18, 35)
    low_value = random.randint(5, 14)
    temp = random.randint(low_value,high_value )

    return {
        "timestamp": generate_time_stamp(),
        "temperature": temp,
        "sunrise": generate_sun("rise", [5, 7], [0, 59]),
        "sunset": generate_sun("set", [5, 7], [0, 59]),
        "high": high_value,
        "low": low_value,
        "wind": random.randint(0, 20),
        "pressure": random.randint(14, 33),
        "ultra violet index": generate_uv_index()
    }

def generate_readings():
    wind_voltage =random.randint(12, 20)
    solar_voltage = random.randint(25, 40)
    wind_current = random.randint(2, 12)
    solar_current = random.randint(4, 12)
    solar_hourly_power_generated = random.randint(6, 30)
    wind_hourly_power_generated = random.randint(6, 30)
    hourly_power_consumed = random.randint(1, 3)
    daily_power_generated = random.randint(5, 30)
    daily_power_consumed = random.randint(3, 30)
    peak_usage_hour = random.randint(1, 24)
    min_usage_hour = random.randint(1, 24)
    peakload = random.randint(5, 25)
    battery_status = random.choice(["true", "false"])
    state_of_charge = random.randint(0, 100)
    state_of_health = random.randint(0, 100)

    return{
        'wind_voltage': wind_voltage,
        'solar_voltage': solar_voltage,
        'wind_current': wind_current,
        'solar_current': solar_current,
        'solar_hourly_generated': solar_hourly_power_generated,
        'wind_hourly_generated': wind_hourly_power_generated,
        'hourly_power_consumed': hourly_power_consumed,
        'daily_power_consumed': daily_power_consumed,
        'daily_power_generated': daily_power_generated,
        'peak_usage_hour': peak_usage_hour,
        'min_usage_hour': min_usage_hour,
        'peakload': peakload,
        'battery_satus': battery_status,
        'state_of_charge': state_of_charge,
        'state_of_health': state_of_health,
    }

def generate_weather_data(num_entries=20, output_file="weather_data.json"):
    data = [generate_entry() for _ in range(num_entries)]
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"{num_entries} entries saved to the {output_file}")

def generate_reading_data(num_entries=20, output_file="instrument_readings.json"):
    data = [generate_readings() for _ in range(num_entries)]
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"{num_entries} entries saved to the {output_file}")

if __name__ == "__main__":
    generate_reading_data(num_entries=25)



