import re
from enum import Enum
import time
import requests
from bs4 import BeautifulSoup
from enum import Enum
from datetime import datetime, timezone


number_regex = r'\d+'

class DataClass(Enum):
    TEMPERATURE = 'temperature'
    HIGH = 'high'
    LOW = 'low'
    SKY = 'sky'
    WIND = 'wind'
    SUNRISE = 'sunrise'
    SUNSET = 'sunset'
    PRESSURE = 'pressure'
    UV = 'ultra violet index'
    CD = 'cloud cover'
    HIGHLOW = 'high/low'
    TIME = 'time'


class ForecastOption(Enum):
    SUMMARY = 'today' 
    HOUR = 'hourbyhour'
    WHOLEDAY = 'hourbyhour'
    WEEKLY = 'tenday'
    WEEKEND = 'weekend'
    MONTHLY = 'monthly'

def day_time_count():
    hourly_increment = 1
    local_time = time.localtime()

    next_hour = time.localtime(time.mktime(local_time) + hourly_increment * 60 * 60)
    next_hour_format = str(time.strftime("%R", next_hour))

    for i in range(24):
        next_hour = time.localtime(time.mktime(local_time) + hourly_increment * 60 * 60)
        next_hour_format = str(time.strftime("%R", next_hour))
        if (next_hour_format[3:5] and next_hour_format[0:2]) != "00":
            next_hour_format = (next_hour_format[0:3]+"00")
            if next_hour_format[0] == "0":
                next_hour_format = next_hour_format[1:5]
                compare_format1 = int(next_hour_format[0])
                if compare_format1 <= 23:
                    print(f"Inner: {compare_format1}")
            else:
                compare_format2 = int(next_hour_format[0:2])
                print(compare_format2)
                if compare_format2 == 23:
                    break
            hourly_increment+=1
        else:
            hourly_increment += 1
    return hourly_increment


def time_conversion(time_string):
    times = {
        "1:00 am": 1,
        "2:00 am": 2,
        "3:00 am": 3,
        "4:00 am": 4,
        "5:00 am": 5,
        '6:00 am': 6,
        "7:00 am": 7,
        "8:00 am": 8,
        "9:00 am": 9,
        "10:00 am": 10,
        "11:00 am": 11,
        "12:00 am": 0,
        "1:00 pm": 13,
        "2:00 pm": 14,
        "3:00 pm": 15,
        "4:00 pm": 16,
        "5:00 pm": 17,
        "6:00 pm": 18,
        "7:00 pm":19,
        "8:00 pm": 20,
        "9:00 pm": 21,
        "10:00 pm": 22,
        "11:00 pm": 23,
        "12:00 pm": 24,
    }
    return times[time_string]

def parse_data(*args):
    if args[0] == DataClass.TEMPERATURE.value:
        try:
            tempe_val = re.findall(number_regex, args[1])
            if tempe_val:
                tempe_val = int(tempe_val[0])
                return int((tempe_val - 32) * (5 / 9))
            else:
                return "Error: Could not find a valid temperature value"
        except Exception as e:
            return f"Error, {e}"

    elif args[0] == DataClass.HIGHLOW.value:
        try:
            high_low_vals = re.findall(number_regex, args[1])
            if len(high_low_vals) > 1:
                high_store = int(high_low_vals[0])
                low_store = int(high_low_vals[1])
                high = int((high_store - 32) * (5 / 9))
                low = int((low_store - 32) * (5 / 9))
                return {'High': high, 'Low': low}
            else:
                high_store = 'NA'
                low_store = int(high_low_vals[0])
                else_low = int((low_store - 32) * (5 / 9))
                return {'High': high_store, 'Low': else_low}
        except Exception as e:
            return f"Error, {e}"
    

    elif args[0] == DataClass.WIND.value:
        try:
            wind_vals = re.findall(number_regex, args[1])
            if wind_vals:
                wind = int(int(wind_vals[0]) / 0.621371)
                return wind
            else:
                return "Error: Could not find wind speed value"
        except Exception as e:
            return f"Error. {e}"

    elif args[0] == DataClass.PRESSURE.value:
        try:
            pressure_vals = re.findall(number_regex, args[1])
            if pressure_vals:
                return int(pressure_vals[0])
            else:
                return "Error: Could not find pressure value"
        except Exception as e:
            return f"Error, {e}"
        
#    elif args[0] == DataClass.TIME.value:
#        try:
#            time = time_conversion(args[1])
#        except Exception as e:
#            return f"Error, {e}"
#        return time

    else:
        return "No defined conversion"


class WeatherCom:
    def __init__(self, user_agent=None):
        self.headers = {
            'User-Agent': user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }     

    def _get_html(self, option: ForecastOption, area_code):
        if option in (ForecastOption.SUMMARY, ForecastOption.HOUR, ForecastOption.WHOLEDAY, ForecastOption.WEEKLY):
            url = f'https://weather.com/weather/{option.value}/l/{area_code}'
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        else:
            for option in ForecastOption:
                print(f'Option: {option}\n')


class Scrapper():
    def __init__(self):
        pass

    def get_summary(self):
        area_code = 'ZIXX0004'
        option = ForecastOption.SUMMARY
        site = WeatherCom()
        html = site._get_html(option, area_code)
        soup = BeautifulSoup(html, 'html.parser')

        try:
            general_data_section = soup.find('div', id='todayDetails')
            current_temperature_section = general_data_section.find('div', {'data-testid':'FeelsLikeSection'})
            sunrise_section = general_data_section.find('div', {'data-testid':'SunriseValue'})
            sunrise = sunrise_section.find('p', class_='TwcSunChart--dateValue--TzXBr').text.strip()
            sunrise_plot = parse_data(DataClass.TIME.value, sunrise)
            sunset_section = general_data_section.find('div', {'data-testid':'SunsetValue'})
            sunset = sunset_section.find('p', class_='TwcSunChart--dateValue--TzXBr').text.strip()
            sunset_plot = parse_data(DataClass.TIME.value, sunset)
            current_temperature_data = current_temperature_section.find('span', {'data-testid':'TemperatureValue'}).text.strip()
            current_temperature = parse_data(DataClass.TEMPERATURE.value, current_temperature_data)
            high_low_temperature_data = general_data_section.find('div', {'data-testid':'wxData'}).text.strip()
            high_low_temperature_data_store = parse_data(DataClass.HIGHLOW.value, high_low_temperature_data)
            high = high_low_temperature_data_store["High"]
            low = high_low_temperature_data_store["Low"]
            wind_data = general_data_section.find('span', {'data-testid':'Wind'}).text.strip()
            wind = parse_data(DataClass.WIND.value, wind_data)
            pressure_data = general_data_section.find('span', {'data-testid':'PressureValue'}).text.strip()
            pressure = parse_data(DataClass.PRESSURE.value, pressure_data)
            uv = general_data_section.find('span', {'data-testid':'UVIndexValue'}).text.strip()    
            summary_data = {'TimeStamp':datetime.now(timezone.utc).now(), DataClass.TEMPERATURE.value:current_temperature,
                            'sunrise_plot': sunrise_plot, 'sunset_plot': sunset_plot, 
                            DataClass.SUNRISE.value:sunrise, DataClass.SUNSET.value:sunset,  
                            DataClass.HIGH.value:high, DataClass.LOW.value:low, DataClass.WIND.value:wind, 
                            DataClass.PRESSURE.value:pressure, DataClass.UV.value:uv}
            
        except Exception as e:
            return f'Error parsing summary data: {e}'
        return summary_data                 


    def get_hour_data(self):
        area_code = 'ZIXX0004'
        option = ForecastOption.HOUR
        site = WeatherCom()
        html = site._get_html(option, area_code)
        soup = BeautifulSoup(html, 'html.parser')
    
        try:
            hour_scrape_section = soup.find('details', {'data-testid': 'ExpandedDetailsCard-0'})
            extracted_time = hour_scrape_section.find('h2', {'data-testid':'daypartName'}).text.strip()
            hour_plot = parse_data(DataClass.TIME.value, extracted_time)
            temperature_section = hour_scrape_section.find('li', {'data-testid':'FeelsLikeSection'})
            temperature_data = temperature_section.find('span', {'data-testid':'TemperatureValue'}).text.strip()
            temperature = parse_data(DataClass.TEMPERATURE.value, temperature_data)
            sky = hour_scrape_section.find('p', {'data-testid':'hourlyWxPhrase'}).text.strip()
            wind_scrape_section = hour_scrape_section.find('li', {'data-testid':'WindSection'})
            wind_data = wind_scrape_section.find('span', {'data-testid':'Wind'}).text.strip()
            wind = parse_data(DataClass.WIND.value, wind_data)
            uv_scrape_section = hour_scrape_section.find('li', {'data-testid':'uvIndexSection'})
            uv = uv_scrape_section.find('span', {'data-testid':'UVIndexValue'}).text.strip()
            cloud_cover_section= hour_scrape_section.find('li', {'data-testid': 'CloudCoverSection'})
            cloud_cover = cloud_cover_section.find('span', {'data-testid':'PercentageValue'}).text.strip()
            hour_data = {'time_extracted':extracted_time, 'hour_plot': hour_plot,
                        'time_stamp': datetime.now(timezone.utc).now(),
                        DataClass.TEMPERATURE.value:temperature, 
                        DataClass.SKY.value:sky, DataClass.WIND.value:wind, 
                        DataClass.UV.value:uv, DataClass.CD.value:cloud_cover}
        except Exception as e:
            return f'Error parsing hour data: {e}'
        return hour_data
        
    def get_day_data(self):
        area_code = 'ZIXX0004'
        option = ForecastOption.WHOLEDAY
        site = WeatherCom()
        html = site._get_html(option, area_code)
        soup = BeautifulSoup(html, 'html.parser')
        day_data = []
        i=1
        time = day_time_count()

        while i < time:
            try:
                hour_scrape_section = soup.find('details', {'data-testid':f'ExpandedDetailsCard-{i}'})
                extracted_time = hour_scrape_section.find('h2', {'data-testid': 'daypartName'}).text.strip()
                temperature = hour_scrape_section.find('span', {'data-testid': 'TemperatureValue'}).text.strip()
                wind = hour_scrape_section.find('span', {'data-testid': 'Wind'}).text.strip()
                sky = hour_scrape_section.find('span', class_='DetailsSummary--extendedData--eJzhb').text.strip()
                day_data.append({'Time': extracted_time, 
                                'Temperature': temperature, 
                                'Wind': wind, 'Sky': sky})
                i+=1
            except Exception as e:
                return f'Error parsinf summary data: {e}'
        return day_data