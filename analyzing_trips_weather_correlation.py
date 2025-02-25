import matplotlib
matplotlib.use('TkAgg',force=True)
import sys
import json
from data_functions import *
from charts_trips_weather import *
from model_visualization import *

config_file =  sys.argv[1] if len(sys.argv) > 1 else "athena_config.json"

with open(config_file) as file:
    config = json.load(file)

weather_data = config["weather_data"]
taxi_data=config["taxi_data"]
year = '2022'
# Connect to Athena
conn = get_athena_connection(config)
weather_df = get_weather_data(conn, weather_data, year)
taxi_df = get_taxi_data(conn, taxi_data, year)
merged_df = merged_data(weather_df, taxi_df)
merged_df = merged_df.query('trip_distance > 0 and trip_distance < 100')

# Calls of plotting functions to show the relationship between daily trip distances/ trip counts and the minimum temperature.
plot_daily_trips_distance(merged_df, 'tmin')
plot_daily_trips_number(merged_df, 'tmin')



