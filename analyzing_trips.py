import sys
import json
import matplotlib
matplotlib.use('TkAgg',force=True)
from data_functions import *
from charts_trips import *
from model_visualization import *

config_file =  sys.argv[1] if len(sys.argv) > 1 else "athena_config.json"

with open(config_file) as file:
    config = json.load(file)

weather_data = config["weather_data"]
taxi_data=config["taxi_data"]
year = '2020'

# Connect to Athena
conn = get_athena_connection(config)
weather_df = get_weather_data(conn, weather_data, year)

# Calls various plotting functions to visualize different aspects of the taxi trip data, including scatter plots, line plots, and normalized trip counts.

reduced_taxi_df_2020_included = get_reduced_taxi_data_2020_included(conn, taxi_data)
reduced_taxi_df_2020_excluded = get_reduced_by_day_taxi_data_2020_excluded(conn, taxi_data)

plot_date_vs_trip_count_scatter(reduced_taxi_df_2020_included, date_column='date', count_column='trip_count')

plot_monthly_trips_vs_annual_average(reduced_taxi_df_2020_included, month_column='month', year_column='year', count_column='trip_count')

plot_normalized_trips(reduced_taxi_df_2020_included, month_column = 'month', year_column = 'year', count_column='trip_count')

plot_monthly_average_trip_count_line(reduced_taxi_df_2020_included, year_column='year', month_column='month', count_column='trip_count')

plot_monthly_trips_average_vs_year_average_scatter(reduced_taxi_df_2020_excluded, month_column = 'month', year_column = 'year', count_column='trip_count')

plot_weekly_trips_average_vs_year_average_scatter(reduced_taxi_df_2020_excluded, year_column = 'year', date_column = 'date', count_column='trip_count')






