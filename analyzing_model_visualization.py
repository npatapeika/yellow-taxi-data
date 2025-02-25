import pandas as pd
import matplotlib
import sys
import json
matplotlib.use('TkAgg',force=True)
from data_functions import *
from model_visualization import*

config_file =  sys.argv[1] if len(sys.argv) > 1 else "athena_config.json"
ddl_output_file = sys.argv[2] if len(sys.argv) > 2 else "ddl_output.sql"

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

# Prepares monthly distance data using 'tmax' and 'prcp' as features. 
print("Before prepare_data call")
X_train, X_test, y_train, y_test, data_for_feeding, X = prepare_monthly_distance_data(
    merged_df, 
    param_labels, 
    features=['tmax', 'prcp'],
    month_column='month',
    distance_column='trip_distance'
)
print("After prepare_data call")
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Trains three different models: neural network, random forest, and linear regression.
print("Before model training")
nn_model = train_neural_network(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)
lr_model = train_linear_regression(X_train, y_train)
print("After model training")

# Plots the performance of these models, comparing their predictions against actual values.
print("Before plot_model_performance call")
plot_model_performance(X_train, X_test, y_train, y_test, nn_model, rf_model, lr_model)
print("After plot_model_performance call")




