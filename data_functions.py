import pandas as pd
from pyathena import connect

# This function establishes and returns a connection to Amazon Athena using the provided configuration.
def get_athena_connection(config):
    s3_bucket = config["s3_bucket"]
    s3_aws_key_id = config["s3_aws_key_id"]
    s3_aws_secret_key = config["s3_aws_secret_key"]
    aws_region = config["aws_region"]
    athena_results_bucket = config["athena_results_bucket"]

    # Connect to Athena
    conn = connect(s3_staging_dir = athena_results_bucket,
                region_name = aws_region,
                aws_access_key_id = s3_aws_key_id ,
                aws_secret_access_key = s3_aws_secret_key)
    return conn

#  Retrieves weather data for a specific year from the given table using an Athena connection.
def get_weather_data(conn, weather_table, year):
    weather_query = f'SELECT * FROM {weather_table} WHERE YEAR(date) = {year}'
    weather_df = pd.read_sql_query(weather_query, conn)
    return weather_df

# Fetches taxi trip data for a specific year with optional distance filters.
def get_taxi_data(conn, taxi_table, year, min_distance=0, max_distance=100):
    taxi_query = f"""
    SELECT DATE(tpep_pickup_datetime) AS date, *
    FROM {taxi_table}
    WHERE YEAR(tpep_pickup_datetime) = {year}
    AND group_number BETWEEN 1 AND 10
    AND trip_distance > {min_distance} 
    AND trip_distance < {max_distance}
    """
    taxi_df = pd.read_sql_query(taxi_query, conn)
    return taxi_df

#  Retrieves aggregated taxi trip counts by date, month, and year from 2010 to 2022, including 2020.
def get_reduced_taxi_data_2020_included(conn, taxi_table):
    taxi_query = f"""
    SELECT DATE(tpep_pickup_datetime) AS date, 
    EXTRACT(MONTH FROM tpep_pickup_datetime) AS month,
    EXTRACT(YEAR FROM tpep_pickup_datetime) AS year,
    COUNT(*) AS trip_count                                                                      
    FROM {taxi_table}
    WHERE EXTRACT(YEAR FROM tpep_pickup_datetime) BETWEEN 2010 AND 2022 
    GROUP BY 
    1, 2, 3
    ORDER BY
    3, 2, 1
    """
    reduced_taxi_df = pd.read_sql_query(taxi_query, conn)
    return reduced_taxi_df

#  Retrieves aggregated taxi trip counts by date, month, and year from 2010 to 2022, excluding 2020.

def get_reduced_by_day_taxi_data_2020_excluded(conn, taxi_table):
    taxi_query = f"""
    SELECT DATE(tpep_pickup_datetime) AS date, 
    EXTRACT(MONTH FROM tpep_pickup_datetime) AS month,
    EXTRACT(YEAR FROM tpep_pickup_datetime) AS year,
    COUNT(*) AS trip_count                                                                      
    FROM {taxi_table}
    WHERE EXTRACT(YEAR FROM tpep_pickup_datetime) BETWEEN 2010 AND 2022 
    AND EXTRACT(YEAR FROM tpep_pickup_datetime) != 2020
    GROUP BY 
    1, 2, 3
    ORDER BY
    3, 2, 1
    """
    reduced_taxi_df = pd.read_sql_query(taxi_query, conn)
    return reduced_taxi_df

#  Merges weather and taxi dataframes based on the date column.
def merged_data(weather_df, taxi_df):
    taxi_df['date'] = pd.to_datetime(taxi_df['date'])

    merged_df = pd.merge(weather_df, taxi_df, on='date', how='inner')

    return merged_df






