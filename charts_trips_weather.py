from matplotlib import pyplot as plt
import seaborn as sns

param_labels = {
    'month':'Seasons by months',
    'tmax': 'Maximum Temperature (°F)',
    'tmin': 'Minimum Temperature (°F)',
    'prcp': 'Precipitation (inches)',
    'snow': 'Snowfall (inches)',
    'snwd': 'Snow Depth (inches)'
}

# The function creates a scatter plot comparing a specified parameter's (weather) effect on mean and median taxi trip distances 
# in New York City for a given year, including correlation analysis.
def plot_daily_trips_distance(data, x_param, year=2022):
    
    filtered_df = data[(data['tpep_pickup_datetime'].dt.year == year) & 
                       (data['trip_distance'] > 0) & 
                       (data['trip_distance'] < 100)]

    
    distance_metrics = filtered_df.groupby([x_param]).agg(
        mean_distance=('trip_distance', 'mean'),
        median_distance=('trip_distance', 'median')
    ).reset_index()

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=distance_metrics, x=x_param, y='mean_distance', alpha=0.3, color='red', label='Mean')
    sns.scatterplot(data=distance_metrics, x=x_param, y='median_distance', alpha=0.3, color='blue', label='Median')

    plt.title(f'Impact of {param_labels[x_param]} on Trip Distance\n'
              f'Mean and Median for New York City Taxis ({year})',
            fontsize=18, fontweight='bold', color='#444444')
    
    
    plt.xlabel(param_labels[x_param], fontsize=14)
    plt.ylabel('Trip Distance (miles)', fontsize=14)
    plt.legend()

    correlation_mean = distance_metrics[x_param].corr(distance_metrics['mean_distance'])
    correlation_median = distance_metrics[x_param].corr(distance_metrics['median_distance'])

    plt.text(0.05, 0.95, f'Correlation with mean: {correlation_mean:.2f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    plt.text(0.05, 0.9, f'Correlation with median: {correlation_median:.2f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.tight_layout()
    plt.show(block=True)

 # Visualizes the impact of a specified parameter (weather) on daily taxi trip numbers in New York City for a given year
 # using a scatter plot and correlation analysis.
def plot_daily_trips_number(data, x_param, year=2022):
    
    filtered_df = data[(data['tpep_pickup_datetime'].dt.year == year) & 
                       (data['trip_distance'] > 0) & 
                       (data['trip_distance'] < 100)]

    daily_trips_metrics = filtered_df.groupby([x_param,'date']).agg({
        'tpep_pickup_datetime': 'size'
    }).reset_index()

    daily_trips_metrics.columns = [x_param, 'date', 'daily_trips_number']

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=daily_trips_metrics, x=x_param, y='daily_trips_number', alpha=0.3)
    plt.title(f'Impact of {x_param.capitalize()} on Number of daily trips\n'
                f'Number of daily trips for New York City Taxis ({year})',
                fontsize=18, fontweight='bold', color='#444444')
        
    plt.xlabel(param_labels.get(x_param, x_param.capitalize()), fontsize=14)
    plt.ylabel('Number of daily trips', fontsize=14)

    correlation = daily_trips_metrics[x_param].corr(daily_trips_metrics['daily_trips_number'])
        
    plt.text(0.05, 0.95, f'Correlation daily_trips_number: {correlation:.2f}', transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
    plt.tight_layout()
    plt.show(block=True)

 







 
