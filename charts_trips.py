from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Creates a scatter plot of trip counts versus dates.
def plot_date_vs_trip_count_scatter(data, date_column='date', count_column='trip_count'):
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.scatterplot(data=data, x=date_column, y=count_column, ax=ax)

    ax.set_title('Scatter Plot of Date vs Trip Count')
    ax.set_xlabel('Date')
    ax.set_ylabel('Trip Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show(block=True)


# Plots monthly average trips against the annual average for each year.
def plot_monthly_trips_vs_annual_average(data, month_column='month', year_column='year', count_column='trip_count'):
    
    monthly_avg = data.groupby([year_column, month_column])[count_column].mean().reset_index()
    monthly_avg.columns = [year_column, month_column, 'monthly_avg']
    monthly_avg['year_month'] = monthly_avg[year_column].apply(str) + '-' + monthly_avg[month_column].apply(str).apply(lambda x : x.zfill(2))
       
    annual_avg = data.groupby(year_column)[count_column].mean().reset_index()
    annual_avg.columns = [year_column, 'annual_avg']
   
    merged_data = pd.merge(monthly_avg, annual_avg, on=year_column)

    plt.figure(figsize=(12, 6))

    for year in merged_data[year_column].unique():
        year_data = merged_data[merged_data[year_column] == year]
        plot = plt.scatter(year_data['year_month'], year_data['monthly_avg'], label=str(year))
        
    plt.plot(merged_data['year_month'], merged_data['annual_avg'], 'k-', linewidth=2, label='Yearly Average')
    plt.title('Monthly Average Trips vs Annual Average')
    plt.xlabel('Month')
    plt.ylabel('Average Number of Trips')
    xticks = np.arange(1,len(merged_data['year_month'])+1,1)
    xlabels = merged_data['year_month']
    plot.axes.set_xticks(xticks,labels=xlabels)
    plt.legend(title='Year')
    plt.tight_layout()
    plt.show(block=True)


#  Visualizes normalized trip counts by year-month. 
def plot_normalized_trips(data, month_column = 'month', year_column = 'year', count_column='trip_count'):

    monthly_avg_trips = data.groupby([year_column, month_column])[count_column].mean().reset_index()
    monthly_avg_trips.columns = [year_column, month_column, 'monthly_avg']

    year_avg_trips = data.groupby(year_column)[count_column].mean().reset_index()
    year_avg_trips.columns = [year_column, 'year_avg']

    merged_data = pd.merge(monthly_avg_trips, year_avg_trips, on=year_column)

    merged_data['normalized_trips'] = merged_data['monthly_avg'] / merged_data['year_avg']

    merged_data['year_month'] = merged_data[year_column].astype(str) + '-' + merged_data[month_column].astype(str).str.zfill(2)
    merged_data = merged_data.sort_values([year_column, month_column])

    plt.figure(figsize=(10, 12))
    sns.set_style("whitegrid")
    sns.lineplot(data=merged_data, x='year_month', y='normalized_trips')
    plt.title('Normalized Trips by Year-Month', fontsize=16)
    plt.xlabel('Year-Month', fontsize=12)
    plt.ylabel('Normalized Trips', fontsize=12)
    plt.axhline(y=1, color='r', linestyle='--', label='Yearly Average')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.legend()
    plt.show(block=True)


# Creates a line plot of average trip count for each month across different years.
def plot_monthly_average_trip_count_line(data, month_column = 'month', year_column = 'year', count_column='trip_count'):

    monthly_avg_trips = data.groupby([year_column, month_column])[count_column].mean().reset_index()
    monthly_avg_trips.columns = [year_column, month_column, 'avg_trips']

    fig, ax = plt.subplots(figsize=(12, 6))
    
    for year in monthly_avg_trips[year_column].unique():
        year_data = monthly_avg_trips[monthly_avg_trips[year_column] == year]
        ax.plot(year_data[month_column], year_data['avg_trips'], marker='o', label=str(year))
    
    ax.set_title('Line Plot of Month vs Average Trip Count by Year')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Trip Count')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend(title='Year')
    plt.tight_layout()
    plt.show(block=True)

#  Calculate the 10th, 50th, and 90th percentiles of a given dataset, respectively.
def pct10(x):
    return np.percentile(x,10)

def pct50(x):
    return np.percentile(x,50)

def pct90(x):
    return np.percentile(x,90)


#  Plots the mean normalized trip count for each month against the yearly average.
def plot_monthly_trips_average_vs_year_average_scatter(data, month_column = 'month', year_column = 'year', count_column='trip_count'):

    monthly_avg_trips = data.groupby([year_column, month_column])[count_column].mean().reset_index()
    monthly_avg_trips.columns = [year_column, month_column, 'monthly_avg']

    year_avg_trips = data.groupby(year_column)[count_column].mean().reset_index()
    year_avg_trips.columns = [year_column, 'year_avg']

    merged_data = pd.merge(monthly_avg_trips, year_avg_trips, on=year_column)

    merged_data['normalized_trips'] = merged_data['monthly_avg'] / merged_data['year_avg']
    detrended_monthly_mean = merged_data.groupby('month').agg({'normalized_trips': [np.mean,pct10,pct50,pct90]}).reset_index()


    narrow_data = detrended_monthly_mean.melt(id_vars=[('month','')], value_vars=[('normalized_trips','mean'),
                                                                             ('normalized_trips','pct10'),
                                                                             ('normalized_trips','pct50'),
                                                                             ('normalized_trips','pct90')
                                                                             ])
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(data=detrended_monthly_mean, x='month', 
                 y=detrended_monthly_mean['normalized_trips']['mean'], 
                 label='Mean', marker='o', linewidth=2, ax=ax)

    # Plot 10th percentile
    sns.lineplot(data=detrended_monthly_mean, x='month', 
                 y=detrended_monthly_mean['normalized_trips']['pct10'], 
                 label='10th Percentile', marker='s', linestyle='--', ax=ax)

    # Plot 50th percentile (median)
    sns.lineplot(data=detrended_monthly_mean, x='month', 
                 y=detrended_monthly_mean['normalized_trips']['pct50'], 
                 label='Median', marker='^', linestyle='-.', ax=ax)

    # Plot 90th percentile
    sns.lineplot(data=detrended_monthly_mean, x='month', 
                 y=detrended_monthly_mean['normalized_trips']['pct90'], 
                 label='90th Percentile', marker='D', linestyle=':', ax=ax)

    ax.set_title('Normalized Monthly Trip Count (Monthly Average / Yearly Average)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Normalized Monthly Trip Count')
    plt.xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    
    plt.legend(title='Statistic')
    plt.tight_layout()
    plt.show(block=True)

# Visualizes the normalized trip counts for each day of the week compared to the yearly average.
def plot_weekly_trips_average_vs_year_average_scatter(data, year_column = 'year', date_column = 'date', count_column='trip_count'):

    data['day_of_week'] = pd.to_datetime(data['date']).dt.day_name()

    year_avg_trips = data.groupby(year_column)[count_column].mean().reset_index()
    year_avg_trips.columns = [year_column, 'year_avg']

    merged_data = pd.merge(data, year_avg_trips, on=year_column)

    merged_data['normalized_trips'] = merged_data[count_column] / merged_data['year_avg']

    
    detrended_daily_mean = merged_data.groupby('day_of_week').agg({'normalized_trips': [np.mean,pct10,pct50,pct90]}).reset_index()
    narrow_data = detrended_daily_mean.melt(id_vars=[('day_of_week','')], value_vars=[('normalized_trips','mean'),
                                                                           ('normalized_trips','pct10'),
                                                                           ('normalized_trips','pct50'),
                                                                           ('normalized_trips','pct90')
                                                                           ])   
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(data=detrended_daily_mean, x='day_of_week', 
                 y=detrended_daily_mean['normalized_trips']['mean'], 
                 label='Mean', marker='o', linewidth=2, ax=ax)

    # Plot 10th percentile
    sns.lineplot(data=detrended_daily_mean, x='day_of_week', 
                 y=detrended_daily_mean['normalized_trips']['pct10'], 
                 label='10th Percentile', marker='s', linestyle='--', ax=ax)

    # Plot 50th percentile (median)
    sns.lineplot(data=detrended_daily_mean, x='day_of_week', 
                 y=detrended_daily_mean['normalized_trips']['pct50'], 
                 label='Median', marker='^', linestyle='-.', ax=ax)

    # Plot 90th percentile
    sns.lineplot(data=detrended_daily_mean, x='day_of_week', 
                 y=detrended_daily_mean['normalized_trips']['pct90'], 
                 label='90th Percentile', marker='D', linestyle=':', ax=ax)

    ax.set_title('Normalized Day of Week Trip Count (Yearly Average)')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Normalized Daily Trip Count')
    plt.xticks(range(7))
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    
    plt.legend(title='Statistic')
    plt.tight_layout()
    plt.show(block=True)