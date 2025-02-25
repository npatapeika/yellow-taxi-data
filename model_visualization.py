from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
import seaborn as sns
from data_functions import *

# A dictionary mapping feature names to their descriptive labels.
param_labels = {
    'month': 'Seasons by months',
    'tmax': 'Maximum Temperature (°F)',
    'tmin': 'Minimum Temperature (°F)',
    'prcp': 'Precipitation (inches)',
    'snow': 'Snowfall (inches)',
    'snwd': 'Snow Depth (inches)',
    'day_of_week': 'Day of Week',
    'weekend': 'Weekend',
    'weekday': 'Weekday'
}

# Prepares daily trip data by grouping and aggregating the merged dataframe, 
# splits it into training and testing sets.
def prepare_daily_trip_data(merged_df, param_labels, features=['tmax', 'month']):
    print("Starting prepare_data function")
    print(f"Shape of merged_df: {merged_df.shape}")
    print(f"Columns in merged_df: {merged_df.columns}")
    print(f"Features: {features}")

    # Ensure all requested features are in param_labels
    for feature in features:
        if feature not in param_labels:
            raise ValueError(f"Feature '{feature}' is not in param_labels dictionary")

    # Create a list with 'date' and all requested features
    groupby_columns = ['date'] + features
    print(f"Groupby columns: {groupby_columns}")

    try:
        data_for_feeding = merged_df.groupby(groupby_columns).agg({
            'tpep_pickup_datetime': 'size'
        }).reset_index()
        print("Groupby operation successful")
    except Exception as e:
        print(f"Error in groupby operation: {e}")
        raise

    column_names = groupby_columns + ['daily_trips_number']
    data_for_feeding.columns = column_names

    X = data_for_feeding[features]
    y = data_for_feeding['daily_trips_number']
    
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Data split successful")
    return X_train, X_test, y_train, y_test

# Prepares monthly distance data, including additional features like day of week and weekend indicators,
# splits it into training and testing sets.
def prepare_monthly_distance_data(merged_df, param_labels, features, month_column = 'month', date_column = 'date', distance_column = 'trip_distance'):
   
    print("Starting prepare_monthly_distance_data function")
    print(f"Shape of merged_df: {merged_df.shape}")
    print(f"Columns in merged_df: {merged_df.columns}")
    print(f"Features: {features}")
    print(f"Month column: {month_column}")
    print(f"Distance column: {distance_column}")

       # Ensure all requested features are in param_labels
    for feature in features:
        if feature not in param_labels:
            raise ValueError(f"Feature '{feature}' is not in param_labels dictionary")

    # Create 'day_of_week', 'weekend', and 'weekday' columns
    merged_df['day_of_week'] = pd.to_datetime(merged_df[date_column]).dt.day_name()
    merged_df['weekend'] = merged_df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    merged_df['weekday'] = 1 - merged_df['weekend']

    features.extend(['weekend', 'weekday', month_column])

    groupby_columns = features
    print(f"Groupby columns: {groupby_columns}")

    try:
        data_for_feeding = merged_df.groupby(groupby_columns).agg({
            distance_column: 'mean'
        }).reset_index()
        print("Groupby operation successful")
    except Exception as e:
        print(f"Error in groupby operation: {e}")
        raise

    column_names = groupby_columns + ['average_trip_distance']
    data_for_feeding.columns = column_names

    X = data_for_feeding[features]
    y = data_for_feeding['average_trip_distance']
    
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Data split successful")
    return X_train, X_test, y_train, y_test, data_for_feeding, X
    
# Creates and trains a neural network model for regression.
def train_neural_network(X_train, y_train):
    print("Starting Neural Network training")
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_data=(X_val, y_val))
    print("Neural Network training complete")
    return model

# Creates and trains a random forest regression model.
def train_random_forest(X_train, y_train):
    print("Starting Random Forest training")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Random Forest training complete")
    return model

# Creates and trains a linear regression model.
def train_linear_regression(X_train, y_train):
    print("Starting Linear Regression training")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression training complete")
    return model

# Calculates and prints performance metrics (MSE and R2) for a given model.
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MSE: {mse:.2f}, R2: {r2:.2f}")
    return mse, r2

# Visualizes the performance of all three models (neural network, random forest, and linear regression) 
# using scatter plots, bar charts for MSE and R2 scores, and a feature importance plot for the random forest model.
def plot_model_performance(X_train, X_test, y_train, y_test, nn_model, rf_model, lr_model):

    # Predictions on test data
    nn_y_pred = nn_model.predict(X_test).flatten()
    rf_y_pred = rf_model.predict(X_test)
    lr_y_pred = lr_model.predict(X_test)

    # Evaluate all models
    print("Model Evaluations:")
    nn_mse, nn_r2 = evaluate_model(y_test, nn_y_pred, "Neural Network")
    rf_mse, rf_r2 = evaluate_model(y_test, rf_y_pred, "Random Forest")
    lr_mse, lr_r2 = evaluate_model(y_test, lr_y_pred, "Linear Regression")

    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, nn_y_pred, alpha=0.5, label='Neural Network')
    plt.scatter(y_test, rf_y_pred, alpha=0.5, label='Random Forest')
    plt.scatter(y_test, lr_y_pred, alpha=0.5, label='Linear Regression')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values for Different Models')
    plt.legend()
    plt.show(block=False)

    # Compare model performances
    models = ['Neural Network', 'Random Forest', 'Linear Regression']
    mse_scores = [nn_mse, rf_mse, lr_mse]
    r2_scores = [nn_r2, rf_r2, lr_r2]

    plt.figure(figsize=(10, 6))
    plt.bar(models, mse_scores)
    plt.title('Mean Squared Error Comparison')
    plt.ylabel('MSE')
    plt.show(block=False)

    plt.figure(figsize=(10, 6))
    plt.bar(models, r2_scores)
    plt.title('R-squared Comparison')
    plt.ylabel('R-squared')
    plt.show(block=False)

    # Plotting feature importance for Random Forest model
    if isinstance(rf_model, RandomForestRegressor):
        feature_importance = rf_model.feature_importances_
        feature_names = X_train.columns
        plt.figure(figsize=(8, 6))
        sns.barplot(x=feature_importance, y=feature_names)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance (Random Forest)')
        plt.show(block=True)




