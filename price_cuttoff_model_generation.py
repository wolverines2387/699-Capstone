import pandas as pd
import os
import psycopg2
from sqlalchemy import create_engine, inspect
#import seaborn as sns
import matplotlib.pyplot as plt

import re
import warnings
warnings.filterwarnings('ignore')
import tqdm
import random

import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

### Recover trimmed_listings_cleased file returning the previously consolidated and cleased dataset.
trimmed_listings = pd.read_pickle('trimmed_listings_cleased.pkl')

### Create a function to one hot encode categorical values

def encoded_listings(df):
    

    def one_hot_encode(df, column_name):
        df_encoded = pd.get_dummies(df, columns=[column_name], prefix=[column_name], drop_first=True)
        return df_encoded

    # one hot encode:
    for column_name in ['room_type', 'neighbourhood_cleansed','city', 'property_type' ]:
        df = one_hot_encode(df, column_name)

    return df

#### Create a Grid-Search Function to enable efficient model selection
def model_grid_search(df, selected_models=None, cv=5, test_size=0.2, n_estimators =[50, 100, 200]):
    # Separate features (X) and target variable (y)
    X = df.drop('price', axis=1)
    y = df['price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Perform feature scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the regression models with their parameter grids
    regressors = {
        'Linear Regression': (LinearRegression(), {}),
        'Ridge': (Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
        'Lasso': (Lasso(tol=0.001, max_iter=2000), {'alpha': [0.1, 1.0, 10.0]}),
        'ElasticNet': (ElasticNet(), {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.25, 0.5, 0.75]}),
        'Decision Tree': (DecisionTreeRegressor(), {'max_depth': [None, 5, 10, 20]}),
        'Random Forest': (RandomForestRegressor(), {'n_estimators': n_estimators}),
        'Gradient Boosting': (GradientBoostingRegressor(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}),
        'XGBoost': (XGBRegressor(), {'n_estimators': n_estimators, 'learning_rate': [0.01, 0.1, 1.0]}),
        'LightGBM': (LGBMRegressor(), {'n_estimators': n_estimators, 'learning_rate': [0.01, 0.1, 1.0]}),
        'k-NN': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7, 10, 100]}),
        'Neural Network': (MLPRegressor(), {'hidden_layer_sizes': [(100,), (100, 100), (50, 50, 50)]})
    }

    # train and evaluate each selected regression 
    results = {'RMSE': [], 'R-squared': [], 'Best Parameters': [], 'elapsed_time': []}

    if selected_models is None:
        selected_models = regressors.keys()

    for name, (regressor, param_grid) in regressors.items():
        if name in selected_models:
            start = datetime.datetime.now()
            print('Starting ', name, ' - ', start)
            grid_search = GridSearchCV(regressor, param_grid, scoring='neg_mean_squared_error', cv=cv)
            grid_search.fit(X_train_scaled, y_train)
            best_estimator = grid_search.best_estimator_
            best_params = grid_search.best_params_
            predictions = best_estimator.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            results['RMSE'].append(rmse)
            results['R-squared'].append(r2)
            results['Best Parameters'].append(best_params)
            
            end = datetime.datetime.now()
            elapsed_time = end - start
            results['elapsed_time'].append(elapsed_time)
            
          

    # Create table
    results_df = pd.DataFrame(results, index=selected_models)
    
    return results_df

#### Run a gridsearch against all methods:
model_grid_search(encoded_listings(trimmed_listings), test_size=0.8)

#### Outlier Analysis -  Remove properties above a certain threshold

outlier_dir = 'outlier_analysis'
if not os.path.exists(outlier_dir):
    os.makedirs(outlier_dir)

def remove_outliers (cut_off):
    selected_models = ['Linear Regression','Lasso', 'Ridge', 'ElasticNet', 'LightGBM']
    
    excluded = len(trimmed_listings[trimmed_listings.price>cut_off])

    results = model_grid_search(encoded_listings(trimmed_listings[trimmed_listings.price<cut_off]),   
                    selected_models = selected_models)
    
    results['cut_off'] = cut_off
    results['excluded'] = excluded
    results.to_pickle(outlier_dir +'/results' + str(cut_off)+'.pkl')
    return results

cutoffs = [400, 500, 700, 1000, 1500, 5000]

for cut_off_val in cutoffs:
    remove_outliers (cut_off_val)

df = []

for cut_off in [400, 500, 700, 1000, 1000, 1500, 5000]:
    df.append(pd.read_pickle(outlier_dir +'/results' + str(cut_off)+'.pkl'))
    df_cutoff = pd.concat(df)

df_cutoff.to_pickle(outlier_dir +'/full_results.pkl')