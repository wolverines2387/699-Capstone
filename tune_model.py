import pandas as pd
import os
import psycopg2
from sqlalchemy import create_engine, inspect
import seaborn as sns
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
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor

#Import encoded models

df_encoded_selected= pd.read_pickle('encoded_selected.pkl')
 

df_encoded_selected = df_encoded_selected.rename(columns=lambda x: re.sub(r'[^A-Za-z0-9_]', '', 
                                                                x.replace(' ', '_').replace('/', '')))

tune_dir = 'model_tuning_analysis'
if not os.path.exists(tune_dir):
    os.makedirs(tune_dir)

def run_lgbm_regression(df, estimators_list, learning_rates_list, target_col):
    # lists to store results
    results = []

    # Split the data 
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Split the data into training, validation]test 
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    # Scale the features 
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    for n_estimators in estimators_list:
        for learning_rate in learning_rates_list:
            
            params = {
                'objective': 'regression',
                'metric': 'mse',
                'learning_rate': learning_rate,
                'n_estimators': n_estimators,
                'verbosity': -1,
                'early_stopping_round': 10
            }

            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            val_data = lgb.Dataset(X_val_scaled, label=y_val)

            # Train 
            model = lgb.train(params, train_data, valid_sets=[val_data])
            #model = lgb.train(params, train_data, num_boost_round=10000, valid_sets=[val_data], early_stopping_round=10)

            # predictions on the training set
            y_train_pred = model.predict(X_train_scaled)

            #predictions on the test set
            y_test_pred = model.predict(X_test_scaled)

            # predictions on the validation set
            y_val_pred = model.predict(X_val_scaled)

            #  metrics for the training set
            r2_train = r2_score(y_train, y_train_pred)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

            # metrics for the test set
            r2_test = r2_score(y_test, y_test_pred)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

            # metrics for the validation set
            r2_val = r2_score(y_val, y_val_pred)
            rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

           
            result = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'R-squared (Train)': r2_train,
                'R-squared (Test)': r2_test,
                'R-squared (Validation)': r2_val,
                'RMSE (Train)': rmse_train,
                'RMSE (Test)': rmse_test,
                'RMSE (Validation)': rmse_val
            }

            # Create a dummy regressor 
            dummy_model = DummyRegressor(strategy='mean')
            dummy_model.fit(X_train_scaled, y_train)

            # predictions using the dummy regressor
            y_train_pred_dummy = dummy_model.predict(X_train_scaled)
            y_test_pred_dummy = dummy_model.predict(X_test_scaled)
            y_val_pred_dummy = dummy_model.predict(X_val_scaled)

            #  metrics for the dummy regressor
            r2_train_dummy = r2_score(y_train, y_train_pred_dummy)
            rmse_train_dummy = np.sqrt(mean_squared_error(y_train, y_train_pred_dummy))
            r2_test_dummy = r2_score(y_test, y_test_pred_dummy)
            rmse_test_dummy = np.sqrt(mean_squared_error(y_test, y_test_pred_dummy))
            r2_val_dummy = r2_score(y_val, y_val_pred_dummy)
            rmse_val_dummy = np.sqrt(mean_squared_error(y_val, y_val_pred_dummy))

            # add dummy regressor metrics to the result dictionary
            result['R-squared (Dummy Train)'] = r2_train_dummy
            result['R-squared (Dummy Test)'] = r2_test_dummy
            result['R-squared (Dummy Validation)'] = r2_val_dummy
            result['RMSE (Dummy Train)'] = rmse_train_dummy
            result['RMSE (Dummy Test)'] = rmse_test_dummy
            result['RMSE (Dummy Validation)'] = rmse_val_dummy

            results.append(result)

    #  DataFrame from the results
    results_df = pd.DataFrame(results)

    return results_df

## Additional Tuning
#During pre-processing we noted that the mode was very sensitive to the number of estimators and the learning rate. We noticed that performance continue to improve as we added additional estimators which raised the concern of overfitting.
#In order to address this concern and test for overfitting we separated the data into train, test and validation sets with a 80-15-5 split and ran a grid manual grid search capturing all results in a dataframe.

df = df_encoded_selected.copy()
results = run_lgbm_regression(df, [ 50, 200, 1000, 2000, 5000, 10000, 15000, 20000], 
                              [0.01,0.1,1], target_col='price')

results.to_pickle(model_tuning_analysis+'/LightGBM_estimators_Grid_Search_1.pkl')

df=results

# dictionary to store the data frames 
learning_rate_dfs = {}


# Group the data by learning rate
grouped = df.groupby('learning_rate')

# series to be plotted
series_to_plot = ['R-squared (Train)', 'R-squared (Test)', 'R-squared (Validation)']


for learning_rate, group in grouped:
    fig, ax = plt.subplots()

    for series in series_to_plot:
        n_estimators = group['n_estimators']
        series_values = group[series]
        ax.plot(n_estimators, series_values, label=series)

    ax.set_title(f'Learning Rate: {learning_rate}')
    ax.set_xlabel('Number of Estimators')
    ax.set_ylabel(series_to_plot[0])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add the current learning rate plot to the dictionary
    learning_rate_dfs[learning_rate] = fig


plt.save(model_tuning_analysis+'/tune_graphs.html')