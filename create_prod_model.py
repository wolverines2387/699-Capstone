import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
import numpy as np
import lightgbm as lgb
from sklearn.dummy import DummyRegressor
import joblib

# Creates the prod model pkl file to call within the dashboard for pricing

df_encoded_selected= pd.read_pickle('encoded_selected.pkl')
 

df_encoded_selected = df_encoded_selected.rename(columns=lambda x: re.sub(r'[^A-Za-z0-9_]', '', x.replace(' ', '_').replace('/', '')))

def run_lgbm_regression(df, estimators_list, learning_rates_list, target_col):

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
                'early_stopping_round': 100
            }

            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            val_data = lgb.Dataset(X_val_scaled, label=y_val)

            # Train 
            model = lgb.train(params, train_data, valid_sets=[val_data])
            #model = lgb.train(params, train_data, num_boost_round=10000, valid_sets=[val_data], early_stopping_round=10)
            
            return model
        
results = run_lgbm_regression(df_encoded_selected, [1000], 
                              [0.1], target_col='price')



# Save the model
joblib.dump(results, 'prod_model.pkl')

