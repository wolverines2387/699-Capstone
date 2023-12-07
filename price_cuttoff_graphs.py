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

# Generates graphs for cutoff analysis visual

## Get Cuttoff Models from outlier_directory
df_cutoff = pd.read_pickle('outlier_analysis' +'/full_results.pkl')

# Get unique model names
models = list(df_cutoff.index.unique())

# removing linear regression as it has large negative values that dominate the plot
models.remove('Linear Regression') 

## R2 vs Cutoff
# Set up the plot
plt.figure(figsize=(15, 6))
plt.xlabel('Cutoff')
plt.ylabel('R-squared')
plt.title('R-squared vs Cutoff')

# Plot each model as a line
for model in models:
    data = df_cutoff[df_cutoff.index == model]
    plt.plot(data['cut_off'], data['R-squared'],marker='o', label=model)

# Add legend and show plot
plt.legend()
plt.grid(True)
plt.savefig('r_squared_vs_cutoff.pdf')

## RMSE vs Cutoff
# Set up the plot
plt.figure(figsize=(15, 6))
plt.xlabel('Cutoff')
plt.ylabel('RMSE')
plt.title('RMSE vs Cutoff')

# Plot each model as a line
for model in models:
    data = df_cutoff[df_cutoff.index == model]
    plt.plot(data['cut_off'], data['RMSE'],marker='o', label=model)

# Add legend and show plot
plt.legend()
plt.grid(True)
plt.savefig('rmse_vs_cutoff.pdf')