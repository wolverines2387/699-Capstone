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

### Creates visual analysis for modesl for r2 results

#### Recover trimmed_listings_cleased file returning the previously consolidated and cleased dataset
combined_results = pd.read_pickle('cv_and_test_size_gridsearch.pkl')

models = list(combined_results.index.unique())

# Removing linear regression as it has large negative values that dominate the plot
models.remove('Linear Regression')

cvs = combined_results['cv'].unique()  # Get unique CV values

image_names = ['r2_vs_test_size_cv_2.pdf', 'r2_vs_test_size_cv_5.pdf', 
               'r2_vs_test_size_cv_10.pdf', 'r2_vs_test_size_cv_20.pdf']

model_analysis_dir = 'outlier_analysis'
if not os.path.exists(model_analysis_dir):
    os.makedirs(model_analysis_dir)

# Iterate over each CV value and create a separate plot
for cv, image_name in zip(cvs,image_names):
    # Set up the plot for each CV value
    plt.figure(figsize=(10, 6))
    plt.xlabel('Test Size')
    plt.ylabel('R-squared')
    plt.title(f'R-squared vs Test Size (CV={cv})')

    # Plot a line for each model for the current CV value
    for model in models:
        data = combined_results[(combined_results.index == model) & (combined_results['cv'] == cv)]
        plt.plot(data['test_size'], data['R-squared'], marker='o', label=model)

    # Add legends and grid
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # Show the plot for the current CV value
    plt.savefig(model_analysis_dir+'/'+image_name)