import pandas as pd
import numpy as np
import json
import math
import requests
from requests.structures import CaseInsensitiveDict
import json
import urllib.parse
import geopandas as gpd
from turfpy.measurement import boolean_point_in_polygon
from geojson import Point, Polygon, Feature, MultiPolygon
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import pgeocode
import random
import os

# This file imports the choropleth data generated in the "convert_hud_zip_to_neighborhood.py" script which takes the zip code from the hud data and correlates it with
# a neighborhood and aggregates the data from the neihborhood lines.

# Create a new folder for data
new_dir = 'agg_choropleth_data'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# Get choropleth_data
def list_files(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    city_file_pairs = [(f, f.split('_neighborhood_df.pkl')[0]) for f in files]
    return city_file_pairs

files = list_files('choropleth_data')

# Define agg columns for the new export data frame
agg_neighborhoods_df = pd.DataFrame(columns=['neighborhood','city','total_units','pct_occupied',
                                             'number_reported','pct_reported','months_since_report',
                                             'pct_movein', 'people_per_unit','people_total','rent_per_month',
                                             'spending_per_month', 'hh_income','person_income', 'pct_lt5k',
                                             'pct_5k_lt10k', 'pct_10k_lt15k','pct_15k_lt20k','pct_ge20k',
                                             'pct_wage_major','pct_welfare_major','pct_other_major',
                                             'pct_median','pct_lt50_median','pct_lt30_median',
                                             'pct_2adults', 'pct_1adult','pct_female_head',
                                             'pct_female_head_child','pct_disabled_lt62','pct_disabled_ge62',
                                             'pct_disabled_all','pct_lt24_head','pct_age25_50','pct_age51_61',
                                             'pct_age62plus','pct_age85plus','pct_minority','pct_black_nonhsp',
                                             'pct_native_american_nonhsp', 'pct_asian_pacific_nonhsp',
                                             'pct_white_nothsp', 'pct_black_hsp','pct_wht_hsp','pct_oth_hsp',
                                             'pct_hispanic','pct_multi','months_waiting', 'months_from_movein',
                                             'pct_utility_allow','ave_util_allow','pct_bed1','pct_bed2', 'pct_bed3', 
                                             'pct_overhoused','tpoverty', 'tminority','tpct_ownsfd'])

# aggregate neighborhood files and introduce a city variable
for file, city in files:
    hud_neighborhoods_df = pd.read_pickle('choropleth_data/'+file)
    hud_neighborhoods_df['city'] = city

    hud_neighborhoods_df_exploded = hud_neighborhoods_df.explode('neighborhood')
    hud_neighborhoods_df_exploded = hud_neighborhoods_df_exploded[pd.notnull(hud_neighborhoods_df_exploded['neighborhood'])]
    hud_neighborhoods_df_exploded = hud_neighborhoods_df_exploded.reset_index(drop=True)
    hud_neighborhoods_df_exploded = hud_neighborhoods_df_exploded[hud_neighborhoods_df_exploded['neighborhood'].notna()]
    
    
    hud_neighborhoods_df_agg = hud_neighborhoods_df_exploded.groupby(['neighborhood','city']).agg({
        'total_units': 'sum',
        'pct_occupied': 'mean',
        'number_reported': 'sum',
        'pct_reported': 'mean',
        'months_since_report': 'mean',
        'pct_movein': 'mean',
        'people_per_unit': 'mean',
        'people_total' : 'sum',
        'rent_per_month' : 'mean',
        'spending_per_month' : 'mean', 
        'hh_income' : 'mean',
        'person_income' : 'mean', 
        'pct_lt5k' : 'mean',
        'pct_5k_lt10k' :'mean',
        'pct_10k_lt15k' : 'mean',
        'pct_15k_lt20k' : 'mean',
        'pct_ge20k' : 'mean',
        'pct_wage_major' : 'mean',
        'pct_welfare_major' : 'mean',
        'pct_other_major' : 'mean',
        'pct_median' : 'mean', 
        'pct_lt50_median' : 'mean',
        'pct_lt30_median' :'mean',
        'pct_2adults' : 'mean',
        'pct_1adult' : 'mean', 
        'pct_female_head' : 'mean',
        'pct_female_head_child' : 'mean',
        'pct_disabled_lt62' : 'mean',
        'pct_disabled_ge62' : 'mean',
        'pct_disabled_all' : 'mean',
        'pct_lt24_head'  : 'mean',
        'pct_age25_50' : 'mean', 
        'pct_age51_61'  : 'mean',
        'pct_age62plus'  : 'mean',
        'pct_age85plus'  : 'mean',
        'pct_minority'  : 'mean',
        'pct_black_nonhsp'  : 'mean',
        'pct_native_american_nonhsp'  : 'mean',
        'pct_asian_pacific_nonhsp'  : 'mean',
        'pct_white_nothsp'  : 'mean', 
        'pct_black_hsp'  : 'mean',
        'pct_wht_hsp': 'mean',
        'pct_oth_hsp': 'mean',
        'pct_hispanic': 'mean',
        'pct_multi' : 'mean',
        'months_waiting'  : 'mean', 
        'months_from_movein'  : 'mean',
        'pct_utility_allow'  : 'mean',
        'ave_util_allow'  : 'mean',
        'pct_bed1'  : 'mean',
        'pct_bed2'  : 'mean', 
        'pct_bed3'  : 'mean', 
        'pct_overhoused'  : 'mean',
        'tpoverty'  : 'mean', 
        'tminority'  : 'mean',
        'tpct_ownsfd'  : 'mean'
    }).reset_index()
    
    agg_neighborhoods_df = pd.concat([agg_neighborhoods_df, hud_neighborhoods_df_agg], ignore_index=True)
    

write_file_path = os.path.join(new_dir, 'agg_neighborhood_df.pkl')

    # Write to a pickle file
agg_neighborhoods_df.to_pickle(write_file_path)