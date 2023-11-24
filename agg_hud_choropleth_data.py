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

# Create a new folderfor data
new_dir = 'agg_choropleth_data'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# Get choropleth_data
def list_files(directory):
    return sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f != '.DS_Store'])

files = list_files('choropleth_data')


## Check order of this:
list_of_cities = sorted(['columbus',
                  'los-angeles',
                  'new-york-city',
                  'fort-worth', 
                  'boston',
                  'broward-county',
                  'chicago',
                  'austin',
                  'seattle', 
                  'rochester',
                  'san-francisco'])

exploded_df_list = []

for file, city in zip(files,list_of_cities):
    hud_neighborhoods_df = pd.read_pickle('Cchoropleth_data/'+file)

    hud_neighborhoods_df_exploded = hud_neighborhoods_df.explode('neighborhood')
    hud_neighborhoods_df_exploded = hud_neighborhoods_df_exploded.reset_index(drop=True)
    hud_neighborhoods_df_exploded = hud_neighborhoods_df_exploded[hud_neighborhoods_df_exploded['neighborhood'].notna()]
    
    
    hud_neighborhoods_df_agg = hud_neighborhoods_df_exploded.groupby('neighborhood').agg({
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
    
    hud_neighborhoods_df_agg['city'] = city
    
    exploded_df_list.append(hud_neighborhoods_df_agg)
    
agg_neighborhoods_df = pd.concat(exploded_df_list)

write_file_path = os.path.join(new_dir, 'agg_neighborhood_df.csv')

    # Write the DataFrame to a CSV file
agg_neighborhoods_df.to_pickle(write_file_path)
   