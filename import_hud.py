import pandas as pd
import altair as alt
import numpy as np
import json
import math
import requests
from requests.structures import CaseInsensitiveDict
import urllib.parse
import geopandas as gpd
from turfpy.measurement import boolean_point_in_polygon
from geojson import Point, Polygon, Feature, MultiPolygon
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import pgeocode
import random

## Import HUD
hud_df = pd.read_excel("Zipcode_2022_2020census.xlsx")

## Clean and transform zip column
def transform_zip_columnt(df):
    
    df['code'] = df['code'].astype(str)
    
    zip_codes = df['code']
    
    updated_codes = []
    
    for zip_code in zip_codes:
        
        if len(zip_code) == 3:
            new_code = "00"+zip_code
            updated_codes.append(new_code)
            
        elif len(zip_code) == 4:
            new_code = "0"+zip_code
            updated_codes.append(new_code)
            
        else:
            updated_codes.append(zip_code)
            
    df['code'] = updated_codes
    
    return df


hud_df = transform_zip_columnt(hud_df)

## Get Lat Lon from Zip
def get_lat_lon_samples_from_zip(zip_code, num_samples=5):
    us_data = pgeocode.Nominatim('us')
    loc = us_data.query_postal_code(zip_code)
    
    # Generate a list of random latitude and longitude samples
    samples = [(loc.latitude + random.uniform(-0.1, 0.1), loc.longitude + random.uniform(-0.1, 0.1)) for _ in range(num_samples)]
    
    return samples

hud_df['LatLonSamples'] = hud_df['code'].progress_apply(lambda x: get_lat_lon_samples_from_zip(x, num_samples=5)) 

# Write to a CSV file
file_path = 'hud_lat_lon_multiple.csv'

hud_df.to_csv(file_path, index=False)