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

# Import Lat Lon Hud DF
file_path = 'hud_lat_lon_multiple.pkl'
hud_df = pd.read_pickle(file_path)


# Function to convert zip codes to neighborhoods
def get_df_all_neighborhood(geojson_file, dataframe):
    
    
    gdf = gpd.read_file(geojson_file)

    neighborhoods = []

    for coordinates_list in  tqdm(dataframe['LatLonSamples']):
        coordinate_list = []

        for lat, lon in coordinates_list:
            point = Point(lon, lat)  # Note the order of longitude and latitude here
            found = False

            for index, row in gdf.iterrows():
                neighborhood_name = row['neighbourhood']
                neighborhood_geometry = row['geometry']

                if neighborhood_geometry.contains(point):
                    coordinate_list.append(neighborhood_name)
                    found = True
                    break

            if not found:
                coordinate_list.append(None)

        neighborhoods.append(coordinate_list)

    dataframe['neighborhood'] = neighborhoods
    
    return dataframe

# Get list of geojsons
def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Usage
files = list_files('geo-jsons')

list_of_cities = ['columbus',
                  'los-angeles',
                  'new-york-city',
                  'fort-worth', 
                  'boston',
                  'broward-county',
                  'chicago',
                  'austin',
                  'seattle', 
                  'rochester',
                  'san-francisco']

# Create a new folder for choropleth dataframes
new_dir = 'choropleth_data'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# Write dataframes with neighborhoods to folders
for file, file_name in tqdm(zip(files,list_of_cities)):
    hud_neighborhoods_df = get_df_all_neighborhood('Chicago_Data/geo-jsons/'+file, hud_df)
    file_path = os.path.join(new_dir, file_name + 'neighborhood_df.csv')

    # Write the DataFrame to a CSV file
    hud_neighborhoods_df.to_csv(file_path, index=False)