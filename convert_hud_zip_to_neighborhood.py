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

# This file takes the HUD data and converts the zip code to a set of neighborhoods for analysis alongside the geojson data.

# Import Lat Lon Hud DF from pkl
read_file_path = 'hud_lat_lon_multiple.pkl'
hud_df = pd.read_pickle(read_file_path)


# Function to convert zip codes to neighborhoods
def get_df_all_neighborhood(geojson_file, dataframe):
    gdf = gpd.read_file(geojson_file)

    neighborhoods = []

    for coordinates_list in tqdm(dataframe['LatLonSamples']):
        coordinate_list = []

        for lat, lon in coordinates_list:
            point = Point(lon, lat) 
            found = False

            for index, row in gdf.iterrows():
                neighborhood_name = row['neighbourhood']
                neighborhood_geometry = row['geometry']

                try:
                    if neighborhood_geometry.contains(point):
                        coordinate_list.append(neighborhood_name)
                        found = True
                        break
                except Exception as e:
                    print(f"Error at point {point}: {e}. Inserting np.nan...")
                    coordinate_list.append(np.nan)
                    continue

            if not found:
                coordinate_list.append(None)

        neighborhoods.append(coordinate_list)

    dataframe['neighborhood'] = neighborhoods
    
    return dataframe


# Get list of geojsons
def list_files(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    city_file_pairs = [(f, 'new-york-city' if f.split('-neighbourhoods.geojson')[0] == 'new-york' else ('broward-county' if f.split('-neighbourhoods.geojson')[0] == 'broward' else f.split('-neighbourhoods.geojson')[0])) for f in files]
    return city_file_pairs


files = list_files('geo-jsons')


# Create a new folder for choropleth dataframes
new_dir = 'choropleth_data'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# Write dataframes with neighborhoods to folders
for file, file_name in tqdm(files):
    hud_neighborhoods_df = get_df_all_neighborhood('geo-jsons/'+file, hud_df)
    write_file_path = os.path.join(new_dir, file_name + '_neighborhood_df.pkl')

    # Write to a PKL file
    hud_neighborhoods_df.to_pickle(write_file_path)