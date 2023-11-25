import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
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

## Need to update this...
city = 'boston'
address = "45 Myrtle St, Boston, MA 02114" # Feel free to change
api_key = "fdc6c7e452234d248e52de75f73957fc" # Do NOT change 


#Get Address geocode and get lat lon
url = get_geocode_url(address, api_key)
lat_long = get_lat_long(url)

def get_geocode_url(address, api_key):
    base_url = "https://api.geoapify.com/v1/geocode/search"
    url_params = {
        "text": address,
        "format": "json",
        "apiKey": api_key
    }
    url_encoded_params = urllib.parse.urlencode(url_params)
    full_url = f"{base_url}?{url_encoded_params}"
    return full_url

def get_lat_long(url):

    headers = CaseInsensitiveDict()
    headers["Accept"] = "application/json"

    resp = requests.get(url, headers=headers)
    response = resp.json()
    lat = response['results'][0]['lat']
    long = response['results'][0]['lon']
    return (lat, long)

address_lat_long = get_lat_long(url)

# Get data for graphs

## Get GeoJSONs
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

def list_files(directory):
    return sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f != '.DS_Store'])

files = list_files('geo-jsons')

zip_city_geo_jsons = list(zip(list_of_cities, files))

# Initialize the variable to None
matching_tuple = None

# Iterate over the list of cities and geojsons
for city_tuple in zip_city_geo_jsons:
    if city_tuple[0] == city:
        matching_tuple = city_tuple
        break  

geo_json_name = matching_tuple[1]
geo_json_path = 'geo-jsons/'+geo_json_name
geo_json = json.load(open(geo_json_path))

## Get HUD Data
read_path = 'agg_choropleth_data/hud_neighborhoods_agg.csv'

agg_neighborhoods_df = pd.read_pickle(read_path)

agg_neighborhoods_df_city = agg_neighborhoods_df[agg_neighborhoods_df.city == city]

## Get City Center Lat Lon
list_of_cities_lat_lon = sorted([('columbus', 39.9612, -82.9988),
                  ('los-angeles', 34.0549, -118.2426),
                  ('new-york-city', 40.7128, -74.0060),
                  ('fort-worth', 32.7555, -97.3308),
                  ('boston',42.3601,-71.0589),
                  ('broward-county', 26.1224, -80.1373),
                  ('chicago', 41.8781, -87.6232),
                  ('austin', 30.2672, -97.7431),
                  ('seattle', 47.6061, -122.3328),
                  ('rochester', 43.1566, -77.6088),
                  ('san-francisco', 37.7749, -122.4194)])

matching_lat_lon = None

for city_lat_lon in list_of_cities_lat_lon:
    if city_lat_lon[0] == city:
        matching_tuple = city_lat_lon
        break  

city_center_lat_lons = matching_tuple
city_center_lat = city_center_lat_lons[1]
city_center_lon = city_center_lat_lons[2]

# Get Graph
poverty_level_max = agg_neighborhoods_df_city['tpoverty'].max()
poverty_level_min = agg_neighborhoods_df_city['tpoverty'].min()

hover_text = address

# Loctation Hover Tet
for _, row in agg_neighborhoods_df_city.iterrows():
    for column in row.index:
        hover_text += f"<br>{column}: {row[column]}"

# Choropleth
fig = px.choropleth_mapbox(agg_neighborhoods_df_city.round(2),
                           geojson=geo_json,
                           locations="neighborhood",
                           featureidkey='properties.neighbourhood',
                           color="tpoverty",
                           color_continuous_scale=px.colors.sequential.Oryel,
                           range_color=(0, 100),
                           mapbox_style="carto-positron",
                           zoom=9, center={"lat": city_center_lat, "lon": city_center_lon},
                           opacity=0.7,
                           hover_name="tpoverty",
                           hover_data=["pct_occupied", "rent_per_month","pct_overhoused","tpoverty","tpct_ownsfd"], 
                           title='Neighborhood Affordability Data'
                          )

fig.update_layout(dragmode=False)

# Property Address Marker
fig.add_trace(go.Scattermapbox(
    lat=[address_lat_long[0]],  
    lon=[address_lat_long[1]],  
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=10,  
        color='green', 
    ),
    text=[hover_text], 
    hoverinfo='text'
))

#Do I need a way to export this?
fig.save('choropleth_map.html')