import pandas as pd
import numpy as np
import os
import json
from shapely.geometry import shape

# File finds neighborhood centers for selected neighborhood views in choropleth

directory = 'geo-jsons'
cities = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
pin_point_df = pd.DataFrame()

def list_files(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    city_file_pairs = [(f, 'new-york-city' if f.split('-neighbourhoods.geojson')[0] == 'new-york' else ('broward-county' if f.split('-neighbourhoods.geojson')[0] == 'broward' else f.split('-neighbourhoods.geojson')[0])) for f in files]
    return city_file_pairs

files = list_files('geo-jsons')

for file, city in files:
    geo_json_file_path = os.path.join('geo-jsons', file)
    geo_json = json.load(open(geo_json_file_path))
    
    neighborhoods = []
    coordinates = []

    
    for feature in geo_json['features']:
        
        neighborhood = feature['properties']['neighbourhood']
        neighborhoods.append(neighborhood)

       
        geom = shape(feature['geometry'])
        point_within = geom.representative_point()
        lon, lat = point_within.x, point_within.y
        coor = lon, lat
        coordinates.append(coor)
    
        
    
    
    city_df = pd.DataFrame({'city': [city]*len(neighborhoods), 'neighborhood': neighborhoods, 'coordinates': coordinates})
    
    
    pin_point_df = pd.concat([pin_point_df, city_df], ignore_index=True)
    
pin_point_df.to_pickle('pin_point_coordinates.pkl')