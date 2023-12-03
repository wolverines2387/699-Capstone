import pandas as pd
import numpy as np
import os
import json
from shapely.geometry import shape

directory = 'geojson_data'
cities = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
pin_point_df = pd.DataFrame()

for city in cities:
    geo_json_file_path = city_file_path = os.path.join('geojson_data', city,'neighbourhoods.geojson')
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