import pandas as pd
import os
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

import requests
from bs4 import BeautifulSoup
import os
import pickle

# Get URL for Inside Airbnb
url = 'http://insideairbnb.com/get-the-data'

# List of cities and files
list_of_cities = ['boston',
                  'broward-county',
                  'chicago',
                  'new-york-city',
                  'seattle',
                  'los-angeles',
                  'austin',
                  'fort-worth',
                  'columbus',
                  'san-francisco',
                  'rochester']

list_of_files = ['neighbourhoods.geojson']

def GetData(url, list_of_cities, list_of_files):


    ## This script gets a list of cities and file names and downloads it from the page only when the data changes
    processed_links = []

    # Check if the pickle file with lisnks exist
    if not os.path.isfile('processed_geojson_links.pkl'):
        # If the file does not exist, create an empty list and save it to the file
        with open('processed_geojson_links.pkl', 'wb') as file:
            pickle.dump(processed_links, file)
    else:
        with open('processed_geojson_links.pkl', 'rb') as pickle_file:
            processed_links = pickle.load(pickle_file)

    # Send get request
    response = requests.get(url)
    response.raise_for_status()


    # Parse the HTML content of the page
    soup = BeautifulSoup(response.content, 'html.parser')

    # Get Page Links
    links = [a['href'] for a in soup.find_all('a', href=True) if a.text]

    # Filter only links that end with desired file names
    filtered_links =[]

    for link in links:
        # Download files if the link ends with any of the file names in your list
        if any(city.lower() in link.lower() for city in list_of_cities):
            if any(link.endswith(file_name) for file_name in list_of_files):
                filtered_links.append(link)


    # Get not seen links for download          
    new_links = [link for link in filtered_links if link not in processed_links]

    processed_links = filtered_links
    with open('processed_geojson_links.pkl', 'wb') as file:
            pickle.dump(processed_links, file)

    for city in list_of_cities:
        folder_name = 'geojson_data/'+ city
        if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

        for link in new_links:
            if city.lower() in link.lower():
                response = requests.get(link)
                response.raise_for_status()

                filename = os.path.join(folder_name, link.split('/')[-1])
                with open(filename, 'wb') as file:
                    file.write(response.content)
                print(f'Downloaded {filename} to {folder_name}')
               
    return

GetData(url, list_of_cities, list_of_files)