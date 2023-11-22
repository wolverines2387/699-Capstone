import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

import requests
from bs4 import BeautifulSoup
import os
import pickle

# Get URL
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

list_of_files = ['listings.csv.gz']

def GetData(url, list_of_cities, list_of_files):


    ## This script gets a list of cities and file names and downloads it from the page only when the data changes
    processed_links = []

    # Check if the pickle file with lisnks exist
    if not os.path.isfile('processed_links.pkl'):
        # If the file does not exist, create an empty list and save it to the file
        with open('processed_links.pkl', 'wb') as file:
            pickle.dump(processed_links, file)
    else:
        with open('processed_links.pkl', 'rb') as pickle_file:
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
    with open('processed_links.pkl', 'wb') as file:
            pickle.dump(processed_links, file)

    for city in list_of_cities:
        folder_name = 'data/'+ city
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

# The data has been saved by by city and then by snapshot. All folder follow the same name convention

#raise NotImplementedError("This cell consolidates all files into 1. Only run once.")


folder_path = "data"

cities = list_of_cities
snapshots = ['2023MAR', '2022DEC', '2023SEP', '2023JUN']
list_of_files=['listings.csv.gz']

df_dic = {}
geojson_list =[]
for city in cities:
    #for snapshot in snapshots:
        # List all files in the folder
    full_path = folder_path +'/'+ city  
    files = os.listdir(full_path  )
    for file in files:
        if file in list_of_files:
            df=pd.DataFrame()
            if file.endswith(('.gz')):
                df = pd.read_csv(full_path + '/' + file , compression='gzip')
            elif file.endswith(('.geojson')):
                with open(os.path.join(full_path, file)) as f:
                    data = json.load(f)
                    geojson_list.append((city,snapshot,file, data))
            elif file.endswith(('.csv')):
                df = pd.read_csv(full_path + '/' + file )
            df['city'] = city
            df['file']=file
            #df ['snapshot'] = snapshot
            if file.endswith(('.gz', '.csv')):
                if file.replace(".",'_') in df_dic:
                    df_dic[file.replace(".",'_')]= pd.concat([df_dic[file.replace(".",'_')],df]) \
                                                     .drop_duplicates()
                else: 
                    df_dic[file.replace(".",'_')]=df
                print(full_path + '/' + file)


# Saving it to pickle as an intermediary step

for key in df_dic:
    df_dic[key].to_pickle(key + ".pkl")
    print (key, 'Save as pickle')
    
# File 'listings_csv_gz.pkl' is saved to disk
# This file contains the raw
