import pandas as pd
import os
#import seaborn as sns
#import matplotlib.pyplot as plt
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

list_of_cities = sorted(['columbus', 'los-angeles', 'new-york-city', 'fort-worth', 'boston', 'broward-county','chicago','austin','seattle','rochester','san-francisco'])

# recover listints from pickle (create df)
listings_csv_gz = pd.DataFrame()

for city in list_of_cities:
    file_path = os.path.join('data', city, 'listings.csv.gz')
    
    city_df = pd.read_csv(file_path, compression='gzip')
    
    # Add a new column for the city
    city_df['city'] = city
    
    # Append the DataFrame to df_all
    listings_csv_gz = pd.concat([listings_csv_gz, city_df])



listings_csv_gz['last_scraped'] = pd.to_datetime(listings_csv_gz['last_scraped'])
listings_csv_gz['host_since'] = pd.to_datetime(listings_csv_gz['host_since'])

listings_csv_gz['age'] = (listings_csv_gz['last_scraped'] - listings_csv_gz['host_since']).dt.days

# Fix pricing
listings_csv_gz.price = listings_csv_gz.price.replace(',','',regex=True
                                                     ).replace('\$','',regex=True).astype('float')


# Trim obvious columns that are not helpufl for analysis

columns_to_drop =[
                  'id', # possible indicator of time in platoform but had to normalize
                  'listing_url', # additional information other than listing ID
                  'scrape_id', #uncorrelated to price
                  'source', #uncorrelated to price
                  'host_id', #uncorrelated to price
                  'host_url', #uncorrelated to price
                  'host_name', #uncorrelated to price
                  'host_thumbnail_url' ,  #uncorrelated to price
                  'host_picture_url' ,  #uncorrelated to price
                  'neighbourhood',
                  'neighbourhood_group_cleansed',
                  'calendar_updated', #uncorrelated to price
                  'host_neighbourhood', # property already has neighboorhood
                  'calendar_updated', # did not have any values
                  'license', # highly incomplete
                  'host_location',  #uncorrelated to price
                  'picture_url', # most location have pictures
                  'bathrooms', # is blank
                  'has_availability', # we would not know it                             
                  'availability_30' , # we would not know it                                
                  'availability_60' , # we would not know it                       
                  'availability_90' , # we would not know it                        
                  'availability_365', # we would not know it 
                  'calendar_last_scraped', # we would not know it  
                  'file', 
                   'last_scraped', 
                    'host_since', 
                    ]

# Also Trim columns with no reviews to remove inactive properties
#trimmed_listings = listings_csv_gz.drop(columns= columns_to_drop).dropna(subset=['first_review'],how='all', inplace=False)
trimmed_listings = listings_csv_gz.drop(columns=columns_to_drop, errors='ignore').dropna(subset=['first_review'], how='all', inplace=False)


# Drop Additional Columns
columns_to_drop =[
                  'host_response_time', 
                  'host_response_rate', 
                  'host_acceptance_rate', 
                  'last_review', 
                  'first_review',
                  'minimum_minimum_nights',
                  'maximum_minimum_nights', 
                  'minimum_maximum_nights',                      
                  'maximum_maximum_nights',                    
                  'minimum_nights_avg_ntm',                   
                  'maximum_nights_avg_ntm',
    'calculated_host_listings_count',                
    'calculated_host_listings_count_entire_homes' ,  
    'calculated_host_listings_count_private_rooms',
    'calculated_host_listings_count_shared_rooms'
]

# drop rows where bathrooms text, bedrooms or beds are null
trimmed_listings = trimmed_listings.drop(columns= columns_to_drop
                            ).dropna(subset=[
                                                'bathrooms_text',
                                                'bedrooms',
                                                'beds'
], 
                                        how='any', inplace=False)

# create staging pickle to be used as based for Feature engineering.

df = trimmed_listings
    

def replace_ft_with_01(series):
    mapping = {'f': 0, 't': 1}
    return series.map(mapping)

# To be mapped to 0 and 1:
for column_name in ['host_is_superhost', 'host_has_profile_pic','host_identity_verified','instant_bookable' ]:
    df[column_name] = replace_ft_with_01(df[column_name])
    
#---------------------

def extract_numeric_value(series):
    numeric_series = series.apply(lambda x: re.findall(r'\d+\.\d+|\d+', x))
    numeric_series = numeric_series.apply(lambda x: float(x[0]) if len(x) > 0 else 0)
    return numeric_series


# Get integer in text:    
df.bathrooms_text =  extract_numeric_value(df.bathrooms_text)


##---------------------

def get_string_lengths(series):
    series.fillna('-')
    lengths = series.str.len()
    return lengths

# Get lenght of the text field:    
##for field in ['host_about', 'neighborhood_overview', 'name', 'description']:
##    df[field] = get_string_lengths(df[field])
##    df[field] = df[field].fillna(0)
    
##---------------------    

def count_lists(series):
    return series.map(lambda x: 0 if eval(x)== None else len(eval(x)))    


# Count Items in list:
##for field in ['amenities', 'host_verifications']:
##    df[field] = count_lists(df[field])


##---------------------    
#fix missing ratings
for column in ['review_scores_accuracy', 
           'review_scores_cleanliness' ,
           'review_scores_checkin' ,
           'review_scores_communication' ,
           'review_scores_location']:

    df[column] = df[column].fillna(df['review_scores_rating'])

numeric_columns = trimmed_listings.select_dtypes(include=[np.number]).columns
trimmed_listings[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())


trimmed_listings.to_pickle('trimmed_listings_cleased.pkl')