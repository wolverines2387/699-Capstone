import numpy as np
import pandas as pd
from scipy import stats
import altair as alt

city = 'boston'
city_neighborhood = 'East Boston'
price_cutoff_upper = 700
price_cutoff_lower = 40

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

list_of_listings = sorted(['data/columbus',
                  'data/los-angeles',
                  'data/new-york-city',
                  'data/fort-worth', 
                  'data/boston',
                  'data/broward-county',
                  'data/chicago',
                  'data/austin',
                  'data/seattle', 
                  'data/rochester',
                  'data/san-francisco'])

zip_city_listings = list(zip(list_of_cities, list_of_listings))

# Initialize the variable to None
matching_tuple = None

# Iterate over the list of tuples
for city_tuple in zip_city_listings:
    # If the first element of the tuple matches the city variable
    if city_tuple[0] == city:
        # Assign the matching tuple to the variable
        matching_tuple = city_tuple
        break  # Exit the loop

listing_directory = matching_tuple[1]

def list_files(directory):
    return sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f != '.DS_Store'])

files = list_files(listing_directory)

listing_df = pd.read_pickle(listing_directory+'/'+files)

def transform_listing_data(df):
    columns_to_keep = ['neighborhood_overview', 'neighbourhood', 'neighbourhood_cleansed',
                   'neighbourhood_group_cleansed', 'bedrooms', 'price', 'bathrooms_text', 'has_availability']

    df = df[columns_to_keep]
    df['price'] = (df['price'].replace( '[\$,)]','', regex=True ).replace( '[(]','-',   regex=True ).astype(float))
    df["bathroom_qty"] = df["bathrooms_text"].str.split(" ", expand=True)[0]
    
    return df

listing_trimmed_df = transform_listing_data(listing_df)

def filter_neighborhood(df, neighborhood, price_cutoff_upper, price_cutoff_lower):
    
    rslt_df = df[df['neighbourhood_cleansed'].isin(neighborhood)] 
    rslt_df = rslt_df[['neighbourhood_cleansed', 'bedrooms', 'price', 'bathroom_qty', 'has_availability']]
    rslt_df = rslt_df[rslt_df['price'] <= price_cutoff_upper]
    rslt_df = rslt_df[rslt_df['price'] >= price_cutoff_lower]  
    rslt_df = rslt_df.dropna().reset_index(drop=True)
    
    return rslt_df

city_neighborhood_df = filter_neighborhood(chi_listing_trimmed_df, [city_neighborhood],price_cutoff_upper, price_cutoff_lower)

# Model results example
data = {'price': [175, 250, 400, 425, 550, 600], 'bedrooms': [1, 2, 3, 4, 5, 6], 
        'neighbourhood_cleansed': ['Loop', 'Loop', 'Loop', 'Loop', 'Loop', 'Loop'],
        'instant_bookable': ['t','t','t','t','t','t']
       }
pretend_model_range = pd.DataFrame.from_dict(data)
pretend_model_results = pretend_model_range.iloc[3].to_frame().T

#Get graph
brush = alt.selection_interval()
base = alt.Chart(city_neighborhood_df).add_params(brush)

# Configure the points
points = base.mark_point().encode(
    x=alt.X('bedrooms', title='Bedrooms'),
    y=alt.Y('price', title='Price'),
    color=alt.condition(brush, 'has_availability', alt.value('grey')),
    size=alt.Size('bathroom_qty')
).properties(
        title='Model Results to Neighborhood Prices',
        width=600,  # Double the width
        height=600
)

# Configure the ticks
tick_axis = alt.Axis(labels=False, domain=False, ticks=False)

x_ticks = base.mark_tick().encode(
    alt.X('bedrooms', axis=tick_axis),
    alt.Y('has_availability', title='', axis=tick_axis),
    color=alt.condition(brush, 'has_availability', alt.value('lightgrey'))
).properties(
        width=600
)

y_ticks = base.mark_tick().encode(
    alt.X('has_availability', title='', axis=tick_axis),
    alt.Y('price', axis=tick_axis),
    color=alt.condition(brush, 'has_availability', alt.value('lightgrey'))
).properties(
        height=600
)

model_results_line = alt.Chart(pretend_model_range).mark_line(color='red').encode(
    x='bedrooms:Q',
    y='price'
)

your_property_marker = alt.Chart(pretend_model_results).mark_point(color='green', shape="cross", filled=True, size=500).encode(
    x='bedrooms:Q',
    y='price',
    tooltip=['price', 'bedrooms', 'neighbourhood_cleansed', 'instant_bookable']
)

points_layered = (points + model_results_line + your_property_marker)

final_chart = y_ticks | points_layered & x_ticks

##push to json....Do we need this?
chart_json = final_chart.to_json()