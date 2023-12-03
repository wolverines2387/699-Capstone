import json
from datetime import date
from urllib.request import urlopen
import time
import os 
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import joblib
## No longer need this with this version of python
#from pandas.io.json import json_normalize

_ENABLE_PROFILING = False

if _ENABLE_PROFILING:
    import cProfile, pstats, io
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()

today = date.today()

column_df = pd.read_pickle('encoded_selected.pkl')
column_df.drop(columns = ['price'], inplace=True)
column_lst = column_df.columns.tolist()
loaded_model = joblib.load('prod_model.pkl')

def get_neighborhoods(directory):
    #cities = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    cities = ['columbus','los-angeles', 'new-york-city','fort-worth', 'boston', 'broward-county',
     'chicago', 'seattle', 'rochester', 'san-francisco']
    city_dir = {}
    
    for city in cities:
        city_file_path = os.path.join(directory, city,'listings.csv.gz')
        city_df = pd.read_csv(city_file_path)
        neighbourhood_list = city_df.neighbourhood_cleansed.unique().tolist()
        
        city_dir[city] = neighbourhood_list
    
    return city_dir

directory = 'data'
neighborhoods = get_neighborhoods(directory)

def extract_property_type(column_list):
    property_type_list = []

    for string in column_list:
        if string.startswith("property_type_"):
            property_type_list.append(string)
    
    return property_type_list

property_type_columns = extract_property_type(column_lst)

def property_values_for_dropdown(property_list):
    
    type_list = []
    
    for property in property_list:
    
        property_type_string = "_".join(property.split("_")[2:])
    
        type_list.append(property_type_string)
    
    return type_list

def extract_room_type(column_list):
    room_type_list = []

    for string in column_list:
        if string.startswith("room_type_"):
            room_type_list.append(string)
    
    return room_type_list

room_type_columns = extract_room_type(column_lst)

def room_values_for_dropdown(room_list):
    
    type_list = ["N/A"]
    
    for room in room_list:
    
        room_type_string = "".join(room.split("_")[2:])
    
        type_list.append(room_type_string)
    
    return type_list

def transform_listing_data(df):
    columns_to_keep = ['neighborhood_overview', 'neighbourhood', 'neighbourhood_cleansed',
                   'neighbourhood_group_cleansed', 'bedrooms', 'price', 'bathrooms_text', 'instant_bookable']

    df = df[columns_to_keep].copy()
    df.loc[:, 'price'] = (df['price'].replace( '[\$,)]','', regex=True ).replace( '[(]','-',   regex=True ).astype(float))
    df.loc[:, "bathroom_qty"] = df["bathrooms_text"].str.split(" ", expand=True)[0]
    
    return df

def filter_neighborhood(df, neighborhood, price_cutoff_upper, price_cutoff_lower):
    
    rslt_df = df[df['neighbourhood_cleansed'].isin(neighborhood)] 
    rslt_df = rslt_df[['neighbourhood_cleansed', 'bedrooms', 'price', 'bathroom_qty', 'instant_bookable']]
    rslt_df = rslt_df[rslt_df['price'] <= price_cutoff_upper]
    rslt_df = rslt_df[rslt_df['price'] >= price_cutoff_lower]  
    rslt_df = rslt_df.dropna().reset_index(drop=True)
    
    return rslt_df

st.set_page_config(
    page_title="Short-term Rental Pricing Predictor",
    layout='wide',
    initial_sidebar_state='auto',
)

#################################################################
#Insert values into the submit df
def update_property_type_submission(submit_df, selected_property_type, property_type_values):
    header = "property_type_"
    
    un_selected_property_type_values = property_type_values.copy()
    
    un_selected_property_type_values.remove(selected_property_type)
    
    #Insert dummy for selected property type
    #selected_property_type_word_list = selected_property_type.split()
    
    selected_property_column = header+selected_property_type
    
    #selected_property_column = header + '_' + selected_property_type_word
    
    submit_df.loc[0, selected_property_column] = 1
    
    #Insert dummy for unselected property types
    
    for unselected_property_type in un_selected_property_type_values:
        #unselected_property_type_word_list = unselected_property_type.split()
    

        unselected_property_column = header + unselected_property_type

        submit_df.loc[0, unselected_property_column] = 0
        
    return submit_df
        
def update_city_submission(submit_df, sidebar_city):
    header = "city_"
    
    city_list = ['columbus','los-angeles', 'new-york-city','fort-worth', 'boston', 'broward-county',
     'chicago', 'seattle', 'rochester', 'san-francisco']
    
    un_selected_city_values = city_list.copy()
    
    un_selected_city_values.remove(sidebar_city)
    
    #Insert dummy for selected property type
    
    selected_city_column = header + sidebar_city
    
    submit_df.loc[0, selected_city_column] = 1
    
    #Insert dummy for unselected property types
    
    for unselected_city in un_selected_city_values:

        unselected_city_column = header + unselected_city

        submit_df.loc[0, unselected_city_column] = 0
        
    return submit_df

def update_neighborhood_submission(submit_df, sidebar_neighborhood, directory):
    neighborhoods = get_neighborhoods(directory)
    
    header = "neighbourhood_cleansed_"
    
    neighborhood_list = []
    
    city_list = ['columbus','los-angeles', 'new-york-city','fort-worth', 'boston', 'broward-county',
     'chicago','seattle', 'rochester', 'san-francisco']
    
    for city in city_list:
        neighborhood_list = neighborhood_list + neighborhoods[city]
    
    un_selected_neighborhoods = neighborhood_list.copy()
    
    un_selected_neighborhoods.remove(sidebar_neighborhood)
    
    #Insert dummy for selected property type
    
    selected_neighborhood_column = header + sidebar_neighborhood
    
    submit_df.loc[0, selected_neighborhood_column] = 1
    
    #Insert dummy for unselected property types
    
    for un_selected_neighborhood in un_selected_neighborhoods:
        un_selected_neighborhood = str(un_selected_neighborhood)
        unselected_neighborhood_column = header + un_selected_neighborhood

        submit_df.loc[0, unselected_neighborhood_column] = 0
        
    return submit_df.iloc[:, :-11]

def update_room_type_submission(submit_df, selected_room_type, room_type_values):
    header = "room_type_"
    
    if selected_room_type == 'N/A':
        submit_df.loc[0, 'room_type_Hotel room'] = 0
        submit_df.loc[0, 'room_type_Private room'] = 0
    
    else:
    
        un_selected_room_type_values = room_type_values.copy()

        un_selected_room_type_values.remove(selected_room_type)
        un_selected_room_type_values.remove("N/A")

        #Insert dummy for selected property type

        selected_room_column = header + selected_room_type

        submit_df.loc[0, selected_room_column] = int(1)

        #Insert dummy for unselected property types

        for unselected_room_type in un_selected_room_type_values:

            unselected_room_column = header + unselected_room_type

            submit_df.loc[0, unselected_room_column] = int(0)
        
    return submit_df


##################################################################

def model_results_graph(model, dataframe, city, neighborhood, model_value, model_bedrooms, instant_bookable):
    
    model_value = round(model_value,0)
    if instant_bookable == 1:
        instant_bookable = 't'
    elif instant_bookable == 0:
        instant_bookable = 'f'
    
    #Transform and filter listing data
    graph_listing_df = pd.read_csv('data/'+ city +'/listings.csv.gz')#update this for data/city values
    transformed_graph_listing_df = transform_listing_data(graph_listing_df)
    filtered_graph_listing_df = filter_neighborhood(transformed_graph_listing_df, [neighborhood], 1000, 40)
    
    #Get Model Values
    bedroom_count = [1,2,3,4,5,6]
    price_trend = []
    
    for bedroom in bedroom_count:
        dataframe_copy = dataframe.copy()
        dataframe_copy.loc[0, 'bedrooms'] = bedroom
        
        dataframe_copy = dataframe_copy.apply(pd.to_numeric, errors='ignore')
        
        chart_prediction = model.predict(dataframe_copy)
        chart_prediction = chart_prediction[0]
        chart_prediction = np.round(chart_prediction,0)
        
        price_trend.append(chart_prediction)
        
    trend_data = {'price': price_trend, 'bedrooms': bedroom_count, 
            'neighbourhood_cleansed': [neighborhood,neighborhood,neighborhood,neighborhood,neighborhood,neighborhood],
            'instant_bookable': [instant_bookable,instant_bookable,instant_bookable,instant_bookable,instant_bookable,instant_bookable]
           }
    trend_model_range = pd.DataFrame.from_dict(trend_data)
    
    trend_model_results = pd.DataFrame(columns=['price', 'bedrooms', 'neighbourhood_cleansed', 'instant_bookable'])
    trend_model_results.loc[0, 'price'] = model_value
    trend_model_results.loc[0, 'bedrooms'] = model_bedrooms
    trend_model_results.loc[0, 'neighbourhood_cleansed'] = neighborhood
    trend_model_results.loc[0, 'instant_bookable'] = instant_bookable
    
    ############################################################
    # Chart creation
    
    # Configure the options common to all layers
    brush = alt.selection_interval()
    base = alt.Chart(filtered_graph_listing_df).add_params(brush)

    points = base.mark_point().encode(
        x=alt.X('bedrooms', title='Bedrooms'),
        y=alt.Y('price', title='Price', scale=alt.Scale(domain=[0, 1000])),  # adjust the domain values as needed
        color=alt.condition(brush, 'instant_bookable', alt.value('grey')),
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
        alt.Y('instant_bookable', title='', axis=tick_axis),
        color=alt.condition(brush, 'instant_bookable', alt.value('lightgrey'))
    ).properties(
            width=600
    )

    y_ticks = base.mark_tick().encode(
        alt.X('instant_bookable', title='', axis=tick_axis),
        alt.Y('price', axis=tick_axis),
        color=alt.condition(brush, 'instant_bookable', alt.value('lightgrey'))
    ).properties(
            height=600
    )

    model_results_line = alt.Chart(trend_model_range).mark_line(color='red').encode(
        x='bedrooms:Q',
        y=alt.Y('price', scale=alt.Scale(domain=[0, 1000]))  # adjust the domain values as needed
    )

    your_property_marker = alt.Chart(trend_model_results).mark_point(color='green', shape="cross", filled=True, size=500).encode(
        x='bedrooms:Q',
        y=alt.Y('price', scale=alt.Scale(domain=[0, 1000])),  # adjust the domain values as needed
        tooltip=['price', 'bedrooms', 'neighbourhood_cleansed', 'instant_bookable']
    )

    points_layered = (points + model_results_line + your_property_marker)

    final_chart = y_ticks | points_layered & x_ticks

    st.altair_chart(final_chart)



def choropleth(city, neighborhood):
    # Get GeoJson File and format
    geo_json_file_path = city_file_path = os.path.join('geojson_data', city, 'neighbourhoods.geojson')
    geo_json = json.load(open(geo_json_file_path))

    # Get HUD Data
    read_path = 'agg_choropleth_data/agg_neighborhood_df.pkl'  # CHANGE THIS IN PERM CODE!
    agg_neighborhoods_df = pd.read_pickle(read_path)

    agg_neighborhoods_df_city = agg_neighborhoods_df[agg_neighborhoods_df.city == city]
    agg_neighborhoods_df_city.rename(columns={'total_units': 'Affordable Units Available', 'pct_occupied':
                                              '% Affordable Units Occupied', 'rent_per_month': 'Avg Rent Per Month',
                                              'pct_lt50_median': '% Very Low Income', 'tpoverty': '% In Poverty'},
                                     inplace=True)

    # Get Lat Lons for City Center
    list_of_cities_lat_lon = sorted([('columbus', 39.9612, -82.9988),
                                      ('los-angeles', 34.0549, -118.2426),
                                      ('new-york-city', 40.7128, -74.0060),
                                      ('fort-worth', 32.7555, -97.3308),
                                      ('boston', 42.3601, -71.0589),
                                      ('broward-county', 26.1224, -80.1373),
                                      ('chicago', 41.8781, -87.6232),
                                      ('austin', 30.2672, -97.7431),
                                      ('seattle', 47.6061, -122.3328),
                                      ('rochester', 43.1566, -77.6088),
                                      ('san-francisco', 37.7749, -122.4194)])

    # Initialize the variable to None
    matching_lat_lon = None

    # Iterate over the list of tuples
    for city_lat_lon in list_of_cities_lat_lon:
        # If the first element of the tuple matches the city variable
        if city_lat_lon[0] == city:
            # Assign the matching tuple to the variable
            matching_tuple = city_lat_lon
            break  # Exit the loop

    city_center_lat_lons = matching_tuple
    city_center_lat = city_center_lat_lons[1]
    city_center_lon = city_center_lat_lons[2]

    # Selected neighborhood
    pin_point_df = pd.read_pickle('pin_point_coordinates.pkl')
    neigh_val = pin_point_df.coordinates[
        (pin_point_df['city'] == city) & (pin_point_df['neighborhood'] == neighborhood)]
    neigh_lat_lons = neigh_val.values

    ## Generate Figure
    hover_text = neighborhood

    # Location Hover Text
    for _, row in agg_neighborhoods_df_city.iterrows():
        for column in row.index:
            hover_text += f"<br>{column}: {row[column]}"

    # Choropleth
    fig = px.choropleth_mapbox(agg_neighborhoods_df_city.round(2),
                               geojson=geo_json,
                               locations="neighborhood",
                               featureidkey='properties.neighbourhood',
                               color='% Affordable Units Occupied',
                               color_continuous_scale=px.colors.sequential.Oryel,
                               range_color=(0, 100),
                               mapbox_style="carto-positron",
                               zoom=9, center={"lat": city_center_lat, "lon": city_center_lon},
                               opacity=0.7,
                               hover_name='% Affordable Units Occupied',
                               hover_data=['Affordable Units Available', '% Affordable Units Occupied',
                                           'Avg Rent Per Month',
                                           '% Very Low Income', '% In Poverty'],
                               title='Neighborhood Affordability Data'
                               )

    fig.update_layout(dragmode=False)

    # Property Address Marker
    fig.add_trace(go.Scattermapbox(
        lat=[neigh_lat_lons[0][0]],
        lon=[neigh_lat_lons[0][1]],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=10,
            color='green',
        ),
        text=[hover_text],
        hoverinfo='text'
    ))

    # Use st.plotly_chart to display the figure in Streamlit
    st.plotly_chart(fig)

def get_neighborhood_to_avg(city, city_neighborhood):
    read_path = 'agg_choropleth_data/agg_neighborhood_df.pkl'

    agg_neighborhoods_df = pd.read_pickle(read_path)

    agg_neighborhoods_df_city = agg_neighborhoods_df[agg_neighborhoods_df.city == city]
    
    agg_neighborhoods_df_city = agg_neighborhoods_df_city.drop(columns = ['city'])
    agg_neighborhoods_df_city = agg_neighborhoods_df_city.set_index('neighborhood')
    agg_neighborhoods_df_city.loc['Average Values'] = agg_neighborhoods_df_city.mean()
    agg_neighborhoods_df_city.loc['Median Values'] = agg_neighborhoods_df_city.median()
    agg_neighborhoods_df_city = agg_neighborhoods_df_city.reset_index().round(2)
    
    columns = agg_neighborhoods_df_city.columns.to_list()
    neighborhood = columns[0]
    other_columns = columns[1::]
    neighborhood_percentiles = ['25th Percentile','75th Percentile',"Max","Min"]
    percentiles = ['25%', '75%','max','min']

    for column_value, percentile in zip(neighborhood_percentiles, percentiles):
        value_list = [column_value]
        for column in other_columns:
            agg_neighborhoods_df_city[column].apply(pd.to_numeric)
            value = agg_neighborhoods_df_city[column].describe()
            value_list.append(value[percentile])
        agg_neighborhoods_df_city.loc[len(agg_neighborhoods_df_city)]=value_list
        
    avg_vals = "Average Values"
    hud_filtered_df = agg_neighborhoods_df_city.loc[agg_neighborhoods_df_city['neighborhood'].isin([avg_vals,city_neighborhood])]
    
    hud_filtered_df = hud_filtered_df.groupby('neighborhood').agg({
        'total_units': 'sum',
        'pct_occupied': 'mean',
        'number_reported': 'sum',
        'pct_reported': 'mean',
        'months_since_report': 'mean',
        'pct_movein': 'mean',
        'people_per_unit': 'mean',
        'people_total' : 'sum',
        'rent_per_month' : 'mean',
        'spending_per_month' : 'mean', 
        'hh_income' : 'mean',
        'person_income' : 'mean', 
        'pct_lt5k' : 'mean',
        'pct_5k_lt10k' :'mean',
        'pct_10k_lt15k' : 'mean',
        'pct_15k_lt20k' : 'mean',
        'pct_ge20k' : 'mean',
        'pct_wage_major' : 'mean',
        'pct_welfare_major' : 'mean',
        'pct_other_major' : 'mean',
        'pct_median' : 'mean', 
        'pct_lt50_median' : 'mean',
        'pct_lt30_median' :'mean',
        'pct_2adults' : 'mean',
        'pct_1adult' : 'mean', 
        'pct_female_head' : 'mean',
        'pct_female_head_child' : 'mean',
        'pct_disabled_lt62' : 'mean',
        'pct_disabled_ge62' : 'mean',
        'pct_disabled_all' : 'mean',
        'pct_lt24_head'  : 'mean',
        'pct_age25_50' : 'mean', 
        'pct_age51_61'  : 'mean',
        'pct_age62plus'  : 'mean',
        'pct_age85plus'  : 'mean',
        'pct_minority'  : 'mean',
        'pct_black_nonhsp'  : 'mean',
        'pct_native_american_nonhsp'  : 'mean',
        'pct_asian_pacific_nonhsp'  : 'mean',
        'pct_white_nothsp'  : 'mean', 
        'pct_black_hsp'  : 'mean',
        'pct_wht_hsp': 'mean',
        'pct_oth_hsp': 'mean',
        'pct_hispanic': 'mean',
        'pct_multi' : 'mean',
        'months_waiting'  : 'mean', 
        'months_from_movein'  : 'mean',
        'pct_utility_allow'  : 'mean',
        'ave_util_allow'  : 'mean',
        'pct_bed1'  : 'mean',
        'pct_bed2'  : 'mean', 
        'pct_bed3'  : 'mean', 
        'pct_overhoused'  : 'mean',
        'tpoverty'  : 'mean', 
        'tminority'  : 'mean',
        'tpct_ownsfd'  : 'mean'
    }).reset_index()
    
    hud_filtered_df = hud_filtered_df[['neighborhood', 'total_units', 'pct_occupied','people_per_unit', 'person_income','months_waiting','pct_overhoused', 'tpoverty','tpct_ownsfd']].round(2) 
    
    columns = ['total_units','pct_occupied','people_per_unit',
           'person_income','months_waiting',
           'tpoverty','tpct_ownsfd']

    titles = ['Total Units in Area w/ Government Subsidies', 'Occupied Units in Area as a % of Units Available', 
              'Average Size of Household in Area', 'Average Household Income per Person per Year in Area', 
              'Months on Waiting List in Area', '% of Surrounding Area in Poverty', 
              '% Surrounding Area Single Family Owners']


    plots = []

    for i, column in enumerate(columns):

        all_neighborhood_vals = alt.Chart(agg_neighborhoods_df_city).mark_boxplot(extent='min-max', color='teal', size=30).encode(
            x=alt.X(column+':Q'),
            tooltip=alt.Tooltip(column+':Q')
        ).properties(
            title=titles[i], 
            width=600, 
            height=200
        )


        neighborhood_vals = alt.Chart(hud_filtered_df[hud_filtered_df['neighborhood'].isin([city_neighborhood, 'Average Values'])]).mark_point(
            filled=True,
            size=100 
        ).encode(
            x=alt.X(column+':Q'),
            color=alt.Color('neighborhood:N', scale=alt.Scale(domain=[city_neighborhood, 'Average Values'], range=['red', 'blue'])),
            tooltip=[alt.Tooltip('neighborhood:N'), alt.Tooltip(column+':Q')]
        )


        box_and_marks = alt.layer(all_neighborhood_vals, neighborhood_vals)


        plots.append(box_and_marks)


    chart = alt.vconcat(*plots)


    chart = chart.configure_view(stroke='transparent', fill='white').configure_title(color='darkgrey').configure_axis(
        labelColor='darkgrey',
        titleColor='darkgrey'
    )
    # Display Altair chart in Streamlit
    st.altair_chart(chart)



######################################################################################


## functions end here, title, sidebar setting and descriptions start here
t1, t2 = st.columns(2)
with t1:
    st.title('Short-term Rental Pricing Predictor')

with t2:
    st.write("")
    st.write("")
    st.write("""
    *Prediction data provided by InsideAirbnb* | *Neighborhood level data provided by Department of Housing and Urban Development*
    """)

st.write("")
st.markdown("""
The short-term rental market landscape is quickly changing as inflation in operating expenses, increasing interest rates, and the end of covid stimulus threatens to erode operator profits. At the same time, the proliferation of properties owned solely for short term rental purposes has seen an inflationary impact on housing markets and has threatened renter affordability. Studies such as Horn and Merante's "Is home sharing driving up rents? Evidence from Airbnb in Boston", as well as Barron, Kung, and Proserpio's "The Sharing Economy and Housing Affordability: Evidence from Airbnb", have shown a significant impact of short term rental listings on the neighboring housing market. If this is true, and we believe as other studies such as Scott Susin’s "Rent vouchers and the price of low-income housing", show that the introduction of programs like housing vouchers have a direct impact on housing market affordability, we can logically connect the impact of the affordable housing market to the impact of short-term rentals on the private market.

Our mission is to create a toolset that will allow owner/operators to predict their optimal rental prices, and also visualize their property’s location and provide insight on the impact of listing their property in that neighborhood.


For additional information please contact *ryanwt@umich.edu* or *moura@umich.edu*.  
""")




#with st.sidebar.expander("Click to learn more about this dashboard"):
#    st.markdown(f"""
#    Using data from Inside Airbnb as well as the US Department of Housing and Urban Development, we've compiled a dasbhoard to help potential investors and renters find the ideal marketplace listing price, as well as gain a deeper understanding of the affordable rental market and constraints that disadvantaged indivduals face. 
#                
#    Our belief is that constraint of available units will be further tightened by pressure for short-term rentals. 
#    """)

# Add container for input widgets
st.sidebar.markdown("Model Input Parameters")

sidebar_city = st.sidebar.selectbox(
    'Select city:',
    ['columbus','los-angeles', 'new-york-city','fort-worth', 'boston', 'broward-county',
     'chicago', 'seattle', 'rochester', 'san-francisco'],
)

sidebar_neighborhood = st.sidebar.selectbox(
    'Select neighborhood:',
    neighborhoods[sidebar_city],
)

# 'host_is_superhost'
host_is_superhost = st.sidebar.toggle("Is Superhost?", value=1)
# 'host_listings_count'
host_listings_count = st.sidebar.number_input("Host Listings Count", min_value=1)
# 'host_total_listings_count'
host_total_listings_count = st.sidebar.number_input("Host Total Listings Count", min_value=1)
# 'host_identity_verified'
host_identity_verified = st.sidebar.toggle("Host Identity Verified?", value=1)
# 'accommodates'
accommodates = st.sidebar.number_input("Accommodates", min_value=0, value =4)
# 'bathrooms_text'
bathrooms_text = st.sidebar.number_input("Number of Bathrooms", min_value=1.0, value = 2.0)
# 'bedrooms'
bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=1, value=2)
# 'beds'
beds = st.sidebar.number_input("Number of Beds", min_value=1, value = 3)
# 'minimum_nights'
minimum_nights = st.sidebar.number_input("Minimum Nights", min_value=1, value = 1)
# 'maximum_nights'
maximum_nights = st.sidebar.number_input("Maximum Nights", min_value=1, value = 30)
# 'number_of_reviews'
number_of_reviews = st.sidebar.number_input("Number of Reviews", min_value=0, value = 5)
# 'number_of_reviews_ltm'
number_of_reviews_ltm = st.sidebar.number_input("Number of Reviews LTM", min_value=0, value = 2)
# 'number_of_reviews_l30d'
number_of_reviews_l30d = st.sidebar.number_input("Number of Reviews L30D", min_value=0, value = 1)
# 'review_scores_rating'
review_scores_rating = st.sidebar.number_input("Review Scores Rating", min_value=0.0, max_value=5.0, value = 4.5)
# 'review_scores_accuracy'
review_scores_accuracy = st.sidebar.number_input("Review Scores Accuracy", min_value=0.0, max_value=5.0, value = 4.5)
# 'review_scores_cleanliness'
review_scores_cleanliness = st.sidebar.number_input("Review Scores Cleanliness", min_value=0.0, max_value=5.0, value = 4.5)
# 'review_scores_checkin'
review_scores_checkin = st.sidebar.number_input("Review Scores Checkin", min_value=0.0, max_value=5.0, value = 4.5)
# 'review_scores_communication'
review_scores_communication = st.sidebar.number_input("Review Scores Communication", min_value=0.0, max_value=5.0, value = 4.5)
# 'review_scores_location'
review_scores_location = st.sidebar.number_input("Review Scores Location", min_value=0.0, max_value=5.0, value = 4.5)
# 'review_scores_value'
review_scores_value = st.sidebar.number_input("Review Scores Value", min_value=0.0, max_value=5.0, value = 4.5)
# 'instant_bookable'
instant_bookable = st.sidebar.toggle("Instant Bookable?", value =1)
# 'reviews_per_month'
reviews_per_month = st.sidebar.number_input("Reviews Per Month", min_value=0, value = 5)
# 'age'
age = st.sidebar.number_input("Age", min_value=0, value = 1)
# room_type
room_type_options = room_values_for_dropdown(room_type_columns)
room_type = st.sidebar.selectbox("Room Type", room_type_options)
# ‘property_type
property_type_options = property_values_for_dropdown(property_type_columns)
property_type = st.sidebar.selectbox("Property Type", property_type_options)

regenerate_button = st.sidebar.button("Regenerate Graphs")

#Create DF inputs to Model
submit_df = pd.DataFrame(columns=column_lst)
submit_df = update_city_submission(submit_df, sidebar_city)
submit_df = update_neighborhood_submission(submit_df, sidebar_neighborhood, directory)
submit_df.loc[0, 'host_is_superhost'] = host_is_superhost
submit_df.loc[0, 'host_listings_count'] = host_listings_count
submit_df.loc[0, 'host_total_listings_count'] = host_total_listings_count
submit_df.loc[0, 'host_identity_verified'] = host_identity_verified
submit_df.loc[0, 'accommodates'] = accommodates
submit_df.loc[0, 'bathrooms_text'] = host_identity_verified
submit_df.loc[0, 'bedrooms'] = bedrooms
submit_df.loc[0, 'beds'] = beds
submit_df.loc[0, 'minimum_nights'] = minimum_nights
submit_df.loc[0, 'maximum_nights'] = maximum_nights
submit_df.loc[0, 'number_of_reviews'] = number_of_reviews
submit_df.loc[0, 'number_of_reviews_ltm'] = number_of_reviews_ltm
submit_df.loc[0, 'number_of_reviews_l30d'] = number_of_reviews_l30d
submit_df.loc[0, 'review_scores_rating'] = review_scores_rating
submit_df.loc[0, 'review_scores_accuracy'] = review_scores_accuracy
submit_df.loc[0, 'review_scores_cleanliness'] = review_scores_cleanliness
submit_df.loc[0, 'review_scores_checkin'] = review_scores_checkin
submit_df.loc[0, 'review_scores_communication'] = review_scores_communication
submit_df.loc[0, 'review_scores_location'] = review_scores_location
submit_df.loc[0, 'review_scores_value'] = review_scores_value
submit_df.loc[0, 'instant_bookable'] = instant_bookable
submit_df.loc[0, 'reviews_per_month'] = reviews_per_month
submit_df.loc[0, 'age'] = age
submit_df = update_room_type_submission(submit_df, room_type, room_type_options)
submit_df = update_property_type_submission(submit_df, property_type, property_type_options)
submit_df = submit_df.apply(pd.to_numeric, errors='ignore')
predictions = loaded_model.predict(submit_df)
prediction_val = round(predictions[0],2)
text_prediciton_val = "Your estimated nightly rental value is $"+str(prediction_val)

model_value_text = st.header(body = text_prediciton_val, divider = 'orange')

# Create tabs
selected_tab = st.radio("Select a tab", ["Model Results", "Choropleth", "Neighborhood Stats"])

# Check if the "Regenerate Graphs" button is clicked
if regenerate_button:
    # Check which tab is selected
    if selected_tab == "Model Results":
        # Display the model results graph
        model_results_graph(loaded_model, submit_df, sidebar_city, sidebar_neighborhood, prediction_val, bedrooms, instant_bookable)
    elif selected_tab == "Choropleth":
        # Display the choropleth graph
        choropleth(sidebar_city, sidebar_neighborhood)
    elif selected_tab == "Neighborhood Stats":
        # Display the neighborhood stats graph
        get_neighborhood_to_avg(sidebar_city, sidebar_neighborhood)

# Check which tab is selected (outside the "if regenerate_button" block)
if selected_tab == "Model Results":
    # Display the model results graph
    model_results_graph(loaded_model, submit_df, sidebar_city, sidebar_neighborhood, prediction_val, bedrooms, instant_bookable)
elif selected_tab == "Choropleth":
    # Display the choropleth graph
    choropleth(sidebar_city, sidebar_neighborhood)
elif selected_tab == "Neighborhood Stats":
    # Display the neighborhood stats graph
    get_neighborhood_to_avg(sidebar_city, sidebar_neighborhood)




if _ENABLE_PROFILING:
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    ts = int(time.time())
    with open(f"perf_{ts}.txt", "w") as f:
        f.write(s.getvalue())