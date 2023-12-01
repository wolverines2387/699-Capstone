import json
from datetime import date
from urllib.request import urlopen
import time
import os 
import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
## No longer need this with this version of python
#from pandas.io.json import json_normalize

_ENABLE_PROFILING = False

if _ENABLE_PROFILING:
    import cProfile, pstats, io
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()

today = date.today()

def get_neighborhoods(directory):
    cities = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    city_dir = {}
    
    for city in cities:
        city_file_path = os.path.join(directory, city,'listings.csv.gz')
        city_df = pd.read_csv(city_file_path)
        neighbourhood_list = city_df.neighbourhood_cleansed.unique().tolist()
        
        city_dir[city] = neighbourhood_list
    
    return city_dir

directory = 'data'
neighborhoods = get_neighborhoods(directory)

st.set_page_config(
    page_title="Short-term Rental Pricing Predictor",
    layout='wide',
    initial_sidebar_state='auto',
)

sidebar_city = st.sidebar.selectbox(
    'Select city:',
    ['columbus','los-angeles', 'new-york-city','fort-worth', 'boston', 'broward-county',
     'chicago', 'austin', 'seattle', 'rochester', 'san-francisco'],
)

sidebar_neighborhood = st.sidebar.selectbox(
    'Select neighborhood:',
    neighborhoods[sidebar_city],
)

###Figure out where to place this: ##################################################################################


#################################################################

@st.cache_data(ttl=3*60*60)
def get_data():
    US_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
    US_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
    confirmed = pd.read_csv(US_confirmed)
    deaths = pd.read_csv(US_deaths)
    return confirmed, deaths


confirmed, deaths = get_data()
FIPSs = confirmed.groupby(['Province_State', 'Admin2']).FIPS.unique().apply(pd.Series).reset_index()
FIPSs.columns = ['State', 'County', 'FIPS']
FIPSs['FIPS'].fillna(0, inplace = True)
FIPSs['FIPS'] = FIPSs.FIPS.astype(int).astype(str).str.zfill(5)

@st.cache_data(ttl=3*60*60)
def get_testing_data(County):
    apiKey = '9fe19182c5bf4d1bb105da08e593a578'
    if len(County) == 1:
        #print(len(County))
        f = FIPSs[FIPSs.County == County[0]].FIPS.values[0]
        #print(f)
        path1 = 'https://data.covidactnow.org/latest/us/counties/'+f+'.OBSERVED_INTERVENTION.timeseries.json?apiKey='+apiKey
        #print(path1)
        df = json.loads(requests.get(path1).text)
        #print(df.keys())
        data = pd.DataFrame.from_dict(df['actualsTimeseries'])
        data['Date'] = pd.to_datetime(data['date'])
        data = data.set_index('Date')
        #print(data.tail())
        try:
            data['new_negative_tests'] = data['cumulativeNegativeTests'].diff()
            data.loc[(data['new_negative_tests'] < 0)] = np.nan
        except: 
            data['new_negative_tests'] = np.nan
            st.text('Negative test data not avilable')
        data['new_negative_tests_rolling'] = data['new_negative_tests'].fillna(0).rolling(14).mean()


        try:
            data['new_positive_tests'] = data['cumulativePositiveTests'].diff()
            data.loc[(data['new_positive_tests'] < 0)] = np.nan
        except: 
            data['new_positive_tests'] = np.nan
            st.text('test data not avilable')
        data['new_positive_tests_rolling'] = data['new_positive_tests'].fillna(0).rolling(14).mean()
        data['new_tests'] = data['new_negative_tests']+data['new_positive_tests']
        data['new_tests_rolling'] = data['new_tests'].fillna(0).rolling(14).mean()
        data['testing_positivity_rolling'] = (data['new_positive_tests_rolling'] / data['new_tests_rolling'])*100
        #data['testing_positivity_rolling'].tail(14).plot()
        #plt.show()
        return data['new_tests_rolling'], data['testing_positivity_rolling'].iloc[-1:].values[0]
    elif (len(County) > 1) & (len(County) < 5):
        new_positive_tests = []
        new_negative_tests = []
        new_tests = []
        for c in County:
            f = FIPSs[FIPSs.County == c].FIPS.values[0]
            path1 = 'https://data.covidactnow.org/latest/us/counties/'+f+'.OBSERVED_INTERVENTION.timeseries.json?apiKey='+apiKey
            df = json.loads(requests.get(path1).text)
            data = pd.DataFrame.from_dict(df['actualsTimeseries'])
            data['Date'] = pd.to_datetime(data['date'])
            data = data.set_index('Date')
            try:
                data['new_negative_tests'] = data['cumulativeNegativeTests'].diff()
                data.loc[(data['new_negative_tests'] < 0)] = np.nan
            except: 
                data['new_negative_tests'] = np.nan
                #print('Negative test data not avilable')

            try:
                data['new_positive_tests'] = data['cumulativePositiveTests'].diff()
                data.loc[(data['new_positive_tests'] < 0)] = np.nan
            except: 
                data['new_positive_tests'] = np.nan
                #print('Negative test data not avilable')
            data['new_tests'] = data['new_negative_tests']+data['new_positive_tests']

            new_positive_tests.append(data['new_positive_tests'])
            #new_negative_tests.append(data['new_tests'])
            new_tests.append(data['new_tests'])
            #print(data.head())

        new_positive_tests_rolling = pd.concat(new_positive_tests, axis = 1).sum(axis = 1)
        new_positive_tests_rolling = new_positive_tests_rolling.fillna(0).rolling(14).mean()
        #print('new test merging of counties')
        #print(pd.concat(new_tests, axis = 1).head().sum(axis = 1))
        new_tests_rolling = pd.concat(new_tests, axis = 1).sum(axis = 1)
        new_tests_rolling = new_tests_rolling.fillna(0).rolling(14).mean()
        new_tests_rolling = pd.DataFrame(new_tests_rolling).fillna(0)
        new_tests_rolling.columns = ['new_tests_rolling']
        #print('whole df')
        #print(type(new_tests_rolling))
        #print(new_tests_rolling.head())
        #print('single column')
        #print(new_tests_rolling['new_tests_rolling'].head())
        #print('new_positive_tests_rolling')
        #print(new_positive_tests_rolling.head())
        #print('new_tests_rolling')
        #print(new_tests_rolling.head())
        data_to_show = (new_positive_tests_rolling / new_tests_rolling.new_tests_rolling)*100
        #print(data_to_show.shape)
        #print(data_to_show.head())
        #print(data_to_show.columns)
        #print(data_to_show.iloc[-1:].values[0])
        return new_tests_rolling, data_to_show.iloc[-1:].values[0]
    else:
        st.text('Getting testing data for California State')
        path1 = 'https://data.covidactnow.org/latest/us/states/CA.OBSERVED_INTERVENTION.timeseries.json'
        df = json.loads(requests.get(path1).text)
        data = pd.DataFrame.from_dict(df['actualsTimeseries'])
        data['Date'] = pd.to_datetime(data['date'])
        data = data.set_index('Date')

        try:
            data['new_negative_tests'] = data['cumulativeNegativeTests'].diff()
            data.loc[(data['new_negative_tests'] < 0)] = np.nan
        except:
            data['new_negative_tests'] = np.nan
            print('Negative test data not available')
        data['new_negative_tests_rolling'] = data['new_negative_tests'].fillna(0).rolling(14).mean()


        try:
            data['new_positive_tests'] = data['cumulativePositiveTests'].diff()
            data.loc[(data['new_positive_tests'] < 0)] = np.nan
        except:
            data['new_positive_tests'] = np.nan
            st.text('test data not available')
        data['new_positive_tests_rolling'] = data['new_positive_tests'].fillna(0).rolling(14).mean()
        data['new_tests'] = data['new_negative_tests']+data['new_positive_tests']
        data['new_tests_rolling'] = data['new_tests'].fillna(0).rolling(14).mean()
        data['testing_positivity_rolling'] = (data['new_positive_tests_rolling'] / data['new_tests_rolling'])*100
        return data['new_tests_rolling'], data['testing_positivity_rolling'].iloc[-1:].values[0]
    
#### Our graphs before I comment all this other stuff out....

def choropleth(city, neighborhood):
    # Get GeoJson File and format
    geo_json_file_path = city_file_path = os.path.join('geojson_data', city, 'neighbourhoods.geojson')
    geo_json = json.load(open(geo_json_file_path))

    # Get HUD Data
    read_path = 'Chicago_Data/agg_choropleth_data/hud_neighborhoods_agg.pkl'  # CHANGE THIS IN PERM CODE!
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
    read_path = 'Chicago_Data/agg_choropleth_data/hud_neighborhoods_agg.csv'

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
    # Convert Altair chart to JSON format
    chart_json = chart.to_json()

    # Display Altair chart in Streamlit
    st.altair_chart(chart_json)


######################################################################################


def plot_county(county):
    testing_df, testing_percent = get_testing_data(County=county)
    #print(testing_df.head())
    county_confirmed = confirmed[confirmed.Admin2.isin(county)]
    county_confirmed_time = county_confirmed.drop(county_confirmed.iloc[:, 0:12], axis=1).T
    county_confirmed_time = county_confirmed_time.sum(axis= 1)
    county_confirmed_time = county_confirmed_time.reset_index()
    county_confirmed_time.columns = ['date', 'cases']
    county_confirmed_time['Datetime'] = pd.to_datetime(county_confirmed_time['date'])
    county_confirmed_time = county_confirmed_time.set_index('Datetime')
    del county_confirmed_time['date']
    incidence= pd.DataFrame(county_confirmed_time.cases.diff())
    incidence.columns = ['incidence']
    chart_max = incidence.max().values[0]+500

    county_deaths = deaths[deaths.Admin2.isin(county)]
    population = county_deaths.Population.values.sum()

    del county_deaths['Population']
    county_deaths_time = county_deaths.drop(county_deaths.iloc[:, 0:11], axis=1).T
    county_deaths_time = county_deaths_time.sum(axis= 1)

    county_deaths_time = county_deaths_time.reset_index()
    county_deaths_time.columns = ['date', 'deaths']
    county_deaths_time['Datetime'] = pd.to_datetime(county_deaths_time['date'])
    county_deaths_time = county_deaths_time.set_index('Datetime')
    del county_deaths_time['date']

    cases_per100k  = ((county_confirmed_time) * 100000 / population)
    cases_per100k.columns = ['cases per 100K']
    cases_per100k['rolling average'] = cases_per100k['cases per 100K'].rolling(7).mean()

    deaths_per100k  = ((county_deaths_time) * 100000 / population)
    deaths_per100k.columns = ['deaths per 100K']
    deaths_per100k['rolling average'] = deaths_per100k['deaths per 100K'].rolling(7).mean()


    incidence['rolling_incidence'] = incidence.incidence.rolling(7).mean()
    metric = (incidence['rolling_incidence'] * 100000 / population).iloc[[-1]]

    if len(county) == 1:
        st.subheader('Current situation of COVID-19 cases in '+', '.join(map(str, county))+' county ('+ str(today)+')')
    else:
        st.subheader('Current situation of COVID-19 cases in '+', '.join(map(str, county))+' counties ('+ str(today)+')')

    c1 = st.container()
    c2 = st.container()
    c3 = st.container()

    if len(county)==1:
        C = county[0]
        with c2:
            a1, _, a2 = st.columns((3.9, 0.2, 3.9))     
            with a1:
                f = FIPSs[FIPSs.County == C].FIPS.values[0]
                components.iframe("https://covidactnow.org/embed/us/county/"+f, width=350, height=365, scrolling=False)
                
            with a2:
                st.markdown('New cases averaged over last 7 days = %s' %'{:,.1f}'.format(metric.values[0]))
                st.markdown("Population under consideration = %s"% '{:,.0f}'.format(population))
                st.markdown("Total cases = %s"% '{:,.0f}'.format(county_confirmed_time.tail(1).values[0][0]))
                st.markdown("Total deaths = %s"% '{:,.0f}'.format(county_deaths_time.tail(1).values[0][0]))
                st.markdown("% test positivity (14 day average)* = "+"%.2f" % testing_percent)
    elif len(county) <= 3:
        with c2:
            st.write('')
            st.write('')
            st.markdown("New cases averaged over last 7 days = %s" % "{:,.1f}".format(metric.values[0]))
            st.markdown("Population under consideration = %s"% '{:,.0f}'.format(population))
            st.markdown("Total cases = %s"% '{:,.0f}'.format(county_confirmed_time.tail(1).values[0][0]))
            st.markdown("Total deaths = %s"% '{:,.0f}'.format(county_deaths_time.tail(1).values[0][0]))
            st.markdown("% test positivity (14 day average)* = "+"%.2f" % testing_percent)
        with c3:
            columns = st.columns(len(county))
            for idx, C in enumerate(county):
                with columns[idx]:
                    st.write('')
                    st.write('')
                    f = FIPSs[FIPSs.County == C].FIPS.values[0]
                    components.iframe("https://covidactnow.org/embed/us/county/"+f, width=350, height=365, scrolling=False)

    ### Experiment with Altair instead of Matplotlib.
    with c1:
        a2, _, a1 = st.columns((3.9, 0.2, 3.9))

        incidence = incidence.reset_index()
        incidence['nomalized_rolling_incidence'] = incidence['rolling_incidence'] * 100000 / population
        incidence['Phase 2 Threshold'] = 25
        incidence['Phase 3 Threshold'] = 10
        scale = alt.Scale(
            domain=[
                "rolling_incidence",
                "Phase 2 Threshold",
                "Phase 3 Threshold"
            ], range=['#377eb8', '#e41a1c', '#4daf4a'])
        base = alt.Chart(
            incidence,
            title='(A) Weekly rolling mean of incidence per 100K'
        ).transform_calculate(
            base_="'rolling_incidence'",
            phase2_="'Phase 2 Threshold'",
            phase3_="'Phase 3 Threshold'",
        )
        
        ax4 = base.mark_line(strokeWidth=3).encode(
            x=alt.X("Datetime", axis = alt.Axis(title='Date')),
            y=alt.Y("nomalized_rolling_incidence", axis=alt.Axis(title='per 100 thousand')),
            color=alt.Color("base_:N", scale=scale, title="")
        )

        line1 = base.mark_line(strokeDash=[8, 8], strokeWidth=2).encode(
            x=alt.X("Datetime", axis=alt.Axis(title = 'Date')),
            y=alt.Y("Phase 2 Threshold", axis=alt.Axis(title='Count')),
            color=alt.Color("phase2_:N", scale=scale, title="")
        )

        line2 = base.mark_line(strokeDash=[8, 8], strokeWidth=2).encode(
            x=alt.X("Datetime", axis=alt.Axis(title='Date')),
            y=alt.Y("Phase 3 Threshold", axis=alt.Axis(title='Count')),
            color=alt.Color("phase3_:N", scale=scale, title="")
        )

        with a2:
            st.altair_chart(ax4 + line1 + line2, use_container_width=True)

        ax3 = alt.Chart(incidence, title = '(B) Daily incidence (new cases)').mark_bar().encode(
            x=alt.X("Datetime",axis = alt.Axis(title = 'Date')),
            y=alt.Y("incidence",axis = alt.Axis(title = 'Incidence'), scale=alt.Scale(domain=(0, chart_max), clamp=True))
        )
        
        with a1:
            st.altair_chart(ax3, use_container_width=True)
        
        a3, _, a4 = st.columns((3.9, 0.2, 3.9))
        testing_df = pd.DataFrame(testing_df).reset_index()
        #print(testing_df.head())
        #print(type(testing_df))
        
        base = alt.Chart(testing_df, title = '(D) Daily new tests').mark_line(strokeWidth=3).encode(
            x=alt.X("Date",axis = alt.Axis(title = 'Date')),
            y=alt.Y("new_tests_rolling",axis = alt.Axis(title = 'Daily new tests'))
        )
        with a4:
            st.altair_chart(base, use_container_width=True)

        county_confirmed_time = county_confirmed_time.reset_index()
        county_deaths_time = county_deaths_time.reset_index()
        cases_and_deaths = county_confirmed_time.set_index("Datetime").join(county_deaths_time.set_index("Datetime"))
        cases_and_deaths = cases_and_deaths.reset_index()

        # Custom colors for layered charts.
        # See https://stackoverflow.com/questions/61543503/add-legend-to-line-bars-to-altair-chart-without-using-size-color.
        scale = alt.Scale(domain=["cases", "deaths"], range=['#377eb8', '#e41a1c'])
        base = alt.Chart(
            cases_and_deaths,
            title='(C) Cumulative cases and deaths'
        ).transform_calculate(
            cases_="'cases'",
            deaths_="'deaths'",
        )

        c = base.mark_line(strokeWidth=3).encode(
            x=alt.X("Datetime", axis=alt.Axis(title = 'Date')),
            y=alt.Y("cases", axis=alt.Axis(title = 'Count')),
            color=alt.Color("cases_:N", scale=scale, title="")
        )

        d = base.mark_line(strokeWidth=3).encode(
            x=alt.X("Datetime", axis=alt.Axis(title='Date')),
            y=alt.Y("deaths", axis=alt.Axis(title = 'Count')),
            color=alt.Color("deaths_:N", scale=scale, title="")
        )
        with a3:
            st.altair_chart(c+d, use_container_width=True)


def plot_state():
    @st.cache(ttl=3*60*60, suppress_st_warning=True)
    def get_testing_data_state():
            st.text('Getting testing data for California State')
            path1 = 'https://data.covidactnow.org/latest/us/states/CA.OBSERVED_INTERVENTION.timeseries.json'
            df = json.loads(requests.get(path1).text)
            data = pd.DataFrame.from_dict(df['actualsTimeseries'])
            data['Date'] = pd.to_datetime(data['date'])
            data = data.set_index('Date')

            try:
                data['new_negative_tests'] = data['cumulativeNegativeTests'].diff()
                data.loc[(data['new_negative_tests'] < 0)] = np.nan
            except:
                data['new_negative_tests'] = np.nan
                print('Negative test data not available')
            data['new_negative_tests_rolling'] = data['new_negative_tests'].fillna(0).rolling(14).mean()


            try:
                data['new_positive_tests'] = data['cumulativePositiveTests'].diff()
                data.loc[(data['new_positive_tests'] < 0)] = np.nan
            except:
                data['new_positive_tests'] = np.nan
                st.text('test data not available')
            data['new_positive_tests_rolling'] = data['new_positive_tests'].fillna(0).rolling(14).mean()
            data['new_tests'] = data['new_negative_tests']+data['new_positive_tests']
            data['new_tests_rolling'] = data['new_tests'].fillna(0).rolling(14).mean()
            data['testing_positivity_rolling'] = (data['new_positive_tests_rolling'] / data['new_tests_rolling'])*100
            # return data['new_tests_rolling'], data['testing_positivity_rolling'].iloc[-1:].values[0]
            testing_df, testing_percent = data['new_tests_rolling'], data['testing_positivity_rolling'].iloc[-1:].values[0]
            county_confirmed = confirmed[confirmed.Province_State == 'California']
            #county_confirmed = confirmed[confirmed.Admin2 == county]
            county_confirmed_time = county_confirmed.drop(county_confirmed.iloc[:, 0:12], axis=1).T #inplace=True, axis=1
            county_confirmed_time = county_confirmed_time.sum(axis= 1)
            county_confirmed_time = county_confirmed_time.reset_index()
            county_confirmed_time.columns = ['date', 'cases']
            county_confirmed_time['Datetime'] = pd.to_datetime(county_confirmed_time['date'])
            county_confirmed_time = county_confirmed_time.set_index('Datetime')
            del county_confirmed_time['date']
            #print(county_confirmed_time.head())
            incidence = pd.DataFrame(county_confirmed_time.cases.diff())
            incidence.columns = ['incidence']

            #temp_df_time = temp_df.drop(['date'], axis=0).T #inplace=True, axis=1
            county_deaths = deaths[deaths.Province_State == 'California']
            population = county_deaths.Population.values.sum()

            del county_deaths['Population']
            county_deaths_time = county_deaths.drop(county_deaths.iloc[:, 0:11], axis=1).T #inplace=True, axis=1
            county_deaths_time = county_deaths_time.sum(axis= 1)

            county_deaths_time = county_deaths_time.reset_index()
            county_deaths_time.columns = ['date', 'deaths']
            county_deaths_time['Datetime'] = pd.to_datetime(county_deaths_time['date'])
            county_deaths_time = county_deaths_time.set_index('Datetime')
            del county_deaths_time['date']

            cases_per100k  = ((county_confirmed_time)*100000/population)
            cases_per100k.columns = ['cases per 100K']
            cases_per100k['rolling average'] = cases_per100k['cases per 100K'].rolling(7).mean()

            deaths_per100k  = ((county_deaths_time)*100000/population)
            deaths_per100k.columns = ['deaths per 100K']
            deaths_per100k['rolling average'] = deaths_per100k['deaths per 100K'].rolling(7).mean()

            incidence['rolling_incidence'] = incidence.incidence.rolling(7).mean()
            return population, testing_df, testing_percent, county_deaths_time, county_confirmed_time, incidence
    # metric = (incidence['rolling_incidence']*100000/population).iloc[[-1]]

    #print(county_deaths_time.tail(1).values[0])
    #print(cases_per100k.head())
    population, testing_df, testing_percent, county_deaths_time, county_confirmed_time, incidence = get_testing_data_state()
    st.subheader('Current situation of COVID-19 cases in California ('+ str(today)+')')
    c1 = st.container()
    c2 = st.container()
    c3 = st.container()

    with c2:
        a1, _, a2 = st.columns((3.9, 0.2, 3.9))     
        with a1:
            #f = FIPSs[FIPSs.County == C].FIPS.values[0]
            components.iframe("https://covidactnow.org/embed/us/california-ca", width=350, height=365, scrolling=False)

        with a2:
            st.markdown("Population under consideration = %s"% '{:,.0f}'.format(population))
            st.markdown("% test positivity (14 day average) = "+"%.2f" % testing_percent)
            st.markdown("Total cases = %s"% '{:,.0f}'.format(county_confirmed_time.tail(1).values[0][0]))
            st.markdown("Total deaths = %s"% '{:,.0f}'.format(county_deaths_time.tail(1).values[0][0]))
            
    ### Experiment with Altair instead of Matplotlib.
    with c1:
        a2, _, a1 = st.columns((3.9, 0.2, 3.9))

        incidence = incidence.reset_index()
        incidence['nomalized_rolling_incidence'] = incidence['rolling_incidence'] * 100000 / population
        incidence['Phase 2 Threshold'] = 25
        incidence['Phase 3 Threshold'] = 10
        
        scale = alt.Scale(
            domain=[
                "rolling_incidence",
                "Phase 2 Threshold",
                "Phase 3 Threshold"
            ], range=['#377eb8', '#e41a1c', '#4daf4a'])
        base = alt.Chart(
            incidence,
            title='(A) Weekly rolling mean of incidence per 100K'
        ).transform_calculate(
            base_="'rolling_incidence'",
            phase2_="'Phase 2 Threshold'",
            phase3_="'Phase 3 Threshold'",
        )
        
        ax4 = base.mark_line(strokeWidth=3).encode(
            x=alt.X("Datetime", axis = alt.Axis(title='Date')),
            y=alt.Y("nomalized_rolling_incidence", axis=alt.Axis(title='per 100 thousand')),
            color=alt.Color("base_:N", scale=scale, title="")
        )

        line1 = base.mark_line(strokeDash=[8, 8], strokeWidth=2).encode(
            x=alt.X("Datetime", axis=alt.Axis(title = 'Date')),
            y=alt.Y("Phase 2 Threshold", axis=alt.Axis(title='Count')),
            color=alt.Color("phase2_:N", scale=scale, title="")
        )

        line2 = base.mark_line(strokeDash=[8, 8], strokeWidth=2).encode(
            x=alt.X("Datetime", axis=alt.Axis(title='Date')),
            y=alt.Y("Phase 3 Threshold", axis=alt.Axis(title='Count')),
            color=alt.Color("phase3_:N", scale=scale, title="")
        )
        with a2:
            st.altair_chart(ax4 + line1 + line2, use_container_width=True)

        ax3 = alt.Chart(incidence, title = '(B) Daily incidence (new cases)').mark_bar().encode(
            x=alt.X("Datetime",axis = alt.Axis(title = 'Date')),
            y=alt.Y("incidence",axis = alt.Axis(title = 'Incidence'))
        )
        
        with a1:
            st.altair_chart(ax3, use_container_width=True)
        
        a3, _, a4 = st.columns((3.9, 0.2, 3.9))
        testing_df = pd.DataFrame(testing_df).reset_index()
        #print(testing_df.head())
        #print(type(testing_df))
        
        base = alt.Chart(testing_df, title = '(D) Daily new tests').mark_line(strokeWidth=3).encode(
            x=alt.X("Date",axis = alt.Axis(title = 'Date')),
            y=alt.Y("new_tests_rolling",axis = alt.Axis(title = 'Daily new tests'))
        )
        with a4:
            st.altair_chart(base, use_container_width=True)

        county_confirmed_time = county_confirmed_time.reset_index()
        county_deaths_time = county_deaths_time.reset_index()
        cases_and_deaths = county_confirmed_time.set_index("Datetime").join(county_deaths_time.set_index("Datetime"))
        cases_and_deaths = cases_and_deaths.reset_index()

        # Custom colors for layered charts.
        # See https://stackoverflow.com/questions/61543503/add-legend-to-line-bars-to-altair-chart-without-using-size-color.
        scale = alt.Scale(domain=["cases", "deaths"], range=['#377eb8', '#e41a1c'])
        base = alt.Chart(
            cases_and_deaths,
            title='(C) Cumulative cases and deaths'
        ).transform_calculate(
            cases_="'cases'",
            deaths_="'deaths'",
        )

        c = base.mark_line(strokeWidth=3).encode(
            x=alt.X("Datetime", axis=alt.Axis(title = 'Date')),
            y=alt.Y("cases", axis=alt.Axis(title = 'Count')),
            color=alt.Color("cases_:N", scale=scale, title="")
        )

        d = base.mark_line(strokeWidth=3).encode(
            x=alt.X("Datetime", axis=alt.Axis(title='Date')),
            y=alt.Y("deaths", axis=alt.Axis(title = 'Count')),
            color=alt.Color("deaths_:N", scale=scale, title="")
        )
        with a3:
            st.altair_chart(c+d, use_container_width=True)


## functions end here, title, sidebar setting and descriptions start here
t1, t2 = st.columns(2)
with t1:
    st.markdown('*Short-term Rental Pricing Predictor*')

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


if sidebar_city == 'Select Neighborhoods':
    st.markdown('## Select neighborhoods of interest')
    CA_counties = confirmed[confirmed.Province_State == 'California'].Admin2.unique().tolist()
    counties = st.multiselect('', CA_counties, default=['Yolo', 'Solano', 'Sacramento'])
    # Limit to the first 5 counties.
    counties = counties[:5]
    if not counties:
        # If no counties are specified, just plot the state.
        st.markdown('> No counties were selected, falling back to showing statistics for California state.')
        plot_state()
    else:
        # Plot the aggregate and per-county details.
        plot_county(counties)
        for c in counties:
            st.write('')
            with st.expander(f"Expand for {c} County Details"):
                plot_county([c])
elif sidebar_city == 'California':
    plot_state()

with st.sidebar.expander("Click to learn more about this dashboard"):
    st.markdown(f"""
    Using data from Inside Airbnb as well as the US Department of Housing and Urban Development, we've compiled a dasbhoard to help potential investors and renters find the ideal marketplace listing price, as well as gain a deeper understanding of the affordable rental market and constraints that disadvantaged indivduals face. 
                
    Our belief is that constraint of available units will be further tightened by pressure for short-term rentals. 
    """)

if _ENABLE_PROFILING:
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    ts = int(time.time())
    with open(f"perf_{ts}.txt", "w") as f:
        f.write(s.getvalue())
