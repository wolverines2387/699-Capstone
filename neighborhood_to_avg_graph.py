import numpy as np
import pandas as pd
from scipy import stats
import altair as alt

#City Variable.
city = 'boston'
city_neighborhood = "East Boston"

## Get HUD Data
read_path = 'agg_choropleth_data/agg_neighborhood_df.pkl'

agg_neighborhoods_df = pd.read_pickle(read_path)

agg_neighborhoods_df_city = agg_neighborhoods_df[agg_neighborhoods_df.city == city]

#Columns to keep

# Get Median and Average
agg_neighborhoods_df_city = agg_neighborhoods_df_city.drop(columns = ['city'])
agg_neighborhoods_df_city = agg_neighborhoods_df_city.set_index('neighborhood')
agg_neighborhoods_df_city.loc['Average Values'] = agg_neighborhoods_df_city.mean()
agg_neighborhoods_df_city.loc['Median Values'] = agg_neighborhoods_df_city.median()
agg_neighborhoods_df_city = agg_neighborhoods_df_city.reset_index().round(2) 

#Get 25th and 75th Percentiles
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

def return_neighborhood_values(df, neighborhood):
    
    avg_vals = "Average Values"
    #med_vals = 'Median Values'
    #twenty_fifth = '25th Percentile'
    #seventy_fifth = '75th Percentile'
    #max_val = "Max"
    #min_val = "Min"
    
    #new_df = df[(df['neighborhood']== avg_vals) or (df['neighborhood'] == neighborhood)]
    new_df = df.loc[df['neighborhood'].isin([avg_vals,neighborhood])]
    
    return new_df

hud_filtered_df = return_neighborhood_values(agg_neighborhoods_df_city, city_neighborhood)

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

hud_filtered_df = hud_filtered_df[['neighborhood', 'total_units', 'pct_occupied','people_per_unit', 
                                   'person_income', 'months_waiting','pct_overhoused', 'tpoverty',
                                   'tpct_ownsfd']].round(2)

#Generate Graph
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

# How do I send this to the portal?
chart.save('neighborhood_to_avg.html')



