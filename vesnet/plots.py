import matplotlib.pyplot as plt
import matplotlib.colors as colors
import calendar
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
import seaborn as sns
sns.color_palette("rocket_r", as_cmap=True)
sns.set(font_scale = 1.5)
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

import geopandas as gpd
# import geoplot
from shapely.geometry import Point
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime
import folium
from folium import plugins
import numpy as np 

def get_plots(df_events, vessel_name):
    vessel = df_events[(df_events['vessels.vessel_0.name']==vessel_name) | (df_events['vessels.vessel_1.name']==vessel_name) 
                    |(df_events['vessels.vessel_0.name']==vessel_name) | (df_events['vessels.vessel_1.name']==vessel_name)]
    vessel['day_number'] = pd.to_datetime(vessel['start.time']).dt.weekday
    vessel['hour'] = pd.to_datetime(vessel['start.time']).dt.hour
    vessel['year'] = pd.to_datetime(vessel['start.time']).dt.year
    vessel['month-year']= pd.to_datetime(vessel['start.time']).dt.to_period('M')
    vessel['day_name'] = pd.to_datetime(vessel['start.time']).dt.day_name()
    vessel['month']= pd.to_datetime(vessel['start.time']).dt.month
    cats = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    vessel['day_name'] = pd.Categorical(vessel['day_name'], categories=cats, ordered=True)
    vessel['date'] = pd.to_datetime(vessel['start.time'])

    dark_df_time_day = vessel['event_id'].groupby(vessel['date'].dt.to_period('D')).count()
    dark_df_time_day = dark_df_time_day.resample('D').asfreq().fillna(0)

    hour_day = vessel.groupby(['hour', 'day_name'])['event_id'].count().unstack()
    
    month_day = vessel.groupby(['month-year', 'day_name'])['event_id'].count().unstack()
    
    # month_year = vessel.groupby(['month_name', 'month_name'])['event_id'].count().unstack()

    fig, ((ax1, cbar_ax), (ax2, dummy_ax),(ax3, dummy_ax2),(ax4, cbar_ax2), (ax5, dummy_ax3),(ax6, dummy_ax4), (ax7, dummy_ax5)) = plt.subplots(nrows=7, ncols=2, figsize=(20,50),gridspec_kw={'height_ratios': [5,2,2,5,2,2,2], 'width_ratios': [20, 1]})
  
    sns.heatmap(hour_day,
                    cmap="rocket_r",  # Choose a sequential colormap
                    annot=True, # Label the maximum value
                    ax=ax1,
                    cbar_ax=cbar_ax
                   )

    hour_day2 = pd.DataFrame(vessel.groupby(['hour', 'day_name'])['event_id'].count())
    hour_day2.reset_index(drop=False, inplace=True)
    sns.boxplot(x=hour_day2['day_name'], y=hour_day2['event_id'], data=hour_day2, palette='rocket_r', ax=ax2)
    sns.boxplot(x=hour_day2['hour'], y=hour_day2['event_id'], data=hour_day2, palette='rocket_r', ax=ax3)

    
    sns.heatmap(month_day,
                    cmap="rocket_r",  # Choose a sequential colormap
                    annot=True, # Label the maximum value
                    ax=ax4,
                    cbar_ax=cbar_ax2
                   )
    month_day2 = pd.DataFrame(vessel.groupby(['month-year', 'day_name'])['event_id'].count())
    month_day2.reset_index(drop=False, inplace=True)
    sns.boxplot(x=month_day2['day_name'], y=month_day2['event_id'], data=month_day2, palette='rocket_r', ax=ax5)
    sns.boxplot(x=month_day2['month-year'], y=month_day2['event_id'], data=month_day2, palette='rocket_r', ax=ax6)   
    
    # dark_df_time_week.plot(kind='line',title=('Weekly Trend of Standard RDV Events''\n''Between Vessels '),xlabel='Date', ylabel='Count of Events', ax=ax7)
    month_year = pd.DataFrame(vessel.groupby(['month', 'year'])['event_id'].count())
    month_year.reset_index(inplace=True)
    sns.lineplot(data=month_year, x =month_year['month'], 
             y=month_year['event_id'], markers=True ,palette='rocket_r', hue=month_year['year'], ax=ax7)
    # sns.lineplot(data=month_year, x =dark_df_time_day.index.astype(str), y=dark_df_time_day.values, markers=True ,palette='rocket_r', ax=ax7)
    ax7.set_xticklabels(ax7.get_xticklabels(), rotation=45, horizontalalignment='right')

    
    
    ax1.set_title("Count of Standard RDV Events\n" +
              "by Hour and Day of Week\n" +     
             "(Vessel: " + str(vessel_name) + ")")
    ax1.set_ylabel("Hour of Day (UTC)")
    # ax1.set_xlabel("Day of Week")
    # heatmap1_txt="Use this heatmap to identify combinations of hours of the day and days of the week that have a relatively high count of RDV evets."
    ax1.set_xlabel('Day of Week\n(Use this heatmap to identify combinations of hours of the day and days of the week that have a relatively high count of RDV events.)')

    
    # ax2.set_title("Count of Standard RDV Events\n" +
    #           "by Hour and Day of Week\n" +     
    #          "(Vessel: " + str(vessel_name) + ")")
    ax2.set_ylabel("RDV Event Count per Day")
    # ax2.set_xlabel("Day of Week")
    ax2.set_xlabel('Day of Week\n(Use this boxplot to determine the distribution of RDV events across all hours by day.)')
    
    # ax3.set_title("Count of Standard RDV Events\n" +
    #           "by Day of Week and Month-Year\n" +     
    #          "(Vessel: " + str(vessel_name) + ")")
    ax3.set_ylabel("RDV Event Count per Hour")
    # ax3.set_xlabel("Hour of Day")
    ax3.set_xlabel('Hour of Day\n(Use this boxplot to determine the distribution of RDV events by hour.)')
    
    ax4.set_title("Count of Standard RDV Events\n" +
    "by Hour and Day of Week\n" +     
    "(Vessel: " + str(vessel_name) + ")")
    ax4.set_ylabel("RDV Event Count per Day")
    # ax4.set_xlabel("Day of Week")
    ax4.set_xlabel('Day of Week\n(Use this heatmap to identify combinations of month of year and days of the week that have a relatively high count of RDV events.)')  
    ax5.set_ylabel("RDV Event Count per Day/Month")
    # ax5.set_xlabel("Day of Week")
    ax5.set_xlabel('Day of Week\n(Use this boxplot to determine the distribution of RDV events across all months by day.)')
   
    ax6.set_ylabel("RDV Event Count per Month")
    # ax6.set_xlabel("Month-Year")
    ax6.set_xlabel('Month-Year\n(Use this boxplot to determine the distribution of RDV events by month-year.)')
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, horizontalalignment='right')

    ax7.set_title("Count of Standard RDV Events\n" +
              "Week over Week\n" +     
             "(Vessel: " + str(vessel_name) + ")")
    ax7.set_ylabel("RDV Event Count Week over Week")
    # ax6.set_xlabel("Month-Year")
    ax7.set_xlabel('Week\n(Use this plot to determine trends in RDV events.)')
    # ax7.figure(facecolor='#FFF')
    ax7.xaxis.set_major_locator(ticker.MultipleLocator(31))

    dummy_ax.axis('off')
    dummy_ax2.axis('off')
    dummy_ax3.axis('off')
    dummy_ax4.axis('off')
    dummy_ax5.axis('off')
    
    plt.tight_layout()
    
    path = '../plots/Day_Hour_Heatmap_' +  str(vessel_name)+ '.png'
    
    plt.savefig(path, bbox_inches='tight')
    return plt


def get_map(df_events, vessel_name, color_type='time', map_type=None):
    vessel = df_events[(df_events['vessels.vessel_0.name']==vessel_name) | (df_events['vessels.vessel_1.name']==vessel_name) 
                    |(df_events['vessels.vessel_0.name']==vessel_name) | (df_events['vessels.vessel_1.name']==vessel_name)]
    # simulate dataframe in question, generates a warning, ignore it
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    vessel['date'] = pd.to_datetime(vessel['start.time']).dt.date

    df = vessel[['start.point.lon', 'start.point.lat','event_id','event_type','vessels.vessel_0.name','vessels.vessel_1.name']]
    df['RDV Event Start Date'] = vessel['date'].apply(str)
    df.columns = ['Starting Point Longitude','Starting Point Latitude','Event ID','RDV Event Type','Vessel 1 Name','Vessel 2 Name','RDV Event Start Date']
    df = df[['RDV Event Type', 'RDV Event Start Date','Vessel 1 Name','Vessel 2 Name', 'Starting Point Longitude','Starting Point Latitude','Event ID']]


    geometry = [Point(xy) for xy in zip(df['Starting Point Longitude'], df['Starting Point Latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=4326)

    if color_type=='rdv':
        col = 'RDV Event Type'
        cmap=colors.ListedColormap(['#000000','#FF0000'])
    else:
        col = [mdates.date2num(datetime.strptime(i, "%Y-%m-%dT%H:%M:%S+00:00")) for i in vessel['start.time']]
        cmap="rocket_r"
        
    [mean_long, mean_lat] = [df['Starting Point Longitude'].mean(), df['Starting Point Latitude'].mean()]
    
    if map_type=='heatmap':
        map = folium.Map(location=[mean_lat, mean_long], tiles="Cartodb dark_matter", zoom_start=4)
        heat_data = [[point.xy[1][0], point.xy[0][0]] for point in gdf.geometry]
        plugins.HeatMap(heat_data).add_to(map)
        return map
    
    else:
#         import movingpandas as mpd
#         df.set_index('RDV Event Start Date', inplace=True)
#         gdf2 = gpd.GeoDataFrame(df, geometry=geometry, crs=4326)
#         traj = mpd.Trajectory(gdf2, 1)

#         traj.plot(ax = ax)
#         ctx.add_basemap(ax, crs = 'EPSG:4326')
        
#         return plt.show()
        return gdf.explore(column=col,
                    cmap=cmap,
                    legend=False,
                    marker_type='circle_marker',
                    highlight=True)