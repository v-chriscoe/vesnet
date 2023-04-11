import pandas as pd
import numpy as np


def get_stats(df_events, vessel_name):
    vessel = df_events[(df_events['vessels.vessel_0.name']==vessel_name) | (df_events['vessels.vessel_1.name']==vessel_name) 
                    |(df_events['vessels.vessel_0.name']==vessel_name) | (df_events['vessels.vessel_1.name']==vessel_name)]
    
    count_rdv = len(vessel)
    mean_h = pd.to_datetime(vessel['start.time']).dt.hour.mean()
    std_h = pd.to_datetime(vessel['start.time']).dt.hour.std()
    mode_d = pd.to_datetime(vessel['start.time']).dt.day_name().mode()
    
    # Create list of vessel names
    ves1 = vessel[['vessels.vessel_0.name','event_id']].rename(columns={'vessels.vessel_0.name': "vessels.name"})
    ves2 = vessel[['vessels.vessel_1.name','event_id']].rename(columns={'vessels.vessel_1.name': "vessels.name"})
    ves = pd.concat([ves1, ves2], axis=0)
    out_count = ves[['vessels.name','event_id']].groupby(['vessels.name'])['event_id'] \
                                 .count() \
                                 .reset_index(name='count') \
                                 .sort_values(['count'], ascending=False) \
                                 .head(11)
    
    return [count_rdv, mean_h, std_h, mode_d, out_count]