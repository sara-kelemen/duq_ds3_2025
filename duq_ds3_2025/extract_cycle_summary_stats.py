''' Extract summary statistics for cycle length for each hubid, split by missingness '''

import pandas as pd
from add_pcos_label import AddPCOS

# Load data
df = pd.read_csv('data/missingness_df.csv')

# Add missingness column
df['missingness'] = df['missing_around_spike_6_days'] > 5

# Split data based on missingness
df_missing_false = df[df['missingness'] == False]
df_missing_true = df[df['missingness'] == True]

# Process missingness == False
hubids = df_missing_false['hub_id'].unique().tolist()
cycle_min = []
cycle_max = []
cycle_median = []
cycle_mean = []
cycle_range = []
cycle_std = []
number_of_cycles = []
groups = []

for hubid in hubids:
    group = AddPCOS().mapid_to_group(hubid)
    groups.append(group)
    df_temp = df_missing_false[df_missing_false['hub_id'] == hubid]
    cycle_min.append(df_temp['cycle_length'].min())
    cycle_max.append(df_temp['cycle_length'].max())
    cycle_median.append(df_temp['cycle_length'].median())
    cycle_mean.append(df_temp['cycle_length'].mean())
    cycle_range.append(df_temp['cycle_length'].max() - df_temp['cycle_length'].min())
    cycle_std.append(df_temp['cycle_length'].std())
    number_of_cycles.append(len(df_temp))

df_out_false = pd.DataFrame({
    'hub_id': hubids,
    'pat_cat_map': groups,
    'cycle_min': cycle_min,
    'cycle_max': cycle_max,
    'cycle_median': cycle_median,
    'cycle_mean': cycle_mean,
    'cycle_range': cycle_range,
    'cycle_std': cycle_std,
    'num_cycles': number_of_cycles
})

df_out_false.to_csv('hubid_cycle_features_false.csv', index=False)
print("hubid_cycle_features_false.csv")

# Reset lists for next loop
cycle_min = []
cycle_max = []
cycle_median = []
cycle_mean = []
cycle_range = []
cycle_std = []
number_of_cycles = []
groups = []

# Process missingness == True
hubids = df_missing_true['hub_id'].unique().tolist()

for hubid in hubids:
    group = AddPCOS().mapid_to_group(hubid)
    groups.append(group)
    df_temp = df_missing_true[df_missing_true['hub_id'] == hubid]
    cycle_min.append(df_temp['cycle_length'].min())
    cycle_max.append(df_temp['cycle_length'].max())
    cycle_median.append(df_temp['cycle_length'].median())
    cycle_mean.append(df_temp['cycle_length'].mean())
    cycle_range.append(df_temp['cycle_length'].max() - df_temp['cycle_length'].min())
    cycle_std.append(df_temp['cycle_length'].std())
    number_of_cycles.append(len(df_temp))

df_out_true = pd.DataFrame({
    'hub_id': hubids,
    'pat_cat_map': groups,
    'cycle_min': cycle_min,
    'cycle_max': cycle_max,
    'cycle_median': cycle_median,
    'cycle_mean': cycle_mean,
    'cycle_range': cycle_range,
    'cycle_std': cycle_std,
    'num_cycles': number_of_cycles
})

df_out_true.to_csv('hubid_cycle_features_true.csv', index=False)
print("saved hubid_cycle_features_true.csv")
