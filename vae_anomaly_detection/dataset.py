import torch
import numpy as np
import pandas as pd
import math
from torch.utils.data import Dataset, TensorDataset

# the data has 18300 rows, we will separate into 1830 segments with 10 rows each
data = pd.read_csv('ais_data.csv')
df_vae = data[['time','long', 'lat', 'sog', 'cog']]
df_vae['time'] = pd.to_datetime(df_vae['time'])
transpose = df_vae.T

#Trajectory Feature Extraction
#the distance between x and x-1
df_vae['x-xi'] = df_vae['long'].diff()
#the distance between x and x-1
df_vae['y-yi'] = df_vae['lat'].diff()
#the length of the vector |R|
df_vae['R']= np.sqrt(df_vae['x-xi']**2+df_vae['y-yi']**2)
#cos of vector |R| to x axis
df_vae['cosr'] = df_vae['x-xi']/df_vae['R']
#sin of vector |R| to x axis
df_vae['sinr'] = df_vae['y-yi']/df_vae['R']
#transpose to a row series
sog_list = df_vae['sog'].T
cog_list = df_vae['cog'].T
#temporary list for iterary calculation
speed_list=[]
cosc_list=[]
sinc_list=[]
#calculating average speed = (v(i-1)+v(i))/2
for y in range(sog_list.size-1):
    speed_list.append((sog_list[y]+sog_list[y+1])/2)
#calculating sin and cos of course over ground
for x in range(cog_list.size):
    cosc_list.append(math.cos(math.radians(cog_list[x])))
    sinc_list.append(math.sin(math.radians(cog_list[x])))
speed_list.insert(0,"NaN")
df_vae['speed'] = pd.Series(speed_list)
df_vae['cosc']= pd.Series(cosc_list)
df_vae['sinc']= pd.Series(sinc_list)
#calculate time difference between
df_vae['time_diff'] = df_vae['time'].diff().astype('timedelta64[s]')
#Separate dates for future plotting
train_dates = pd.to_datetime(df_vae['time'])

df_vae = df_vae.drop(['long', 'lat', 'sog', 'cog', 'x-xi', 'y-yi'], axis=1)
df_vae['speed'] = df_vae['speed'].astype(float, errors = 'raise')
df_vae = df_vae.replace(np.nan,0)
#df_vae = np.array_split(df_vae,1830)

# print(df_vae.head())
# print(train_dates.tail(15))

train_cols = list(df_vae)[1:8]
# print(train_cols)
df_for_training = df_vae[train_cols].astype(float)
# print(df_for_training)

# tensor_data = torch.tensor(df_for_training.values)
# print(tensor_data)
def rand_dataset() -> Dataset:
    return TensorDataset(torch.tensor(df_for_training.values))