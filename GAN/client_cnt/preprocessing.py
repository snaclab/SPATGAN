#coding=utf-8
# standard library
import gc
import importlib
import os
import sys
import pickle
import socket
import statistics
import struct
import time
import threading
from datetime import datetime

# pip3 library
import psutil
import scipy
import numpy as np
import pandas as pd
import sklearn.cluster as cluster
from gan_func import *

server_ip = '192.168.1.133'
client_ip = '124.123.243.15'

# Load engineered dataset from EDA section
data = pd.read_csv('skype_eth_5.csv')

# different src ip addr && dst ip addr
mask1 = data['src'] == server_ip
mask2 = data['src'] == client_ip
dataSample = data.loc[(mask1 | mask2)]
mask3 = dataSample['dst'] == client_ip
mask4 = dataSample['dst'] == server_ip
dataSample = dataSample.loc[(mask3 | mask4)]
# dataSample = data.loc[data['dst'] == server_ip]
dataSample['time'] = dataSample['time'].apply(check_time_format)
dataSample['date'] = dataSample['date'].apply(lambda x: x.replace('-', ' '))
dataSample['times'] = dataSample['date'] + ' ' + dataSample['time']
dataSample['times'] = dataSample['times'].apply(lambda x: datetime.strptime(x, '%Y %m %d %H:%M:%S.%f').timestamp())
dataSample['times'] = dataSample['times'].apply(lambda x: float(str(x)[6:]) * 1e6) # x[0:5] in timestamp are all the same

# set float format
# pd.options.display.float_format = '{:,.6f}'.format
# print(dataSample)

# calculate relative value
prev_times = 0
for index, row in dataSample.iterrows():
    dataSample.at[index, 'size'] = float(row['size'])
    original_val = row['times']
    new_val = original_val - prev_times
    prev_times = row['times']
    dataSample.at[index, 'times'] = new_val

# filter src && dst
dataSample = dataSample.loc[dataSample['src'] == server_ip]
dataSample = dataSample.loc[dataSample['dst'] == client_ip]

# remove times
dataSample.drop(dataSample[dataSample.times > 25000].index, inplace=True)
dataSample.drop(dataSample[dataSample.times < 0].index, inplace=True)

# save traffic
# dataSample.to_csv('tmp_traffic.csv')

# remove unused column
dataSample.drop('date', axis=1, inplace=True)
dataSample.drop('src', axis=1, inplace=True)
dataSample.drop('dst', axis=1, inplace=True)
dataSample.drop('time', axis=1, inplace=True)
dataSample.drop('number', axis=1, inplace=True)
dataSample.drop('clean_size', axis=1, inplace=True)

dataSample.to_csv('preprocessing_traffic.csv')

# normalize
print(dataSample.mean(), dataSample.std())
normalized_dataSample = (dataSample - dataSample.mean()) / dataSample.std()                     # Zero-mean normalization
data_cols = ['times', 'size']
# normalized_dataSample.to_csv('normal.csv')

# Add KMeans generated classes to fraud data - see classification section for more details on this
train = normalized_dataSample.copy()
labels = cluster.KMeans(n_clusters=2, random_state=0).fit_predict(train[data_cols])
print(pd.DataFrame([[np.sum(labels == i)] for i in np.unique(labels)], columns=['count'], index=np.unique(labels)))
print('Data preprocessing finish')