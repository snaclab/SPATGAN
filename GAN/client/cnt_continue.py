#coding=utf-8
import importlib
import os
import sys
import csv
import numpy as np
import pandas as pd

filename = 'skype_eth_5.csv'
data = pd.read_csv(filename)
server_ip = '192.168.1.133'
client_ip = '124.123.243.15'
new_name = filename[:-5] + 'cnt'
print(new_name)

with open(new_name + '.txt', 'w') as text_file:

    server_cnt = 0
    client_cnt = 0
    agent = 0
    agent_src = ['server', 'client']
    for index, row in data.iterrows():
        if agent == 0 and row['src'] == server_ip:
            server_cnt += 1
        elif agent == 1 and row['src'] == client_ip:
            client_cnt += 1
        elif agent == 0 and row['src'] == client_ip:
            text_file.write('%s %s %s\n' % (str(index).ljust(5), agent_src[agent], str(server_cnt).ljust(2)))
            agent = 1
            server_cnt = 0
            client_cnt = 1
        elif agent == 1 and row['src'] == server_ip:
            text_file.write('%s %s %s\n' % (str(index).ljust(5), agent_src[agent], str(client_cnt).ljust(2)))
            agent = 0
            server_cnt = 1
            client_cnt = 0