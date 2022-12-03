#coding=utf-8
import importlib
import os
import sys
import csv
from datetime import datetime
import numpy as np
import pandas as pd

def check_time_format(x):
    if len(x) != 15:
        x = x + '.000000'
    return x

def cnt_list(agent_list, agent_cnt, agent_check):
    if agent_cnt in agent_list:
        agent_list[agent_cnt] += 1
    else:
        agent_list[agent_cnt] = 1
    return agent_check + 1

filename = 'skype_eth_5.csv'
data = pd.read_csv(filename)
server_ip = '192.168.1.133'
client_ip = '124.123.243.15'
new_name = filename[:-5] + 'cnt'
ser_list = {}
cli_list = {}
data['time'] = data['time'].apply(check_time_format)
data['date'] = data['date'].apply(lambda x: x.replace('-', ' '))
data['times'] = data['date'] + ' ' + data['time']
data['times'] = data['times'].apply(lambda x: datetime.strptime(x, '%Y %m %d %H:%M:%S.%f').timestamp())
data['times'] = data['times'].apply(lambda x: float(str(x)[6:]) * 1e6) # x[0:5] in timestamp are all the same
print(new_name)

for i in range(10):
    ser_list[i + 1] = 0
    cli_list[i + 1] = 0

total_cnt = 0
server_check = 0
client_check = 0

with open(new_name + '.txt', 'w') as text_file:

    server_cnt = 0
    client_cnt = 0
    server_time = 0
    client_time = 0
    agent = 0
    agent_src = ['server', 'client']
    size_cnt = 0
    final_index = 0
    past_cnt = 0
    for index, row in data.iterrows():
        if agent == 0 and row['src'] == server_ip:
            server_cnt += 1
            server_time = row['times']
        elif agent == 1 and row['src'] == client_ip:
            client_cnt += 1
            client_time = row['times']
        elif agent == 0 and row['src'] == client_ip:
            # text_file.write('%s %s %s %s %s\n' % (str(index).ljust(5), agent_src[agent], str(server_cnt).ljust(2), str(past_cnt).ljust(2), str(row['times'] - server_time)))
            if (row['times'] - server_time) < 30000:
                text_file.write('%s %s\n' % (str(server_cnt), str(row['times'] - server_time))) #, str(past_cnt)
            server_check = cnt_list(ser_list, server_cnt, server_check)
            past_cnt = server_cnt
            size_cnt = 0
            agent = 1
            server_cnt = 0
            client_cnt = 1
            server_time = 0
            client_time = row['times']
        elif agent == 1 and row['src'] == server_ip:
            # text_file.write('%s %s %s %s %s\n' % (str(index).ljust(5), agent_src[agent], str(client_cnt).ljust(2), str(past_cnt).ljust(2), str(row['times'] - client_time)))
            # if (row['times'] - client_time) < 30000:
                # text_file.write('%s %s %s\n' % (str(client_cnt), str(row['times'] - client_time), str(past_cnt)))
            client_check = cnt_list(cli_list, client_cnt, client_check)
            past_cnt = client_cnt
            size_cnt = 0
            agent = 0
            server_cnt = 1
            client_cnt = 0
            server_time = row['times']
            client_time = 0
        size_cnt += row['size']
        total_cnt += 1
        final_index = index
    # if agent == 0:
    #     text_file.write('%s %s %s %s\n' % (str(final_index + 1).ljust(5), agent_src[agent], str(server_cnt).ljust(2), str(server_time - client_time)))
    #     # text_file.write('%s %s\n' % (str(row['times'] - server_time), str(server_cnt)))
    #     cnt_list(ser_list, server_cnt, server_check)
    # else:
    #     text_file.write('%s %s %s %s\n' % (str(final_index + 1).ljust(5), agent_src[agent], str(client_cnt).ljust(2), str(client_time - server_time)))
    #     # text_file.write('%s %s\n' % (str(row['times'] - client_time), str(client_cnt)))
    #     cnt_list(cli_list, client_cnt, client_check)

with open(new_name + '.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(' ') for line in stripped if line)
    
    with open('test.csv', 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('cnt', 'times'))
        writer.writerows(lines)

print('Server')
for key in ser_list:
    print(key, ser_list[key], format(ser_list[key] / server_check, '0.5f'))
print(server_check)

print('Client')
for key in cli_list:
    print(key, cli_list[key], format(cli_list[key] / client_check, '0.5f'))
print(client_check)