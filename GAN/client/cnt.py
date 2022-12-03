#coding=utf-8
import importlib
import os
import sys
import csv
import numpy as np
import pandas as pd

# for idx in range(6):
#     data = pd.read_csv('skype_eth_' + str(idx + 1) + '_udp.csv')
#     if idx > 0:
#         data.drop(data.index[:1], inplace=True)
#     df = data.to_csv('combine_skype_eth_udp.csv', index=False, mode='a+')

data = pd.read_csv('skype_eth_6.csv')
src_ip = data['src']
src_list = {}
total_cnt = 0

ip_list = []

for items in src_ip.iteritems():
    if items[1] not in src_list:
        src_list[items[1]] = {}
        dst_ip = data.loc[data['src'] == items[1]]
        dst_list = {}
        for row in dst_ip.iterrows():
            if row[1]['dst'] not in dst_list:
                dst_list[row[1]['dst']] = 1
            else:
                dst_list[row[1]['dst']] += 1
            total_cnt += 1
        src_list[items[1]] = dst_list

check_cnt = 0
check_list = {}
for src in src_list:
    for dst in src_list[src]:
        if src_list[src][dst] > 999:
            print('%s to %s has: %s' % (src.ljust(15), dst.ljust(15), str(src_list[src][dst]).ljust(6)))
            try:
                print('%s to %s has: %s' % (dst.ljust(15), src.ljust(15), str(src_list[dst][src]).ljust(6)))
            except:
                pass
        check_cnt += src_list[src][dst]

print('Total_cnt: %d' % total_cnt)
print('Check_cnt: %d' % check_cnt)