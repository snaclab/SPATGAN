
# coding: utf-8

# In[10]:



import importlib
import os
import sys
import csv
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

def check_time_format(x):
    if len(x) != 15:
        x = x + '.000000'
    return x
cwd = os.getcwd()
filename = 'skype_eth_5.csv'#'Youtube/youtube_eth_10.csv'#'skype_eth_5.csv'
data = pd.read_csv(cwd+'/GAN/'+filename)
server_ip = '124.123.243.15'#'124.123.243.15'#'180.149.59.78'
client_ip = '192.168.1.133'
data['time'] = data['time'].apply(check_time_format)
data['date'] = data['date'].apply(lambda x: x.replace('-', ' '))
data['times'] = data['date'] + ' ' + data['time']
data['times'] = data['times'].apply(lambda x: datetime.strptime(x, '%Y %m %d %H:%M:%S.%f').timestamp())
data['times'] = data['times'].apply(lambda x: float(str(x)[6:]) * 1e6) # x[0:5] in timestamp are all the same

server = []
client = []
original_size_list = []
original_time_list = []
original_time_seq = []
x = []
x2 = []
time = 0
server_up_cnt = 0
client_up_cnt = 0

prev_times = 0
for index, row in data.iterrows():
    data.at[index, 'size'] = float(row['size'])
    original_val = row['times']
    new_val = original_val - prev_times
    prev_times = row['times']
    data.at[index, 'times'] = new_val

# remove times
#comment this for skype
#data.drop(data[data.times > 250].index, inplace=True)
#data.drop(data[data.times < 0].index, inplace=True)

data.drop('date', axis=1, inplace=True)
data.drop('time', axis=1, inplace=True)
data.drop('number', axis=1, inplace=True)
data.drop('clean_size', axis=1, inplace=True)
# data.to_csv('original.csv')

for index, row in data.iterrows():
    if index < 6727 or index > 7227:
        #if index < 33000 or index > 34000:
        continue
    
    time += row['times']
    if row['src'] == server_ip:
        server.append(row['size'])
        original_size_list.append(row['size'])
        original_time_list.append(row['times'])
        x.append(time)
        if row['size'] > 300:
            server_up_cnt += 1
    elif row['src'] == client_ip:
        client.append(row['size'])
        original_size_list.append(-row['size'])
        original_time_list.append(-row['times'])
        x2.append(time)
        if row['size'] > 300:
            client_up_cnt += 1
    original_time_seq.append(time)

server = np.array(server)
client = np.array(client)
original_size_list = np.array(original_size_list)
original_time_list = np.array(original_time_list)
original_time_seq = np.array(original_time_seq)
plt.figure(figsize=(50, 30))
plt.ylim(-1500, 600)#skype:-1500:600, you:-100:1500
plt.bar(x, server, width=0.8, facecolor='r', edgecolor='g', linewidth=2, label='Server')
plt.bar(x2, -client, width=0.8, facecolor='b', edgecolor='k', linewidth=2, label='Client')
plt.grid(True)
plt.xlabel('Cumulative Time', fontsize=10)
plt.ylabel('Packet Size', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
#for x, y in zip(x, server):
#    plt.text(x, y + 0.2, y, ha='center', va='bottom')
#for x, y in zip(x2, client):
#    plt.text(x, -y - 1.2, y, ha='center', va='bottom')
print(server_up_cnt)
print(client_up_cnt)
#plt.legend(loc=0)
#plt.savefig('original_skype.png')
plt.show()


# In[14]:


for idx in range(1):
    filename = '/GAN/Generate/flow.csv'
    data = pd.read_csv(cwd+filename)

    server = []
    client = []
    x = []
    x2 = []
    time = 0

    server_interval = []
    client_interval = []
    server_time_cnt = 0
    client_time_cnt = 0
    server_up_cnt = 0
    client_up_cnt = 0
    agent = 0
    #with open('./singleGAN/You_Data/generate/'+ str(idx + 1) + '/server_ret_time.txt','r') as f:
    #    for line in f:
    #        server_interval.append(line.strip('\n'))
    #with open('./singleGAN/You_Data/generate/'+ str(idx + 1) + '/client_ret_time.txt','r') as f:
    #    for line in f:
    #        client_interval.append(line.strip('\n'))

    #for index, row in data.iterrows():
    #    if agent == 0 and row['src'] == client_ip:
    #        try:
    #            val = row['times']# + float(client_interval[client_time_cnt])
    #            data.at[index, 'times'] = val
    #            client_time_cnt += 1
    #            agent = 1
    #        except:
    #            print('client_cnt has problem')
    #            pass
    #    elif agent == 1 and row['src'] == server_ip:
    #        val = row['times']# + float(server_interval[server_time_cnt])
    #        data.at[index, 'times'] = val
    #        server_time_cnt += 1
    #        agent = 0
    for index, row in data.iterrows():
        if index < 6000 or index > 6500:
            continue
        time += row['times']
        if row['src'] == server_ip:
            
            server.append(int(row['size']))
            x.append(time)
            if row['size'] > 300:
                server_up_cnt += 1
        elif row['src'] == client_ip:
            #print(row['size'])
            client.append(int(row['size']))
            x2.append(time)
            if row['size'] > 300:
                client_up_cnt += 1

    server = np.array(server)
    client = np.array(client)
    plt.figure(figsize=(50, 30))
    plt.ylim(-1500, 600)
    plt.bar(x, server, width=2, facecolor='r', edgecolor='g', linewidth=2, label='Server')
    plt.bar(x2, -client, width=2, facecolor='b', edgecolor='k', linewidth=2, label='Client')
    plt.grid(True)
    plt.xlabel('Cumulative Time', fontsize=10)
    plt.ylabel('Packet Size', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.rc('font', size=10)
    #for x, y in zip(x, server):
    #    plt.text(x, y + 0.2, y, ha='center', va='bottom')
    #or x, y in zip(x2, client):
    #   plt.text(x, -y - 1.2, y, ha='center', va='bottom')
    print(str(idx + 1))
    print(server_up_cnt)
    print(client_up_cnt)
    #plt.legend(loc=0)
    plt.savefig(cwd+'/GAN/Generate/traffic.png')
    plt.show()


# In[ ]:

'''
for idx in range(0):
    filename = 'You_Data/generate_TimeAvg/' + str(idx + 1) + '/flow.csv'
    data = pd.read_csv(filename)
    server = []
    client = []
    x = []
    x2 = []
    time = 0

    server_interval = []
    client_interval = []
    server_time_cnt = 0
    client_time_cnt = 0
    server_up_cnt = 0
    client_up_cnt = 0
    agent = 0
    with open('You_Data/generate_TimeAvg/' + str(idx + 1) + '/server_ret_time.txt','r') as f:
        for line in f:
            server_interval.append(line.strip('\n'))
    with open('You_Data/generate_TimeAvg/' + str(idx + 1) + '/client_ret_time.txt','r') as f:
        for line in f:
            client_interval.append(line.strip('\n'))

    for index, row in data.iterrows():
        if agent == 0 and row['src'] == client_ip:
            try:
                val = row['times'] + float(client_interval[client_time_cnt])
                data.at[index, 'times'] = val
                client_time_cnt += 1
                agent = 1
            except:
                print('client_cnt has problem')
                pass
        elif agent == 1 and row['src'] == server_ip:
            val = row['times'] + float(server_interval[server_time_cnt])
            data.at[index, 'times'] = val
            server_time_cnt += 1
            agent = 0
    for index, row in data.iterrows():
        if index < 13000 or index > 14000:
            continue
        time += row['times']
        if row['src'] == server_ip:
            server.append(int(row['size']))
            x.append(time)
            if row['size'] > 300:
                server_up_cnt += 1
        elif row['src'] == client_ip:
            client.append(int(row['size']))
            x2.append(time)
            if row['size'] > 300:
                client_up_cnt += 1


    server = np.array(server)
    client = np.array(client)
    plt.figure(figsize=(100, 50))
    plt.ylim(-1450, 1450)
    plt.bar(x, server, width=2, facecolor='r', edgecolor='g', linewidth=2, label='Server')
    plt.bar(x2, -client, width=2, facecolor='b', edgecolor='k', linewidth=2, label='Client')
    plt.grid(True)
    plt.xlabel('Cumulative Time', fontsize=100)
    plt.ylabel('Packet Size', fontsize=100)
    plt.xticks(fontsize=100)
    plt.yticks(fontsize=100)
    #for x, y in zip(x, server):
    #    plt.text(x, y + 0.2, y, ha='center', va='bottom')
    #for x, y in zip(x2, client):
    #    plt.text(x, -y - 1.2, y, ha='center', va='bottom')
    print(str(idx + 1))
    print(server_up_cnt)
    print(client_up_cnt)
    #plt.legend(loc=0)
    plt.savefig('You_Data/generate_TimeAvg/' + str(idx + 1) + '/traffic_new.png')
    plt.show()


# In[10]:


for idx in range(8, 10):
    filename = 'You_Data/poisson/' + str(idx + 1) + '/flow.csv'
    data = pd.read_csv(filename)
    server_ip = '180.149.59.78'#'124.123.243.15'#'180.149.59.78'
    client_ip = '192.168.1.133'

    server = []
    client = []
    x = []
    x2 = []
    time = 0

    server_interval = []
    client_interval = []
    server_time_cnt = 0
    client_time_cnt = 0
    server_up_cnt = 0
    client_up_cnt = 0
    agent = 0
    with open('You_Data/poisson/' + str(idx + 1) + '/server_ret_time.txt','r') as f:
        for line in f:
            server_interval.append(line.strip('\n'))
    with open('You_Data/poisson/' + str(idx + 1) + '/client_ret_time.txt','r') as f:
        for line in f:
            client_interval.append(line.strip('\n'))

    for index, row in data.iterrows():
        if agent == 0 and row['src'] == 'client':
            val = row['times'] + float(client_interval[client_time_cnt])
            data.at[index, 'times'] = val
            client_time_cnt += 1
            agent = 1
        elif agent == 1 and row['src'] == 'server':
            val = row['times'] + float(server_interval[server_time_cnt])
            data.at[index, 'times'] = val
            server_time_cnt += 1
            agent = 0
    for index, row in data.iterrows():
        if index < 14000 or index > 15000-500:
            continue
        time += row['times']
        if row['src'] == 'server':
            server.append(int(row['size']))
            x.append(time)
            if row['size'] > 300:
                server_up_cnt += 1
        elif row['src'] == 'client':
            client.append(int(row['size']))
            x2.append(time)
            if row['size'] > 300:
                client_up_cnt += 1

    server = np.array(server)
    client = np.array(client)
    plt.figure(figsize=(50, 30))
    plt.ylim(-100, 1500)
    plt.bar(x, server, width=2, facecolor='r', edgecolor='g', linewidth=2, label='Server')
    plt.bar(x2, -client, width=2, facecolor='b', edgecolor='k', linewidth=2, label='Client')
    plt.grid(True)
    plt.xlabel('Cumulative Time', fontsize=100)
    plt.ylabel('Packet Size', fontsize=100)
    plt.xticks(fontsize=80)
    plt.yticks(fontsize=100)
    plt.rc('font', size=100)
    #for x, y in zip(x, server):
    #    plt.text(x, y + 0.2, y, ha='center', va='bottom')
    #for x, y in zip(x2, client):
    #    plt.text(x, -y - 1.2, y, ha='center', va='bottom')
    print(str(idx + 1))
    print(server_up_cnt)
    print(client_up_cnt)
    #plt.legend(loc=0)
    plt.savefig('You_Data/poisson/' + str(idx + 1) + '/traffic_new4.png')
    plt.show()


# In[11]:


for idx in range(5, 10):
    filename = 'You_Data/normal/' + str(idx + 1) + '/flow.csv'
    data = pd.read_csv(filename)
    server_ip = '180.149.59.78'#'124.123.243.15'#'180.149.59.78'
    client_ip = '192.168.1.133'#'192.168.1.133'

    server = []
    client = []
    x = []
    x2 = []
    time = 0

    server_interval = []
    client_interval = []
    server_time_cnt = 0
    client_time_cnt = 0
    server_up_cnt = 0
    client_up_cnt = 0
    agent = 0
    with open('You_Data/normal/' + str(idx + 1) + '/server_ret_time.txt','r') as f:
        for line in f:
            server_interval.append(line.strip('\n'))
    with open('You_Data/normal/' + str(idx + 1) + '/client_ret_time.txt','r') as f:
        for line in f:
            client_interval.append(line.strip('\n'))

    for index, row in data.iterrows():
        if agent == 0 and row['src'] == 'client':
            val = row['times'] + float(client_interval[client_time_cnt])
            data.at[index, 'times'] = val
            client_time_cnt += 1
            agent = 1
        elif agent == 1 and row['src'] == 'server':
            val = row['times'] + float(server_interval[server_time_cnt])
            data.at[index, 'times'] = val
            server_time_cnt += 1
            agent = 0
    for index, row in data.iterrows():
        if index < 13500 or index > 14000:
            continue
        time += row['times']
        if row['src'] == 'server':
            server.append(int(row['size']))
            x.append(time)
            if row['size'] > 300:
                server_up_cnt += 1
        elif row['src'] == 'client':
            client.append(int(row['size']))
            x2.append(time)
            if row['size'] > 300:
                client_up_cnt += 1

    server = np.array(server)
    client = np.array(client)
    plt.figure(figsize=(50, 30))
    plt.ylim(-100, 1500)
    plt.bar(x, server, width=2, facecolor='r', edgecolor='g', linewidth=2, label='Server')
    plt.bar(x2, -client, width=2, facecolor='b', edgecolor='k', linewidth=2, label='Client')
    plt.grid(True)
    plt.xlabel('Cumulative Time', fontsize=100)
    plt.ylabel('Packet Size', fontsize=100)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=100)
    plt.rc('font', size=100)
    #for x, y in zip(x, server):
    #    plt.text(x, y + 0.2, y, ha='center', va='bottom')
    #for x, y in zip(x2, client):
    #    plt.text(x, -y - 1.2, y, ha='center', va='bottom')
    print(str(idx + 1))
    print(server_up_cnt)
    print(client_up_cnt)
    #plt.legend(loc=0)
    plt.savefig('You_Data/normal/' + str(idx + 1) + '/traffic_new4.png')
    plt.show()


# In[ ]:


for idx in range(10):
    filename = 'Data/generate/' + str(idx + 1) + '/flow.csv'
    data = pd.read_csv(filename)
    size_list =[]
    time_list = []
    time_seq = []
    time = 0

    server_interval = []
    client_interval = []
    server_time_cnt = 0
    client_time_cnt = 0
    server_up_cnt = 0
    client_up_cnt = 0
    agent = 0
    with open('Data/generate/' + str(idx + 1) + '/server_ret_time.txt','r') as f:
        for line in f:
            server_interval.append(line.strip('\n'))
    with open('Data/generate/' + str(idx + 1) + '/client_ret_time.txt','r') as f:
        for line in f:
            client_interval.append(line.strip('\n'))

    for index, row in data.iterrows():
        if agent == 0 and row['src'] == client_ip:
            try:
                val = row['times'] + float(client_interval[client_time_cnt])
                data.at[index, 'times'] = val
                client_time_cnt += 1
                agent = 1
            except:
                print('client_cnt has problem')
                pass
        elif agent == 1 and row['src'] == server_ip:
            val = row['times'] + float(server_interval[server_time_cnt])
            data.at[index, 'times'] = val
            server_time_cnt += 1
            agent = 0
    for index, row in data.iterrows():
#         if index < 13000 or index > 14000:
#             continue
        time += row['times']
        if row['src'] == server_ip:
            size_list.append(int(row['size']))
            time_list.append(row['times'])
            time_seq.append(time)
            if row['size'] > 300:
                server_up_cnt += 1
        elif row['src'] == client_ip:
            size_list.append(int(row['size']))
            time_list.append(row['times'])
            time_seq.append(time)
            if row['size'] > 300:
                client_up_cnt += 1
    size_list = np.array(size_list)
    time_list = np.array(time_list)
    time_seq = np.array(time_seq)
#     time_seq /= 1e6
    fig, tu = plt.subplots(figsize=(100, 50))
    tu.set_ylim([-1450, 1450])
    tu.set_title('traffic', fontweight="bold", size=50) # Title
    tu.set_ylabel('size', fontsize = 50.0) # Y label
    tu.set_xlabel('Time /s', fontsize = 50) # X label
#     tu.plot(time_list, size_list, label='generate')
    tu.scatter(time_list, size_list, label='generate')
    tu.scatter(original_time_list, original_size_list, label='original')
    tu.set(ylabel='size', xlabel='Time /s', title='traffic')
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    tu.legend()
#     plt.figure(figsize=(100, 50))
#     plt.ylim(-1450, 1450)
#     plt.bar(x, server, width=2, facecolor='r', edgecolor='g', linewidth=2, label='Server')
#     plt.bar(x2, -client, width=2, facecolor='b', edgecolor='k', linewidth=2, label='Client')
    plt.grid(True)
#     for x, y in zip(x, server):
#         plt.text(x, y + 0.2, y, ha='center', va='bottom')
#     for x, y in zip(x2, client):
#         plt.text(x, -y - 1.2, y, ha='center', va='bottom')
    print(str(idx + 1))
    print(server_up_cnt)
    print(client_up_cnt)
#     plt.legend(loc=0)
    plt.savefig('Data/generate/' + str(idx + 1) + '/plot_version_3.png')
    plt.show()


# In[12]:


for idx in range(1):
    idx = 19
    filename = 'Data/generate/' + str(idx + 1) + '/flow.csv'
    data = pd.read_csv(filename)
    server = []
    client = []
    x = []
    x2 = []
    time = 0

    server_interval = []
    client_interval = []
    server_time_cnt = 0
    client_time_cnt = 0
    server_up_cnt = 0
    client_up_cnt = 0
    agent = 0
    with open('Data/generate/'+ str(idx + 1) + '/server_ret_time.txt','r') as f:
        for line in f:
            server_interval.append(line.strip('\n'))
    with open('Data/generate/'+ str(idx + 1) + '/client_ret_time.txt','r') as f:
        for line in f:
            client_interval.append(line.strip('\n'))

    for index, row in data.iterrows():
        if agent == 0 and row['src'] == client_ip:
            try:
                val = row['times'] + float(client_interval[client_time_cnt])
                data.at[index, 'times'] = val
                client_time_cnt += 1
                agent = 1
            except:
                print('client_cnt has problem')
                pass
        elif agent == 1 and row['src'] == server_ip:
            val = row['times'] + float(server_interval[server_time_cnt])
            data.at[index, 'times'] = val
            server_time_cnt += 1
            agent = 0
    for index, row in data.iterrows():
        if index < 8767 or index > 9267:
            continue
        time += row['times']
        if row['src'] == server_ip:
            server.append(int(row['size']))
            x.append(time)
            if row['size'] > 300:
                server_up_cnt += 1
        elif row['src'] == client_ip:
            client.append(int(row['size']))
            x2.append(time)
            if row['size'] > 300:
                client_up_cnt += 1

    server = np.array(server)
    client = np.array(client)
    plt.figure(figsize=(100, 50))
    plt.ylim(-1500, 1500)
    plt.bar(x, server, width=2, facecolor='r', edgecolor='g', linewidth=2, label='Server')
    plt.bar(x2, -client, width=2, facecolor='b', edgecolor='k', linewidth=2, label='Client')
    plt.grid(True)
    for x, y in zip(x, server):
        plt.text(x, y + 0.2, y, ha='center', va='bottom')
    for x, y in zip(x2, client):
        plt.text(x, -y - 1.2, y, ha='center', va='bottom')
    print(str(idx + 1))
    print(server_up_cnt)
    print(client_up_cnt)
    plt.legend(loc=0)
    plt.savefig('Data/generate/traffic.png')
    plt.show()
'''
