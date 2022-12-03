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
import tqdm
import matplotlib
import numpy as np
import pandas as pd
import sklearn.cluster as cluster
import xgboost as xgb

# custom functions
import matplotlib.pyplot as plt
import gan_func
importlib.reload(gan_func)  # For reloading after making changes
from gan_func import *
plt.style.use('ggplot')     # select plt style

rand_dim = 32
data_dim = 2
label_dim = 1
base_n_count = 128

server_ip = '192.168.1.133'
client_ip = '124.123.243.15'

def init_data(agent, csv):

    global server_ip, client_ip
    # Load engineered dataset from EDA section
    data = pd.read_csv(csv)

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

    # calculate relative value
    prev_times = 0
    for index, row in dataSample.iterrows():
        dataSample.at[index, 'size'] = float(row['size'])
        original_val = row['times']
        new_val = original_val - prev_times
        prev_times = row['times']
        dataSample.at[index, 'times'] = new_val

    # remove times smaller than zero
    dataSample.drop(dataSample[dataSample.times > 25000].index, inplace=True)
    dataSample.drop(dataSample[dataSample.times < 0].index, inplace=True)

    # filter src && dst
    if agent == 'server':
        dataSample = dataSample.loc[dataSample['src'] == server_ip]
        dataSample = dataSample.loc[dataSample['dst'] == client_ip]
    else:
        dataSample = dataSample.loc[dataSample['src'] == client_ip]
        dataSample = dataSample.loc[dataSample['dst'] == server_ip]

    # remove unused column
    dataSample.drop('date', axis=1, inplace=True)
    dataSample.drop('time', axis=1, inplace=True)
    dataSample.drop('src', axis=1, inplace=True)
    dataSample.drop('dst', axis=1, inplace=True)
    dataSample.drop('number', axis=1, inplace=True)
    dataSample.drop('clean_size', axis=1, inplace=True)

    # normalize
    normalized_dataSample = (dataSample - dataSample.mean()) / dataSample.std()                     # Zero-mean normalization
    data_cols = ['times', 'size']

    # Add KMeans generated classes to fraud data - see classification section for more details on this
    train = normalized_dataSample.copy()
    labels = cluster.KMeans(n_clusters=2, random_state=0).fit_predict(train[data_cols])
    print(pd.DataFrame([[np.sum(labels == i)] for i in np.unique(labels)], columns=['count'], index=np.unique(labels)))
    print('Data preprocessing finish')

    # run time cnt
    dataSample_2 = pd.read_csv(agent + '.csv')
    normalized_dataSample_2 = (dataSample_2 - dataSample_2.mean()) / dataSample_2.std()
    data_cols_2 = ['times', 'cnt']
    train_2 = normalized_dataSample_2.copy()
    labels_2 = cluster.KMeans(n_clusters=2, random_state=0).fit_predict(train_2[data_cols_2])
    print(pd.DataFrame([[np.sum(labels_2 == i)] for i in np.unique(labels_2)], columns=['count'], index=np.unique(labels_2)))
    print('Data preprocessing finish')
    
    return train, dataSample, train_2, dataSample_2

def generate_noise(rand_dim, loop_size):
    noise_pool = []
    for i in range(loop_size):
        tmp = np.random.normal(size=(1, rand_dim))
        noise_pool.append(tmp)
    return noise_pool

def test_model(agent, train, dataSample):

    global rand_dim, data_dim, label_dim, base_n_count
    path = '../Desktop/' + agent + '_cnt_new/WCGAN_generator_model_weights_step_25000.h5'

    # Define four network models
    print('Load model && weights')
    generator_model, discriminator_model, combined_model = define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count, type='Wasserstein')
    try:
        generator_model.load_weights(path)
    except Exception as e:
        print(e)
        print('Wrong Path!')
    cnt = 0
    while True:
        train_label = get_data_batch(train, 64, seed=cnt)
        noise_pool = generate_noise(rand_dim, 1000)
        z = noise_pool[cnt - 1]
        size, time, num = input().split()
        if size == 'q':
            break
        for i in range(int(num)):
            train_label[i][0] = (float(size) - dataSample.mean()[0]) / dataSample.std()[0]
            train_label[i][1] = (float(time) - dataSample.mean()[1]) / dataSample.std()[1]
        train_label[0][0] = (float(size) - dataSample.mean()[0]) / dataSample.std()[0]
        train_label[0][1] = (float(time) - dataSample.mean()[1]) / dataSample.std()[1]
        labels = train_label[:, -1:]
        g_z = generator_model.predict([z, labels])  

        # denormailize the generator's result
        result = g_z.copy()
        result[:, 0] *= dataSample.std()[0]
        result[:, 0] += dataSample.mean()[0]

        result[:, 1] *= dataSample.std()[1]
        result[:, 1] += dataSample.mean()[1]

        # save respond packet to tmp.txt
        print(str(result[:, 0]) + ' ' + str(result[:, 1]))
        cnt += 1

def generate_times(train_2, dataSample_2, seed, noise_pool, generator_model):
    cnt = 0
    while True:
        x = get_data_batch(train_2, 64, seed=seed)
        z = noise_pool[seed - 1]
        labels = x[:, :1]
        result = generator_model.predict([z, labels])
        result[:, 1] *= dataSample_2.std()[1]
        result[:, 1] += dataSample_2.mean()[1]
        cnt = int(result[:, 0][0])
        times = int(result[:, 1][0])
        seed += 1
        if cnt > 0 and times > 0:
            break
    
    return cnt, times, seed

def main(agent, train, dataSample, train_2, dataSample_2, loop_size):

    global rand_dim, data_dim, label_dim, base_n_count
    path = '../Desktop/' + agent + '/WCGAN_generator_model_weights_step_25000.h5'
    path_cnt = '../Desktop/' + agent + '_cnt_new/WCGAN_generator_model_weights_step_25000.h5'

    # Define four network models
    print('Load model && weights')
    generator_model, discriminator_model, combined_model = define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count, type='Wasserstein')
    generator_model_cnt, discriminator_model_cnt, combined_model_cnt = define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count, type='Wasserstein')
    try:
        generator_model.load_weights(path)
        generator_model_cnt.load_weights(path_cnt)
    except Exception as e:
        print(e)
        print('Wrong Path!')

    if agent == 'server':
        src = server_ip
        dst = client_ip
    else:
        src = client_ip
        dst = server_ip

    print('Start simulate network')
    cnt = 0
    print_cnt = 0
    total = 0
    seed = 0
    noise_pool = generate_noise(rand_dim, (loop_size * 20) + 1000)
    times_cnt = {}
    times_total = 0
    print('Generate respond packet')
    while cnt < loop_size:   
        packet_info = ''
        cnt += 1
        return_cnt, return_times, seed = generate_times(train_2, dataSample_2, seed, noise_pool, generator_model_cnt)
        if return_cnt not in times_cnt:
            times_cnt[return_cnt] = 1
        else:
            times_cnt[return_cnt] += 1
        times_total += 1
        for i in range(return_cnt):
            x = get_data_batch(train, 64, seed=seed)
            seed += 1
            
            # read tmp.txt to get packet info
            while True:
                if not os.path.isfile('tmp.txt'):
                    break
                with open('tmp.txt', 'r+') as packet:
                    info = packet.read()
                    packet_info = info

                tmp = []
                if packet_info != '':
                    for data in packet_info.split():
                        tmp.append(data)
                if cnt == 1:
                    break
                elif len(tmp) < 3:
                    continue
                elif agent == 'server' and tmp[0] == 'client':
                    break
                elif agent == 'client' and tmp[0] == 'server':
                    break
                elif tmp[0] == 'print':
                    print(cnt)
                if print_cnt != cnt:
                    state = 'client' if agent == 'server' else 'server'
                    print('Wait for ' + state)
                    print_cnt += 1

            while True:
                # generate respond packet
                z = noise_pool[seed - 1]
                labels = x[:, -1:]
                g_z = generator_model.predict([z, labels])

                # denormailize the generator's result
                result = g_z.copy()
                result[:, 0] *= dataSample.std()[0]
                result[:, 0] += dataSample.mean()[0]

                result[:, 1] *= dataSample.std()[1]
                result[:, 1] += dataSample.mean()[1]

                if result[:, 1] > 0 and result[:, 0] > 0:
                    break
                seed += 1

            # save respond packet to tmp.txt
            size = str(result[:, 0])
            time = str(result[:, 1])
            print(size + ' ' + time)

            # record the server and client packet
            with open('flow.txt', 'a+') as result:
                result.write(src + ' ' + dst + ' ' + size[1:len(size) - 1].ljust(10) + ' ' + time[1:len(time) - 1].ljust(10) + '\n')

            with open(agent + '_time.txt', 'a+') as a:
                a.write(time[1:len(time) - 1] + '\n')

            with open(agent + '_size.txt', 'a+') as a:
                a.write(size[1:len(size) - 1] + '\n')

        info = agent + ' is finish'
        with open('tmp.txt', 'w+') as packet:
            packet.write(info)
        
        with open(agent + '_ret_cnt.txt', 'a+') as a:
            a.write(str(return_cnt) + '\n')
        with open(agent + '_ret_time.txt', 'a+') as a:
            a.write(str(return_times) + '\n')
    
    with open(agent + '_cnt.txt', 'a+') as a:
        for key in times_cnt:
            a.write(str(key) + ' ' + str(times_cnt[key]) + ' ' + str(format(times_cnt[key] / times_total, '0.5f')) + '\n')

if __name__ == '__main__':
    # get model
    if len(sys.argv) > 6:
        print('python3 run.py <agent> <csv> <size> <model>')
    train, dataSample, train_2, dataSample_2 = init_data(sys.argv[1], sys.argv[2])
    if int(sys.argv[4]) == 1:
        test_model(sys.argv[1], train_2, dataSample_2)
    else:
        server = threading.Thread(target = main, args=(sys.argv[1], train, dataSample, train_2, dataSample_2, int(sys.argv[3])))
        server.start()
        server.join()
        if os.path.isfile('tmp.txt'):
            os.remove('tmp.txt')