#coding=utf-8
import os
import sys
import threading
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

### param
# Server traffic
server_size_std = 53.208221
server_times_std = 56.125846
server_size_mean = 1431.234051
server_times_mean = 115.400025

# Server cnt
server_cnt_std = 2.211023
server_interval_std = 55.979579
server_cnt_mean = 2.139698
server_interval_mean = 74.975884

# Client traffic
client_size_std = 17.931729
client_times_std = 29.627563
client_size_mean = 66.881962
client_times_mean = 70.696050

# Client cnt
client_cnt_std = 0.156459
client_interval_std = 149.206191
client_cnt_mean = 1.006877
client_interval_mean = 143.399221

server_list = [server_size_std, server_times_std, server_size_mean, server_times_mean, server_cnt_std, server_interval_std, server_cnt_mean, server_interval_mean]
client_list = [client_size_std, client_times_std, client_size_mean, client_times_mean, client_cnt_std, client_interval_std, client_cnt_mean, client_interval_mean]

# seed
seed = 0

# Generate function
def Generate_random(state, std, mean):
    # state: 1: numpy_normal, 2: numpy_poisson, 3: scipy_normal, 4: scipy_poisson
    global seed
    result = 0
    while True:
        if state == 1:
            result = np.random.normal(mean, std)
        elif state == 2:
            result = np.random.poisson(mean)
        elif state == 3:
            result = st.norm.rvs(mean, std)
        elif state == 4:
            result = st.poisson.rvs(mean)

        # check value bigger than zero
        if result > 0:
            break
    return result

def Draw_fig(agent, data, name):
    # plt.hist(data, bins=100, density=False)
    # plt.show()
    # plt.savefig(agent + '_' + name + '.png')
    # plt.clf()
    # plt.close()
    print(np.mean(data), np.std(data))


def main(agent, state, loop_size):
    param_list = server_list if agent == 'server' else client_list
    src = 'server' if agent == 'server' else 'client'
    dst = 'client' if agent == 'server' else 'server'
    traffic_size = []
    traffic_times = []
    traffic_cnt = []
    traffic_interval = []
    times_cnt = {}
    times_total = 0
    loop = 0
    print_cnt = 0
    zero_check = 0
    while loop < loop_size:

        # Generate cnt
        gen_cnt = Generate_random(state, param_list[4], param_list[6])
        gen_interval = Generate_random(state, param_list[5], param_list[7])
        if int(gen_cnt) == 0:
            zero_check += 1
            gen_cnt = 1
        if int(gen_cnt) not in times_cnt:
            times_cnt[int(gen_cnt)] = 1
        else:
            times_cnt[int(gen_cnt)] += 1
        times_total += 1

        # Generate traces
        for i in range(int(gen_cnt)):
            # read tmp.txt to get packet info
            packet_info = ''
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
                if loop == 0:
                    break
                elif len(tmp) < 3:
                    continue
                elif agent == 'server' and tmp[0] == 'client':
                    break
                elif agent == 'client' and tmp[0] == 'server':
                    break
                if print_cnt != loop:
                    waitfor = 'client' if agent == 'server' else 'server'
                    print('Wait for ' + waitfor + ' ' + str(loop) + ' ' + tmp[3])
                    print_cnt += 1

            gen_size = Generate_random(state, param_list[0], param_list[2])
            gen_times = Generate_random(state, param_list[1], param_list[3])

            # Append to list, Write in txt
            traffic_size.append(gen_size)
            traffic_times.append(gen_times)

            with open('Generate/' + agent + '_size.txt', 'a+') as a:
                a.write(str(gen_size) + '\n')

            with open('Generate/' + agent + '_time.txt', 'a+') as a:
                a.write(str(gen_times) + '\n')

            if i == 0:
                gen_times += gen_interval

            # save respond packet to tmp.txt
            size = str(gen_size)
            time = str(gen_times)
            print(size + ' ' + time)

            # record the server and client packet
            with open('Generate/' + 'flow.txt', 'a+') as result:
                result.write(src + ' ' + dst + ' ' + size.ljust(10) + ' ' + time.ljust(10) + '\n')
        
        info = agent + ' is finish ' + str(loop)
        with open('tmp.txt', 'w+') as packet:
            packet.write(info)
        with open('Generate/' + agent + '_ret_cnt.txt', 'a+') as a:
            a.write(str(gen_cnt) + '\n')
        with open('Generate/' + agent + '_ret_time.txt', 'a+') as a:
            a.write(str(gen_interval) + '\n')

        traffic_cnt.append(gen_cnt)
        traffic_interval.append(gen_interval)
        loop += 1
    
    with open('Generate/' + agent + '_cnt.txt', 'a+') as a:
        for key in times_cnt:
            a.write(str(key) + ' ' + str(times_cnt[key]) + ' ' + str(format(times_cnt[key] / times_total, '0.5f')) + '\n')
    print(zero_check)

if __name__ == '__main__':
    # get model
    if len(sys.argv) != 4:
        print('python3 simulate.py <agent> <state> <size>')
    run = threading.Thread(target=main, args=(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))
    run.start()
    run.join()
    if os.path.isfile('tmp.txt'):
        os.remove('tmp.txt')