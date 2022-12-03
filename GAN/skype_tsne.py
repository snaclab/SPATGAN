import numpy as np
import sys
import csv, os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
cwd = os.getcwd()
#server_ip = '180.149.59.78'
#client_ip = '192.168.1.133'
client_ip = '192.168.1.133'
server_ip ='124.123.243.15'

# Get Original Data
ori_time = 0
ori_time_list = []
ori_data = []
with open(cwd+'/GAN/Skype_Data/original.txt','r') as f:
    cnt = 0
    server_cnt = 0
    client_cnt = 0
    for line in f:
        cnt += 1
        line_list = line.split()
        ori_time += float(line_list[4])
        
        #ori_data.append([float(line_list[3]), float(line_list[4])])
        #ori_data.append([ori_time, float(line_list[3]), float(line_list[4])])
        # ori_data.append([float(line_list[3]), ori_time])

        if line_list[1] == server_ip:
            ori_data.append([ori_time, float(line_list[3]), float(line_list[4])])
            # ori_data.append([float(line_list[3]), float(line_list[4])])
            # ori_data.append([float(line_list[3]), ori_time])
            server_cnt += 1
        else:
            ori_data.append([ori_time, -float(line_list[3]), float(line_list[4])])
            # ori_data.append([-float(line_list[3]), float(line_list[4])])
            # ori_data.append([-float(line_list[3]), ori_time])
            client_cnt += 1
        ori_time_list.append(ori_time)
        if cnt == 10000:
            break
    # print(cnt, server_cnt, client_cnt)

# Get Select model Data
sel_time = 0
#idx = int(sys.argv[2])
sel_time_list = []
sel_data = []
server_interval = []
client_interval = []
#with open(path + sys.argv[1] + '/' + str(idx) + '/server_ret_time.txt','r') as f:
#    for line in f:
#        server_interval.append(line.strip('\n'))
#with open(path + sys.argv[1] + '/' + str(idx) + '/client_ret_time.txt','r') as f:
#    for line in f:
#        client_interval.append(line.strip('\n'))
with open(cwd+ '/GAN/Generate/flow.txt','r') as f:
    cnt = 0
    agent = 0
    server_cnt = 0
    client_cnt = 0
    server_time_cnt = 0
    client_time_cnt = 0
    for line in f:
        cnt +=1
        line_list = line.split()
        sel_time += float(line_list[3])
        # 0 = server, 1 = client
        # normal && poisson -> client, generate && generate_TimeAvg -> client_ip
        if agent == 0 and (line_list[0] == client_ip or line_list[0] == 'client'):
            try:
                sel_time += 0#float(client_interval[client_time_cnt])
                client_time_cnt += 1
                agent = 1
            except Exception as e:
                print('client_cnt has problem: ', e)
        elif agent == 1 and (line_list[0] == server_ip or line_list[0] == 'server'):
            sel_time += 0#float(server_interval[server_time_cnt])
            server_time_cnt += 1
            agent = 0

        #sel_data.append([float(line_list[2]), float(line_list[3])])
        #sel_data.append([sel_time, float(line_list[2]), float(line_list[3])])
        # sel_data.append([float(line_list[2]), sel_time])

        if line_list[0] == server_ip or line_list[0] == 'server':
            sel_data.append([sel_time, float(line_list[2]), float(line_list[3])])
            # sel_data.append([float(line_list[2]), float(line_list[3])])
            # sel_data.append([float(line_list[2]), sel_time])
            server_cnt += 1
        else:
            sel_data.append([sel_time, -float(line_list[2]), float(line_list[3])])
            # sel_data.append([-float(line_list[2]), float(line_list[3])])
            # sel_data.append([-float(line_list[2]), sel_time])
            client_cnt += 1
        sel_time_list.append(sel_time)
        if cnt == 10000:
            break
    # print(cnt, server_cnt, client_cnt)

ori = np.asarray(ori_data)
sel = np.asarray(sel_data)

# feature normalization (feature scaling)
ori_scaler = StandardScaler()
ori = ori_scaler.fit_transform(ori)
sel_scaler = StandardScaler()
sel = sel_scaler.fit_transform(sel)

anal_sample_no = 10000
n_samples, n_features = ori.shape

'''t-SNE'''
total_data = np.concatenate((ori, sel), axis = 0)
tsne = TSNE(n_components=2, verbose=1, perplexity=30)
X_tsne = tsne.fit_transform(total_data)

# sel_tsne = TSNE(n_components=2, perplexity = 40)
# sel_X_tsne = tsne.fit_transform(sel)

# print("Org data dimension is {}.\nEmbedded data dimension is {}".format(ori.shape[-1], X_tsne.shape[-1]))
# print("Sel data dimension is {}.\nEmbedded data dimension is {}".format(sel.shape[-1], sel_X_tsne.shape[-1]))


'''嵌入空间可视化'''
# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_norm = (X_tsne - x_min) / (x_max - x_min)

# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_tsne = (X_tsne - x_min) / (x_max - x_min)

# sel_x_min, sel_x_max = sel_X_tsne.min(0), sel_X_tsne.max(0)
# sel_X_norm = (sel_X_tsne - sel_x_min) / (sel_x_max - sel_x_min)

# plt.figure(figsize=(8, 8))
select_label = 'GAN points'
# plt.scatter(X_norm[:, 0], X_norm[:, 1], s=2, color='red', label='original points', alpha=0.4)
# plt.scatter(sel_X_norm[:, 0], sel_X_norm[:, 1], s=2, color='blue', label=select_label, alpha=0.4)

plt.scatter(X_tsne[:anal_sample_no, 0], X_tsne[:anal_sample_no, 1], s=2, color='red', label='original points', alpha=0.4)
plt.scatter(X_tsne[anal_sample_no:, 0], X_tsne[anal_sample_no:, 1], s=2, color='blue', label=select_label, alpha=0.4)

plt.legend(loc=0)
plt.savefig('tsne_GAN_skype')
plt.show()
