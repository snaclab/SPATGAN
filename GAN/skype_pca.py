import numpy as np
import sys
import csv, os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
cwd = os.getcwd()
#youtube
#server_ip = '180.149.59.78'
#client_ip = '192.168.1.133'
#skype
server_ip = '192.168.1.133'
client_ip ='124.123.243.15'

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
        # ori_data.append([float(line_list[3]), ori_time])
        ori_data.append([ori_time, float(line_list[3]), float(line_list[4])])
        #ori_data.append([float(line_list[3]), float(line_list[4])])
        '''
        if line_list[1] == server_ip:
            # ori_data.append([float(line_list[3]), float(line_list[4])])
            # ori_data.append([ori_time, float(line_list[3]), float(line_list[4])])
            ori_data.append([float(line_list[3]), ori_time])
            server_cnt += 1
        else:
            # ori_data.append([-float(line_list[3]), float(line_list[4])])
            # ori_data.append([ori_time, -float(line_list[3]), float(line_list[4])])
            ori_data.append([-float(line_list[3]), ori_time])
            client_cnt += 1
        '''
        ori_time_list.append(ori_time)
        if cnt == 25000:
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
with open(cwd+'/GAN/Generate/flow.txt','r') as f:
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

        # sel_data.append([float(line_list[2]), sel_time])
        sel_data.append([sel_time, float(line_list[2]), float(line_list[3])])
        #sel_data.append([float(line_list[2]), float(line_list[3])])
        '''
        if line_list[0] == server_ip or line_list[0] == 'server':
            #sel_data.append([float(line_list[2]), float(line_list[3])])
            # sel_data.append([sel_time, float(line_list[2]), float(line_list[3])])
            sel_data.append([float(line_list[2]), sel_time])
            server_cnt += 1
        else:
            #sel_data.append([-float(line_list[2]), float(line_list[3])])
            # sel_data.append([sel_time, -float(line_list[2]), float(line_list[3])])
            sel_data.append([-float(line_list[2]), sel_time])
            client_cnt += 1
        '''
        sel_time_list.append(sel_time)
        if cnt == 25000:
            break
    # print(cnt, server_cnt, client_cnt)
ori = np.asarray(ori_data)
sel = np.asarray(sel_data)

# anal_sample_no = 2000
# if len(ori) >= len(sel):
#     idx = np.random.permutation(len(sel))[:anal_sample_no]
# else:
#     idx = np.random.permutation(len(ori))[:anal_sample_no]

# ori = ori[idx]
# sel = sel[idx]
# print(ori.shape)

# feature normalization (feature scaling)
ori_scaler = StandardScaler()
ori = ori_scaler.fit_transform(ori)
sel_scaler = StandardScaler()
sel = sel_scaler.fit_transform(sel)

# PCA
# ori_pca = PCA(n_components=2)
# original_data = ori_pca.fit_transform(ori)
# print(ori)
# print(original_data)
# print(ori_pca.explained_variance_ratio_)

# sel_pca = PCA(n_components=2)
# select_data = sel_pca.fit_transform(sel)
# print(sel)
# print(select_data)
# print(sel_pca.explained_variance_ratio_)

# PCA
pca = PCA(n_components = 2)
pca.fit(ori)
pca_results = pca.transform(ori)
print(pca_results)
pca_hat_results = pca.transform(sel)
print(pca_hat_results)

# Plotting
select_label = 'GAN points'
plt.scatter(pca_results[:, 0], pca_results[:, 1], s=2, color='red', label='original points', alpha=0.2, zorder=2)
plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1], s=2, color='blue', label=select_label, alpha=0.2, zorder=1)
plt.legend(loc=0)
plt.title('PCA plot')
plt.xlabel('X-pca')
plt.ylabel('Y-pca')
plt.savefig('pca_GAN_skype')
plt.show()
