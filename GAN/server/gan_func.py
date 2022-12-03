#coding=utf-8
#version: 3.0
import psutil
import socket
import struct
import statistics
import scipy
import matplotlib
import numpy as np
import pandas as pd
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import xgboost as xgb

import pickle
import gc
import os
import sys

from keras import applications
from keras import backend as K
from keras import layers
from keras import models
from keras import optimizers
from keras import applications
from keras.layers import LeakyReLU
import tensorflow as tf

nb_cat = 10  # number of possible categories for the `categorical_latent_dim` latent variables
categorical_latent_dim = 2  # number of categorical latent vars
continuous_latent_dim = 2  # number of continuous latent vars

'''
GAN function
'''

def check_memory():
    # collect trash and check memory
    print('-------------------------')
    print('Check memory')
    print(list(psutil.virtual_memory())[0:2])
    gc.collect()
    print(list(psutil.virtual_memory())[0:2])
    print('-------------------------')

def ip2int(addr):
    return struct.unpack("!I", socket.inet_aton(addr))[0]

def int2ip(addr):
    return socket.inet_ntoa(struct.pack("!I", addr))

def check_time_format(x):
    if len(x) != 15:
        x = x + '.000000'
    return x

def show_info(x, y, base=100, N=100):
    v = []
    for i in range(5):
        count = 0
        ave = 0
        for idx in range(len(x)):
            if x[idx] > min(x) + base - 100 and x[idx] < min(x) + base:   # 5 range: min(x) + 0~100, 100~200, 200~300, 300~400, 400~500
                count += 1
                ave += y[idx]
        if count != 0:
            ave = ave / count
        v.append([count, ave])
        base += 100

    # Calculate some distance value here
    cumsum, x_moving_aves = [0], []
    for i, x_tmp in enumerate(x, 1):
        cumsum.append(cumsum[i - 1] + x_tmp)
        if i >= N:
            x_moving_ave = (cumsum[i] - cumsum[i - N]) / N
            # can do stuff with moving_ave here
            x_moving_aves.append(x_moving_ave)

    cumsum, y_moving_aves = [0], []
    for i, y_tmp in enumerate(y, 1):
        cumsum.append(cumsum[i - 1] + y_tmp)
        if i >= N:
            y_moving_ave = (cumsum[i] - cumsum[i - N]) / N
            # can do stuff with moving_ave here
            y_moving_aves.append(y_moving_ave)
    print(f'MAX Time: {max(x)}')
    print(f'MIN Time: {min(x)}')
    print(f'AVE Time: {sum(x) / len(x)}')
    print(f'STD Time: {statistics.stdev(x)}')
    print(f'SKEW Time: {scipy.stats.skew(x)}')
    # print(f'MOVING AVE Time: {x_moving_aves}')
    print(f'MAX Size: {max(y)}')
    print(f'MIN Size: {min(y)}')
    print(f'AVE Size: {sum(y) / len(y)}')
    print(f'STD Size: {statistics.stdev(y)}')
    print(f'SKEW Size: {scipy.stats.skew(y)}')
    # print(f'MOVING AVE Time: {y_moving_aves}')

def unsmoothed_losses(GAN_losses, label, linestyle, w=0):
    data_fields = ['combined_losses_', 'real_losses_', 'generated_losses_', 'xgb_losses']
    sampling_intervals = [1, 1, 1, 10]

    for data_ix in range(len(data_fields)):
        data_sets = GAN_losses[data_ix]

        plt.figure(figsize=(10, 5))
        data = data_sets
        print(w)
        if w != 0:
            plt.plot(np.array(range(0, len(data))) * sampling_intervals[data_ix], data, label=label, linestyle=linestyle)
        else:
            plt.plot(np.array(range(0, len(data))) * sampling_intervals[data_ix], pd.DataFrame(data).rolling(w).mean(), label=label, linestyle=linestyle)
        plt.ylabel(data_fields[data_ix])
        plt.xlabel('training step')
        plt.legend()
        plt.show()
#         data_dir = 'cache/'
#         if not os.path.isdir(data_dir):
#             os.mkdir(data_dir)              # check dir is exist
#         save_name = data_dir + 'w_' + str(w) + '_' + str(data_ix) + '.png'
#         plt.savefig(save_name)
#         plt.clf()
#         plt.close()

def divide_chunks(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 


'''
original function
'''

def BaseMetrics(y_pred,y_true):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return TP, TN, FP, FN

def SimpleMetrics(y_pred,y_true):
    TP, TN, FP, FN = BaseMetrics(y_pred,y_true)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    
    # Reporting
    from IPython.display import display
    print('Confusion Matrix')
    display(pd.DataFrame([[TN,FP],[FN,TP]], columns=['Pred 0','Pred 1'], index=['True 0', 'True 1']))
    print('Accuracy : {}'.format(ACC))
    
def SimpleAccuracy(y_pred,y_true):
    TP, TN, FP, FN = BaseMetrics(y_pred,y_true)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    return ACC
    
def get_data_batch(train, batch_size, seed=0):
    # # random sampling - some samples will have excessively low or high sampling, but easy to implement
    # np.random.seed(seed)
    # x = train.loc[np.random.choice(train.index, batch_size)].values
    
    # iterate through shuffled indices, so every sample gets covered evenly
    start_i = (batch_size * seed) % len(train)
    stop_i = start_i + batch_size
    #shuffle_seed = (batch_size * seed) // len(train)
    #np.random.seed(shuffle_seed)
    train_ix = list(train.index)#np.random.choice(list(train.index), replace=False, size=len(train)) # wasteful to shuffle every time
    train_ix = list(train_ix) + list(train_ix) # duplicate to cover ranges past the end of the set
    x = train.loc[train_ix[start_i:stop_i]].values
    
    return np.reshape(x, (batch_size, -1))

def get_generator_input():
    batch_size = 64
    rand_dim = 64
    categorical_latent_var = np.random.randint(0, nb_cat, size=(batch_size, categorical_latent_dim))
    continuous_latent_var = np.random.uniform(-1, 1, size=(batch_size, continuous_latent_dim))

    gen_input = np.concatenate((categorical_latent_var, continuous_latent_var, np.random.normal(
        size=(batch_size, rand_dim - categorical_latent_dim - continuous_latent_dim))), axis=1)

    assert gen_input.shape == (batch_size, rand_dim), gen_input.shape
    return gen_input, categorical_latent_var, continuous_latent_var
    
def CheckAccuracy(x, g_z, data_cols, label_cols=[], seed=0, with_class=False, data_dim=2):
    # Slightly slower code to create dataframes to feed into the xgboost dmatrix formats
    
    # real_samples = pd.DataFrame(x, columns=data_cols+label_cols)
    # gen_samples = pd.DataFrame(g_z, columns=data_cols+label_cols)
    # real_samples['syn_label'] = 0
    # gen_samples['syn_label'] = 1
    
    # training_fraction = 0.5
    # n_real, n_gen = int(len(real_samples)*training_fraction), int(len(gen_samples)*training_fraction)
    # train_df = pd.concat([real_samples[:n_real],gen_samples[:n_gen]],axis=0)
    # test_df = pd.concat([real_samples[n_real:],gen_samples[n_gen:]],axis=0)

    # X_col = test_df.columns[:-1]
    # y_col = test_df.columns[-1]
    # dtrain = xgb.DMatrix(train_df[X_col], train_df[y_col], feature_names=X_col)
    # dtest = xgb.DMatrix(test_df[X_col], feature_names=X_col)
    # y_true = test_df['syn_label']

    dtrain = np.vstack([x[:int(len(x) / 2)], g_z[:int(len(g_z) / 2)]])              # Use half of each real and generated set for training
    dlabels = np.hstack([np.zeros(int(len(x) / 2)), np.ones(int(len(g_z) / 2))])    # Synthetic labels
    dtest = np.vstack([x[int(len(x) / 2):], g_z[int(len(g_z) / 2):]])               # Use the other half of each set for testing
    y_true = dlabels # Labels for test samples will be the same as the labels for training samples, assuming even batch sizes
    
    dtrain = xgb.DMatrix(dtrain, dlabels, feature_names=data_cols+label_cols)
    dtest = xgb.DMatrix(dtest, feature_names=data_cols+label_cols)
    
    xgb_params = {
        # 'tree_method': 'hist', # for faster evaluation
        'max_depth': 4, # for faster evaluation
        'objective': 'binary:logistic',
        'random_state': 0,
        'eval_metric': 'auc', # allows for balanced or unbalanced classes 
        }
    xgb_test = xgb.train(xgb_params, dtrain, num_boost_round=10) # limit to ten rounds for faster evaluation

    y_pred = np.round(xgb_test.predict(dtest))

    # return '{:.2f}'.format(SimpleAccuracy(y_pred, y_true)) # assumes balanced real and generated datasets
    return SimpleAccuracy(y_pred, y_true) # assumes balanced real and generated datasets
    
def PlotData(x, g_z, data_cols, label_cols=[], seed=0, with_class=False, data_dim=2, save=False, prefix=''):
    
    real_samples = pd.DataFrame(x, columns=data_cols+label_cols)
    gen_samples = pd.DataFrame(g_z, columns=data_cols+label_cols)
    
    f, axarr = plt.subplots(1, 2, figsize=(6,2))
    if with_class:
        axarr[0].scatter(real_samples[data_cols[0]], real_samples[data_cols[1]], c=real_samples[label_cols[0]] / 2) #, cmap='plasma')
        axarr[1].scatter(gen_samples[data_cols[0]], gen_samples[data_cols[1]], c=gen_samples[label_cols[0]] / 2) #, cmap='plasma')
        
        # For when there are multiple one-hot encoded label columns
        # for i in range(len(label_cols)):
            # temp = real_samples.loc[real_samples[label_cols[i]] == 1]
            # axarr[0].scatter(temp[data_cols[0]], temp[data_cols[1]], c='C'+str(i), label=i)
            # temp = gen_samples.loc[gen_samples[label_cols[i]] == 1]
            # axarr[1].scatter(temp[data_cols[0]], temp[data_cols[1]], c='C'+str(i), label=i)
        
    else:
        axarr[0].scatter(real_samples[data_cols[0]], real_samples[data_cols[1]]) #, cmap='plasma')
        axarr[1].scatter(gen_samples[data_cols[0]], gen_samples[data_cols[1]]) #, cmap='plasma')
    axarr[0].set_title('real')
    axarr[1].set_title('generated')   
    axarr[0].set_ylabel(data_cols[1])                                               # Only add y label to left plot
    for a in axarr: a.set_xlabel(data_cols[0])                                      # Add x label to both plots
    axarr[1].set_xlim(axarr[0].get_xlim()), axarr[1].set_ylim(axarr[0].get_ylim())  # Use axes ranges from real data for generated data
    
    if save:
        plt.savefig(prefix + '.xgb_check.png')
#         plt.cla()
#     plt.close(f)
    plt.show()


#### Functions to define the layers of the networks used in the 'define_models' functions below

def generator_network(x, data_dim, base_n_count): 
    x = layers.Dense(base_n_count, activation='relu')(x)
    x = layers.Dense(base_n_count*2, activation='relu')(x)
    x = layers.Dense(base_n_count*4, activation='relu')(x)
    x = layers.Dense(data_dim)(x)
    return x
    
def generator_network_w_label(x, labels, data_dim, label_dim, base_n_count): 
    x = layers.concatenate([x,labels])
    x = layers.Dense(base_n_count*1, activation='relu')(x) # 1
    x = layers.Dense(base_n_count*2, activation='relu')(x) # 2
    x = layers.Dense(base_n_count*4, activation='relu')(x)
    # x = layers.Dense(base_n_count*4, activation='relu')(x) # extra
    # x = layers.Dense(base_n_count*4, activation='relu')(x) # extra
    x = layers.Dense(data_dim)(x)
    x = layers.concatenate([x, labels])
    return x
    
def discriminator_network(x, data_dim, base_n_count):
    x = layers.Dense(base_n_count*4, activation='relu')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(base_n_count*2, activation='relu')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(base_n_count, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    # x = layers.Dense(1)(x)
    return x
    
def critic_network(x, data_dim, base_n_count):
    x = layers.Dense(base_n_count*4, activation='relu')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(base_n_count*2, activation='relu')(x) # 2
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(base_n_count*1, activation='relu')(x) # 1
    # x = layers.Dense(base_n_count*4, activation='relu')(x) # extra
    # x = layers.Dense(base_n_count*4, activation='relu')(x) # extra
    # x = layers.Dense(1, activation='sigmoid')(x)
    x = layers.Dense(1)(x)
    return x

def generator_Info_network(x, data_dim, base_n_count): 
    x = layers.Dense(base_n_count, activation='relu')(x)
    x = layers.Dense(base_n_count*2, activation='relu')(x)
    x = layers.Dense(base_n_count*4, activation='relu')(x)
    x = layers.Dense(data_dim)(x)
    return x

def discriminator_Info_network(x, data_dim, base_n_count):
    x = layers.Dense(base_n_count*4, activation='relu')(x)
    x = layers.Dense(base_n_count*2, activation='relu')(x)
    shared = layers.Dense(base_n_count, activation='relu')(x)
    disc = layers.Dense(1, activation='sigmoid')(shared)
    x = layers.Dense(base_n_count*2, activation='relu')(shared)

    # the `softmax` activation will be applied in the custom loss
    cat = layers.Dense(categorical_latent_dim * nb_cat)(x)

    # use `tanh` non-linearity since cont latent variables are in range [-1, 1]
    cont = layers.Dense(continuous_latent_dim, activation='tanh')(x)
    return disc, cat, cont
    
    
#### Functions to define the keras network models    
    
def define_models_GAN(rand_dim, data_dim, base_n_count, type=None):
    
    # Adversarial training of the generator network Gθ and discriminator network Dφ.
    generator_input_tensor = layers.Input(shape=(rand_dim, ))
    generated_image_tensor = generator_network(generator_input_tensor, data_dim, base_n_count)
    generated_or_real_image_tensor = layers.Input(shape=(data_dim,))
    
    if type == 'Wasserstein':
        discriminator_output = critic_network(generated_or_real_image_tensor, data_dim, base_n_count)
    else:
        discriminator_output = discriminator_network(generated_or_real_image_tensor, data_dim, base_n_count)

    generator_model = models.Model(inputs=[generator_input_tensor], outputs=[generated_image_tensor], name='generator')
    discriminator_model = models.Model(inputs=[generated_or_real_image_tensor], outputs=[discriminator_output], name='discriminator')

    combined_output = discriminator_model(generator_model(generator_input_tensor))
    combined_model = models.Model(inputs=[generator_input_tensor], outputs=[combined_output], name='combined')
    
    return generator_model, discriminator_model, combined_model

# With label, updated for class
def define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count, type=None):

    # Adversarial training of the generator network Gθ and discriminator network Dφ.
    generator_input_tensor = layers.Input(shape=(rand_dim, ))
    labels_tensor = layers.Input(shape=(label_dim,))
    generated_image_tensor = generator_network_w_label(generator_input_tensor, labels_tensor, data_dim, label_dim, base_n_count)
    generated_or_real_image_tensor = layers.Input(shape=(data_dim + label_dim,))
    
    if type == 'Wasserstein':
        discriminator_output = critic_network(generated_or_real_image_tensor, data_dim + label_dim, base_n_count)
    else:
        discriminator_output = discriminator_network(generated_or_real_image_tensor, data_dim + label_dim, base_n_count)

    generator_model = models.Model(inputs=[generator_input_tensor, labels_tensor], outputs=[generated_image_tensor], name='generator')
    discriminator_model = models.Model(inputs=[generated_or_real_image_tensor], outputs=[discriminator_output], name='discriminator')

    combined_output = discriminator_model(generator_model([generator_input_tensor, labels_tensor]))
    combined_model = models.Model(inputs=[generator_input_tensor, labels_tensor], outputs=[combined_output], name='combined')
    
    return generator_model, discriminator_model, combined_model


#### Functions specific to the WGAN architecture
#### The train discrimnator step is separated out to facilitate pre-training of the discriminator by itself
#### https://github.com/mjdietzx/GAN-Sandbox/blob/wGAN/gan.py

def em_loss(y_coefficients, y_pred):
    # define earth mover distance (wasserstein loss)
    # literally the weighted average of the critic network output
    # this is defined separately so it can be fed as a loss function to the optimizer in the WGANs
    return tf.reduce_mean(tf.multiply(y_coefficients, y_pred))


def train_discriminator_step(model_components, seed=0):

    [cache_prefix, with_class, starting_step, train, train_o, data_cols, data_dim, label_cols, label_dim,
     generator_model, discriminator_model, combined_model, rand_dim, nb_steps, batch_size, 
     k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
     data_dir, generator_model_path, discriminator_model_path, sess, _z, _x, 
     _labels, _g_z, epsilon, x_hat, gradients, _gradient_penalty, _disc_loss_generated, _disc_loss_real,
     _disc_loss, disc_optimizer, show, combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = model_components
    
    rd_seed = int(np.random.rand(1)*10000)
    if with_class:
        d_l_g, d_l_r, _ = sess.run([_disc_loss_generated, _disc_loss_real, disc_optimizer], feed_dict={
            _z: np.random.normal(size=(batch_size, rand_dim)),
            # _x: get_data_batch(train, batch_size, seed=seed),  # size have problem, so can't use class
            _x: np.hstack((get_data_batch(train, batch_size, seed=rd_seed), get_data_batch(train_o, batch_size, seed=rd_seed-1)[:, -label_dim:])),
            _labels: get_data_batch(train_o, batch_size, seed=rd_seed-1)[:,-label_dim:], # .reshape(-1, label_dim),
            epsilon: np.random.uniform(low=0.0, high=1.0, size=(batch_size, 1))
        })
    else:
        d_l_g, d_l_r, _ = sess.run([_disc_loss_generated, _disc_loss_real, disc_optimizer], feed_dict={
            _z: np.random.normal(size=(batch_size, rand_dim)),
            _x: get_data_batch(train, batch_size, seed=seed),
            epsilon: np.random.uniform(low=0.0, high=1.0, size=(batch_size, 1))
        })
    return d_l_g, d_l_r

def training_steps_WGAN(model_components):
    
    [cache_prefix, with_class, starting_step, train, train_o, data_cols, data_dim,
     label_cols, label_dim, generator_model, discriminator_model, combined_model,
     rand_dim, nb_steps, batch_size, k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
     data_dir, generator_model_path, discriminator_model_path, sess, _z, _x, 
     _labels, _g_z, epsilon, x_hat, gradients, _gradient_penalty,
     _disc_loss_generated, _disc_loss_real, _disc_loss, disc_optimizer,
     show, combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = model_components
    
    for i in range(starting_step, starting_step + nb_steps):
        K.set_learning_phase(1) # 1 = train

        # train the discriminator
        for j in range(k_d):
            d_l_g, d_l_r = train_discriminator_step(model_components, seed=i+j)
        disc_loss_generated.append(d_l_g)
        disc_loss_real.append(d_l_r)

        # train the generator
        for j in range(k_g):
            np.random.seed(i+j)
            z = np.random.normal(size=(batch_size, rand_dim))
            if with_class:
                labels = get_data_batch(train_o, batch_size, seed=i+j-1)[:,-label_dim:]
                loss = combined_model.train_on_batch([z, labels], [-np.ones(batch_size)])
            else:
                loss = combined_model.train_on_batch(z, [-np.ones(batch_size)])
        combined_loss.append(loss)

        # Determine xgb loss each step, after training generator and discriminator
        if not i % 10:                      # 2x faster than testing each step...
            K.set_learning_phase(0)         # 0 = test
            test_size = 518                 # test using all of the actual fraud data
            x = get_data_batch(train, test_size, seed=i)
            z = np.random.normal(size=(test_size, rand_dim))
            if with_class:
                x = np.hstack((get_data_batch(train, test_size, seed=i), get_data_batch(train_o, test_size, seed=i-1)[:, -label_dim:]))
                labels = x[:, -label_dim:]
                g_z = generator_model.predict([z, labels])
            else:
                g_z = generator_model.predict(z)
            xgb_loss = CheckAccuracy(x, g_z, data_cols, label_cols, seed=0, with_class=with_class, data_dim=data_dim)
            xgb_losses = np.append(xgb_losses, xgb_loss)
        
        if not i % log_interval:
            print('Step: {} of {}.'.format(i, starting_step + nb_steps))
            # K.set_learning_phase(0) # 0 = test
                        
            # loss summaries   
            print('Losses: G, D Gen, D Real, Xgb: {:.4f}, {:.4f}, {:.4f}, {:.4f}'
                .format(combined_loss[-1], disc_loss_generated[-1], disc_loss_real[-1], xgb_losses[-1]))
            print('D Real - D Gen: {:.4f}'.format(disc_loss_real[-1] - disc_loss_generated[-1]))
            # print('Generator model loss: {}.'.format(combined_loss[-1]))
            # print('Discriminator model loss gen: {}.'.format(disc_loss_generated[-1]))
            # print('Discriminator model loss real: {}.'.format(disc_loss_real[-1]))
            # print('xgboost accuracy: {}'.format(xgb_losses[-1]) )
            
            if show:
                PlotData(x, g_z, data_cols, label_cols, seed=0, with_class=with_class, 
                    data_dim=data_dim, save=True, prefix= data_dir + cache_prefix + '_' + str(i))

            # save model checkpoints
            model_checkpoint_base_name = data_dir + cache_prefix + '_{}_model_weights_step_{}.h5'
            generator_model.save_weights(model_checkpoint_base_name.format('generator', i))
            discriminator_model.save_weights(model_checkpoint_base_name.format('discriminator', i))
            pickle.dump([combined_loss, disc_loss_generated, disc_loss_real, xgb_losses], 
                open(data_dir + cache_prefix + '_losses_step_{}.pkl'.format(i) ,'wb'))
    
    return [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses]

def adversarial_training_WGAN(arguments, train, train_o, data_cols, label_cols=[], seed=0, starting_step=0):

    [rand_dim, nb_steps, batch_size, k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
     data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show ] = arguments
            
    np.random.seed(seed)                # set random seed
    data_dim = len(data_cols)
    label_dim = 0
    with_class = False
    if len(label_cols) > 0: 
        with_class = True
        label_dim = len(label_cols)
    
    # define network models  
    K.set_learning_phase(1)             # 1 = train

    if with_class:
        cache_prefix = 'WCGAN'
        generator_model, discriminator_model, combined_model = define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count, type='Wasserstein')
    else:
        cache_prefix = 'WGAN'
        generator_model, discriminator_model, combined_model = define_models_GAN(rand_dim, data_dim, base_n_count, type='Wasserstein')

    # construct computation graph for calculating the gradient penalty (improved WGAN) and training the discriminator
    # sample a batch of noise (generator input)
    _z = tf.placeholder(tf.float32, shape=(batch_size, rand_dim))
    
    _labels = None

    if with_class:
        _x = tf.placeholder(tf.float32, shape=(batch_size, data_dim + label_dim))    
        _labels = tf.placeholder(tf.float32, shape=(batch_size, label_dim))
        _g_z = generator_model(inputs=[_z, _labels])
    else:
        # sample a batch of real data
        _x = tf.placeholder(tf.float32, shape=(batch_size, data_dim))

        # generate a batch of data with the current generator
        _g_z = generator_model(_z)
    
    # calculate `x_hat`
    epsilon = tf.placeholder(tf.float32, shape=(batch_size, 1))
    x_hat = epsilon * _x + (1.0 - epsilon) * _g_z

    # gradient penalty
    gradients = tf.gradients(discriminator_model(x_hat), [x_hat])
    _gradient_penalty = 10.0 * tf.square(tf.norm(gradients[0], ord=2) - 1.0)

    # calculate discriminator's loss
    _disc_loss_generated = em_loss(tf.ones(batch_size), discriminator_model(_g_z))
    _disc_loss_real = em_loss(tf.ones(batch_size), discriminator_model(_x))
    _disc_loss = _disc_loss_generated - _disc_loss_real + _gradient_penalty

    # update φ by taking an SGD step on mini-batch loss LD(φ)
    disc_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9).minimize(
        _disc_loss, var_list=discriminator_model.trainable_weights)

    sess = K.get_session()

    # compile models

    adam = optimizers.Adam(lr=learning_rate, beta_1=0.5, beta_2=0.9)

    discriminator_model.trainable = False
    combined_model.compile(optimizer=adam, loss=[em_loss])

    combined_loss, disc_loss_generated, disc_loss_real, xgb_losses = [], [], [], []
    
    model_components = [cache_prefix, with_class, starting_step, train, train_o, data_cols, data_dim,
                        label_cols, label_dim, generator_model, discriminator_model, combined_model,
                        rand_dim, nb_steps, batch_size, k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
                        data_dir, generator_model_path, discriminator_model_path, sess, _z, _x, 
                        _labels, _g_z, epsilon, x_hat, gradients, _gradient_penalty,
                        _disc_loss_generated, _disc_loss_real, _disc_loss, disc_optimizer,
                        show, combined_loss, disc_loss_generated, disc_loss_real, xgb_losses]

    if show:
        print(generator_model.summary())
        print(discriminator_model.summary())
        print(combined_model.summary())

    if loss_pickle_path:
        print('Loading loss pickles')
        [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = pickle.load(open(loss_pickle_path,'rb'))
    if generator_model_path:
        print('Loading generator model')
        generator_model.load_weights(generator_model_path) #, by_name=True)
    if discriminator_model_path:
        print('Loading discriminator model')
        discriminator_model.load_weights(discriminator_model_path) #, by_name=True)
    else:
        print('pre-training the critic...')
        K.set_learning_phase(1) # 1 = train
        
        for i in range(critic_pre_train_steps):
            print('Step: {} of {} critic pre-training.'.format(i, critic_pre_train_steps))
            loss = train_discriminator_step(model_components, seed=i)
        print('Last batch of critic pre-training disc_loss: {}.'.format(loss))
        discriminator_model.save(os.path.join(data_dir, 'discriminator_model_pre_trained.h5'))

    model_components = [cache_prefix, with_class, starting_step,
                        train, train_o, data_cols, data_dim, label_cols, label_dim,
                        generator_model, discriminator_model, combined_model,
                        rand_dim, nb_steps, batch_size, k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
                        data_dir, generator_model_path, discriminator_model_path, sess, _z, _x, 
                        _labels, _g_z, epsilon, x_hat, gradients, _gradient_penalty,
                        _disc_loss_generated, _disc_loss_real, _disc_loss, disc_optimizer, show,
                        combined_loss, disc_loss_generated, disc_loss_real, xgb_losses]

    [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = training_steps_WGAN(model_components)


#### Functions specific to the vanilla GAN architecture   
        
def training_steps_GAN(model_components):
    
    [cache_prefix, with_class, starting_step,
     train, data_cols, data_dim,
     label_cols, label_dim,
     generator_model, discriminator_model, combined_model,
     rand_dim, nb_steps, batch_size, 
     k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
     data_dir, generator_model_path, discriminator_model_path, show,
     combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = model_components  
    
    for i in range(starting_step, starting_step + nb_steps):
        K.set_learning_phase(1) # 1 = train

        # train the discriminator
        for j in range(k_d):
            np.random.seed(i + j)
            z = np.random.normal(size=(batch_size, rand_dim))
            x = get_data_batch(train, batch_size, seed=i+j)
            if with_class:
                x = np.hstack((get_data_batch(train, batch_size, seed=i+j), get_data_batch(train, batch_size, seed=i+j)[:, -label_dim:]))
                labels = x[:, -label_dim:]
                g_z = generator_model.predict([z, labels])
            else:
                g_z = generator_model.predict(z)
                # code to train the discriminator on real and generated data at the same time, but you have to run network again for separate losses
            # x = np.vstack([x,g_z])
            # classes = np.hstack([np.zeros(batch_size),np.ones(batch_size)])
            # d_l_r = discriminator_model.train_on_batch(x, classes)
            
            d_l_r = discriminator_model.train_on_batch(x, 
                np.random.uniform(low=0.999, high=1.0, size=batch_size)) # 0.7, 1.2 # GANs need noise to prevent loss going to zero
            d_l_g = discriminator_model.train_on_batch(g_z, 
                np.random.uniform(low=0.0, high=0.001, size=batch_size)) # 0.0, 0.3 # GANs need noise to prevent loss going to zero
            # d_l_r = discriminator_model.train_on_batch(x, np.ones(batch_size)) # without noise
            # d_l_g = discriminator_model.train_on_batch(g_z, np.zeros(batch_size)) # without noise
        disc_loss_real.append(d_l_r)
        disc_loss_generated.append(d_l_g)
        
        # train the generator
        for j in range(k_g):
            np.random.seed(i+j)
            z = np.random.normal(size=(batch_size, rand_dim))
            if with_class:
                # loss = combined_model.train_on_batch([z, labels], np.ones(batch_size)) # without noise
                loss = combined_model.train_on_batch([z, labels], 
                    np.random.uniform(low=0.999, high=1.0, size=batch_size)) # 0.7, 1.2 # GANs need noise to prevent loss going to zero
            else:
                # loss = combined_model.train_on_batch(z, np.ones(batch_size)) # without noise
                loss = combined_model.train_on_batch(z, 
                    np.random.uniform(low=0.999, high=1.0, size=batch_size)) # 0.7, 1.2 # GANs need noise to prevent loss going to zero
        combined_loss.append(loss)
        
        # Determine xgb loss each step, after training generator and discriminator
        if not i % 10:                      # 2x faster than testing each step...
            K.set_learning_phase(0)         # 0 = test
            test_size = 518                 # test using all of the actual fraud data
            x = get_data_batch(train, test_size, seed=i)
            z = np.random.normal(size=(test_size, rand_dim))
            if with_class:
                x = np.hstack((get_data_batch(train, test_size, seed=i), get_data_batch(train, test_size, seed=i)[:, -label_dim:]))
                labels = x[:,-label_dim:]
                g_z = generator_model.predict([z, labels])
            else:
                g_z = generator_model.predict(z)
            xgb_loss = CheckAccuracy( x, g_z, data_cols, label_cols, seed=0, with_class=with_class, data_dim=data_dim )
            xgb_losses = np.append(xgb_losses, xgb_loss)

        # Saving weights and plotting images
        if not i % log_interval:
            print('Step: {} of {}.'.format(i, starting_step + nb_steps))
            K.set_learning_phase(0) # 0 = test
                        
            # loss summaries      
            print( 'Losses: G, D Gen, D Real, Xgb: {:.4f}, {:.4f}, {:.4f}, {:.4f}'
                .format(combined_loss[-1], disc_loss_generated[-1], disc_loss_real[-1], xgb_losses[-1]) )
            print( 'D Real - D Gen: {:.4f}'.format(disc_loss_real[-1]-disc_loss_generated[-1]) )            
            # print('Generator model loss: {}.'.format(combined_loss[-1]))
            # print('Discriminator model loss gen: {}.'.format(disc_loss_generated[-1]))
            # print('Discriminator model loss real: {}.'.format(disc_loss_real[-1]))
            # print('xgboost accuracy: {}'.format(xgb_losses[-1]) )
            
            if show:
                PlotData(x, g_z, data_cols, label_cols, seed=0, with_class=with_class, data_dim=data_dim, 
                            save=True, prefix= data_dir + cache_prefix + '_' + str(i))
            
            # save model checkpoints
            model_checkpoint_base_name = data_dir + cache_prefix + '_{}_model_weights_step_{}.h5'
            generator_model.save_weights(model_checkpoint_base_name.format('generator', i))
            discriminator_model.save_weights(model_checkpoint_base_name.format('discriminator', i))
            pickle.dump([combined_loss, disc_loss_generated, disc_loss_real, xgb_losses], 
                open(data_dir + cache_prefix + '_losses_step_{}.pkl'.format(i) ,'wb'))
    
    return [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses]
    
def adversarial_training_GAN(arguments, train, data_cols, label_cols=[], seed=0, starting_step=0):

    [rand_dim, nb_steps, batch_size, k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
     data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show] = arguments
    
    np.random.seed(seed)     # set random seed
    data_dim = len(data_cols)
    label_dim = 0
    with_class = False
    if len(label_cols) > 0: 
        with_class = True
        label_dim = len(label_cols)
    
    # define network models
    K.set_learning_phase(1) # 1 = train
    if with_class:
        cache_prefix = 'CGAN'
        generator_model, discriminator_model, combined_model = define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count)
    else:
        cache_prefix = 'GAN'
        generator_model, discriminator_model, combined_model = define_models_GAN(rand_dim, data_dim, base_n_count)
    
    # compile models
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.5, beta_2=0.999)
    generator_model.compile(optimizer=adam, loss='binary_crossentropy')
    discriminator_model.compile(optimizer=adam, loss='binary_crossentropy')
    discriminator_model.trainable = False
    combined_model.compile(optimizer=adam, loss='binary_crossentropy')
    
    if show:
        print(generator_model.summary())
        print(discriminator_model.summary())
        print(combined_model.summary())

    combined_loss, disc_loss_generated, disc_loss_real, xgb_losses = [], [], [], []
    
    if loss_pickle_path:
        print('Loading loss pickles')
        [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = pickle.load(open(loss_pickle_path,'rb'))
    if generator_model_path:
        print('Loading generator model')
        generator_model.load_weights(generator_model_path, by_name=True)
    if discriminator_model_path:
        print('Loading discriminator model')
        discriminator_model.load_weights(discriminator_model_path, by_name=True)

    model_components = [cache_prefix, with_class, starting_step,
                        train, data_cols, data_dim,
                        label_cols, label_dim,
                        generator_model, discriminator_model, combined_model,
                        rand_dim, nb_steps, batch_size, 
                        k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
                        data_dir, generator_model_path, discriminator_model_path, show,
                        combined_loss, disc_loss_generated, disc_loss_real, xgb_losses]
        
    [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = training_steps_GAN(model_components)


#### Functions specific to the Info GAN architecture   

def define_models_InfoGAN(rand_dim, data_dim, base_n_count, type=None):
    
    ## Adversarial training of the generator network Gθ and discriminator network Dφ.
    # define model input and output tensors
    generator_input_tensor = layers.Input(shape=(rand_dim, ))
    generated_image_tensor = generator_Info_network(generator_input_tensor, data_dim, base_n_count)
    
    generated_or_real_image_tensor = layers.Input(shape=(data_dim, ))
    discriminator_output = discriminator_Info_network(generated_or_real_image_tensor, data_dim, base_n_count)

    combined_output = discriminator_Info_network(generator_Info_network(generator_input_tensor, data_dim, base_n_count), data_dim, base_n_count)

    # define models
    generator_model = models.Model(inputs=generator_input_tensor, outputs=generated_image_tensor, name='generator')
    discriminator_model = models.Model(inputs=generated_or_real_image_tensor, outputs=discriminator_output, name='discriminator')

    # real images don't have categorical or continuous latent variables so this loss should be ignored
    discriminator_model_real = models.Model(inputs=generated_or_real_image_tensor, outputs=discriminator_output[0], name='discriminator_real')
    combined_model = models.Model(inputs=generator_input_tensor, outputs=combined_output, name='combined')
    
    return generator_model, discriminator_model, combined_model

def training_steps_InfoGAN(model_components):
    
    [cache_prefix, with_class, starting_step,
     train, data_cols, data_dim,
     label_cols, label_dim,
     generator_model, discriminator_model, combined_model, discriminator_model_real,
     rand_dim, nb_steps, batch_size, 
     k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
     data_dir, generator_model_path, discriminator_model_path, show,
     combined_loss, disc_loss_generated, disc_loss_real, xgb_losses ] = model_components

    # the target labels for the binary cross-entropy loss layer are 0 for every yj (real) and 1 for every xi (generated)
    y_real = np.array([0] * batch_size)
    y_generated = np.array([1] * batch_size)

    loss = np.zeros(shape=len(combined_model.metrics_names))
    d_l_r = np.zeros(shape=len(discriminator_model_real.metrics_names))
    d_l_g = np.zeros(shape=len(discriminator_model.metrics_names))
    
    for i in range(starting_step, starting_step + nb_steps):
        K.set_learning_phase(1) # 1 = train
        # train the discriminator
        for j in range(k_d):
            generator_input, cat, cont = get_generator_input()
            # sample a mini-batch of real images
            real_image_batch = get_data_batch(train, batch_size, seed=i+j)

            # generate a batch of images with the current generator
            generated_image_batch = generator_model.predict(generator_input)
            
            d_l_r = np.add(discriminator_model_real.train_on_batch(real_image_batch, y_real), d_l_r)
            d_l_g = np.add(discriminator_model.train_on_batch(generated_image_batch, [y_generated, cat, cont]), d_l_g)
        
        # train the generator
        for j in range(k_g * 2):
            generator_input, cat, cont = get_generator_input()

            # update θ by taking an SGD step on mini-batch loss LR(θ)
            loss = np.add(combined_model.train_on_batch(generator_input, [y_real, cat, cont]), loss)

        test_size = 518 # test using all of the actual fraud data
        x = get_data_batch(train, test_size, seed=i)
        z = np.random.normal(size=(test_size, rand_dim))
        g_z = generator_model.predict(z)


        # Saving weights and plotting images
        if not i % log_interval:
            K.set_learning_phase(0) # 0 = test
            print('Step: {} of {}.'.format(i, starting_step + nb_steps))
                        
            # loss summaries      
            print('Generator model loss: {}.'.format(loss / (log_interval * k_g * 2)))
            print('Discriminator model loss real: {}.'.format(d_l_r / (log_interval * k_d * 2)))
            print('Discriminator model loss generated: {}.'.format(d_l_g / (log_interval * k_d * 2)))
            
            if show:
                PlotData(x, g_z, data_cols, label_cols, seed=0, with_class=with_class, data_dim=data_dim, 
                            save=True, prefix= data_dir + cache_prefix + '_' + str(i))
            
            loss = np.zeros(shape=len(combined_model.metrics_names))
            d_l_r = np.zeros(shape=len(discriminator_model_real.metrics_names))
            d_l_g = np.zeros(shape=len(discriminator_model.metrics_names))

            # save model checkpoints
            model_checkpoint_base_name = data_dir + cache_prefix + '_{}_model_weights_step_{}.h5'
            generator_model.save_weights(model_checkpoint_base_name.format('generator', i))
            discriminator_model.save_weights(model_checkpoint_base_name.format('discriminator', i))
    
    return [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses]
    
def adversarial_training_InfoGAN(arguments, train, data_cols, label_cols=[], seed=0, starting_step=0):

    [rand_dim, nb_steps, batch_size, 
     k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
     data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show ] = arguments
    
    np.random.seed(seed)     # set random seed
    
    data_dim = len(data_cols)
    
    label_dim = 0
    
    # define network models
    K.set_learning_phase(1) # 1 = train
    cache_prefix = 'InfoGAN'

    ## Adversarial training of the generator network Gθ and discriminator network Dφ.
    # define model input and output tensors
    generator_input_tensor = layers.Input(shape=(rand_dim, ))
    generated_image_tensor = generator_Info_network(generator_input_tensor, data_dim, base_n_count)
    
    generated_or_real_image_tensor = layers.Input(shape=(data_dim, ))
    discriminator_output = discriminator_Info_network(generated_or_real_image_tensor, data_dim, base_n_count)

    combined_output = discriminator_Info_network(generator_Info_network(generator_input_tensor, data_dim, base_n_count), data_dim, base_n_count)

    # define models
    generator_model = models.Model(inputs=generator_input_tensor, outputs=generated_image_tensor, name='generator')
    discriminator_model = models.Model(inputs=generated_or_real_image_tensor, outputs=discriminator_output, name='discriminator')

    # real images don't have categorical or continuous latent variables so this loss should be ignored
    discriminator_model_real = models.Model(inputs=generated_or_real_image_tensor, outputs=discriminator_output[0], name='discriminator_real')
    combined_model = models.Model(inputs=generator_input_tensor, outputs=combined_output, name='combined')
    
    
    # custom loss functions
    def categorical_latent_loss(y_true, y_pred):
        delta = 1.0  # tune `delta` so `categorical_latent_loss` is on the same scale as other GAN objectives

        # y_true.shape == (batch_size, categorical_latent_dim) =>
        y_true = tf.reshape(y_true, (-1, ))
        y_true = tf.cast(y_true, tf.int32)

        # y_pred.shape == (batch_size, categorical_latent_dim * nb_cat) =>
        y_pred = tf.reshape(y_pred, (-1, nb_cat))
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        return tf.multiply(delta, tf.reduce_mean(loss))

    # TODO: not sure if this is correct
    def continuous_latent_loss(y_true, y_pred):
        delta = 1.0

        y_true = tf.reshape(y_true, (-1, ))
        y_pred = tf.reshape(y_pred, (-1, ))
        loss = tf.square(y_pred - y_true)

        return tf.multiply(delta, tf.reduce_mean(loss))
    
    # compile models
    adam = optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)

    generator_model.compile(optimizer=adam, loss='binary_crossentropy')
    # discriminator_model.compile(optimizer=adam, loss='binary_crossentropy')
    # discriminator_model.trainable = False
    # combined_model.compile(optimizer=adam, loss='binary_crossentropy')

    discriminator_model_real.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    discriminator_model.compile(optimizer=adam,
                                loss=['binary_crossentropy', categorical_latent_loss, continuous_latent_loss],
                                metrics=['accuracy'])

    discriminator_model.trainable = False
    combined_model.compile(optimizer=adam, loss=['binary_crossentropy', categorical_latent_loss, continuous_latent_loss], metrics=['accuracy'])
    
    if show:
        print(generator_model.summary())
        print(discriminator_model.summary())
        print(combined_model.summary())

    combined_loss, disc_loss_generated, disc_loss_real, xgb_losses = [], [], [], []
    
    if loss_pickle_path:
        print('Loading loss pickles')
        [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = pickle.load(open(loss_pickle_path,'rb'))
    if generator_model_path:
        print('Loading generator model')
        generator_model.load_weights(generator_model_path, by_name=True)
    if discriminator_model_path:
        print('Loading discriminator model')
        discriminator_model.load_weights(discriminator_model_path, by_name=True)

    with_class = False
    model_components = [ cache_prefix, with_class, starting_step,
                        train, data_cols, data_dim,
                        label_cols, label_dim,
                        generator_model, discriminator_model, combined_model, discriminator_model_real,
                        rand_dim, nb_steps, batch_size, 
                        k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
                        data_dir, generator_model_path, discriminator_model_path, show,
                        combined_loss, disc_loss_generated, disc_loss_real, xgb_losses]
        
    [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = training_steps_InfoGAN(model_components)

######################### No Used #########################
#### Functions specific to the DRAGAN architecture
#### Note the DRAGAN is implemented in tensorflow without Keras libraries 
#### https://github.com/kodalinaveen3/DRAGAN
        
def sample_z(m, n): # updated to normal distribution
#     return np.random.uniform(-1., 1., size=[m, n])
    return np.random.normal(size=[m, n])

def xavier_init(size): # updated to uniform distribution using standard xavier formulation
#     in_dim = size[0]
#     xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
#     return tf.random_normal(shape=size, stddev=xavier_stddev, seed=global_seed)
    xavier_range = tf.sqrt( 6 / ( size[0] + size[1] ) )
    return tf.random_uniform(shape=size, minval=-xavier_range, maxval=xavier_range)

def get_perturbed_batch(minibatch):
    return minibatch + 0.5 * minibatch.std() * np.random.random(minibatch.shape)

     
def G(z, G_W, G_b): # The Generator Network
    # for i in range(len(G_layer_dims)-2):
    for i in range(len(G_W)-1):
        z = tf.nn.relu(tf.matmul(z, G_W[i]) + G_b[i])
#         print(i,G_W[i],z)
    return tf.matmul(z, G_W[-1]) + G_b[-1]     
    
def D(x, D_W, D_b): # The Discriminator Network
    # for i in range(len(D_layer_dims)-2):
    for i in range(len(D_W)-1):
        x = tf.nn.relu(tf.matmul(x, D_W[i]) + D_b[i])
    return tf.nn.sigmoid(tf.matmul(x, D_W[-1]) + D_b[-1])
     
     
def define_DRAGAN_network( X_dim=2, h_dim=128, z_dim=2, lambda0=10, learning_rate=1e-4, mb_size=128, seed=0 ):
    
    X = tf.placeholder(tf.float32, shape=[None, X_dim], name='X' )
    X_p = tf.placeholder(tf.float32, shape=[None, X_dim], name='X_p' )
    z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z' )

    D_layer_dims = [X_dim, h_dim*4, h_dim*2, h_dim, 1 ]
    D_W, D_b = [], []   
    for i in range(len(D_layer_dims)-1):
        D_W.append( tf.Variable( xavier_init([D_layer_dims[i], D_layer_dims[i+1]] ), name='D_W'+str(i) ) )
    #     D_W.append( tf.Variable(  initializer=tf.contrib.layers.xavier_initializer(seed=global_seed) ) # working towards using tf's own xavier initializer
        D_b.append( tf.Variable( tf.zeros(shape=[D_layer_dims[i+1]]), name='D_b'+str(i) ) )
    theta_D = D_W + D_b

    G_layer_dims = [z_dim, h_dim, h_dim*2, h_dim*4, X_dim ]
    G_W, G_b = [], []
    for i in range(len(G_layer_dims)-1):
        G_W.append( tf.Variable( xavier_init([G_layer_dims[i], G_layer_dims[i+1]] ), name='G_W'+str(i) ) )
        G_b.append( tf.Variable( tf.zeros(shape=[G_layer_dims[i+1]]), name='g_b'+str(i) ) )
    theta_G = G_W + G_b
    # print( theta_D + theta_G )
        
    G_sample = G(z, G_W, G_b)
    D_real = D(X, D_W, D_b)
    D_fake = D(G_sample, D_W, D_b)
    D_real_perturbed = D(X_p, D_W, D_b)

    # D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
    # D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
    # disc_cost = D_loss_real + D_loss_fake 
    # gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
    D_loss_real = tf.reduce_mean(tf.log( D_real ))
    D_loss_fake = tf.reduce_mean(tf.log( 1 - D_fake ))
    disc_cost = - D_loss_real - D_loss_fake
    gen_cost = D_loss_fake

    #Gradient penalty
    alpha = tf.random_uniform(
        shape=[mb_size,1], 
        minval=0.,
        maxval=1.) # do not set seed
        
    differences = X_p - X
    interpolates = X + (alpha*differences)
    gradients = tf.gradients(D(interpolates, D_W, D_b), [interpolates])[0]
    # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    # gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    gradient_penalty = tf.square(tf.norm(gradients, ord=2) - 1.0 )  # corrected?

    disc_cost += lambda0 * gradient_penalty / mb_size # corrected?

    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=theta_G)
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=theta_D)
    
    return [ D_solver, disc_cost, D_loss_real, D_loss_fake,
                X, X_p, z,
                G_solver, gen_cost, G_sample ]

# End of function list
