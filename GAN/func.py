#coding=utf-8
import psutil
import socket
import struct
import statistics
import scipy
import matplotlib
import numpy as np
import pandas as pd
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
get_data_seed = 0

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

def divide_chunks(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 


'''
original function
'''

# https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_curve
def BaseMetrics(y_pred, y_true):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return TP, TN, FP, FN

def SimpleAccuracy(y_pred, y_true):
    TP, TN, FP, FN = BaseMetrics(y_pred, y_true)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    return ACC

def SimpleMetrics(y_pred, y_true):
    TP, TN, FP, FN = BaseMetrics(y_pred,y_true)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    
    # Reporting
    from IPython.display import display
    print('Confusion Matrix')
    display(pd.DataFrame([[TN, FP],[FN, TP]], columns=['Pred 0', 'Pred 1'], index=['True 0', 'True 1']))
    print('Accuracy : {}'.format(ACC))
    
def get_data_batch(train, batch_size, seed=0):
    # # random sampling - some samples will have excessively low or high sampling, but easy to implement
    # np.random.seed(seed)
    # x = train.loc[np.random.choice(train.index, batch_size)].values
    
    # iterate through shuffled indices, so every sample gets covered evenly
    global get_data_seed
    start_i = (batch_size * seed) % len(train)
    stop_i = start_i + batch_size
    shuffle_seed = (batch_size * seed) // len(train)
    np.random.seed(shuffle_seed + get_data_seed)
    get_data_seed += 1
    train_ix = np.random.choice(list(train.index), replace=False, size=len(train)) # wasteful to shuffle every time
    train_ix = list(train_ix) + list(train_ix) # duplicate to cover ranges past the end of the set
    x = train.loc[train_ix[start_i:stop_i]].values
    return np.reshape(x, (batch_size, -1))

def get_label_batch(train, batch_size, seed=0):
    # # random sampling - some samples will have excessively low or high sampling, but easy to implement
    # np.random.seed(seed)
    # x = train.loc[np.random.choice(train.index, batch_size)].values
    
    # iterate through shuffled indices, so every sample gets covered evenly
    global get_data_seed
    start_i = (batch_size * seed) % len(train)
    stop_i = start_i + batch_size
    shuffle_seed = (batch_size * seed) // len(train)
    np.random.seed(shuffle_seed + get_data_seed)
    get_data_seed += 1
    train_ix = np.random.choice(list(train.index), replace=False, size=len(train)) # wasteful to shuffle every time
    train_ix = list(train_ix) + list(train_ix) # duplicate to cover ranges past the end of the set
    x = train.loc[train_ix[start_i:stop_i]].values
    
    return np.reshape(x, (batch_size, -1))[:, -1:]
    
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
        'eval_metric': 'auc', # allows for balanced or unbalanced classes, "auc": Area under the curve for ranking evaluation.
    }
    xgb_test = xgb.train(xgb_params, dtrain, num_boost_round=10) # limit to ten rounds for faster evaluation
    y_pred = np.round(xgb_test.predict(dtest))

    # return '{:.2f}'.format(SimpleAccuracy(y_pred, y_true)) # assumes balanced real and generated datasets
    return SimpleAccuracy(y_pred, y_true)                    # assumes balanced real and generated datasets
    
def PlotData(x, g_z, data_cols, label_cols=[], seed=0, with_class=False, data_dim=2, save=False, prefix=''):
    
    real_samples = pd.DataFrame(x, columns=data_cols+label_cols)
    gen_samples = pd.DataFrame(g_z, columns=data_cols+label_cols)
    
    f, axarr = plt.subplots(1, 2, figsize=(6,2))
    axarr[0].scatter(real_samples[data_cols[0]], real_samples[data_cols[1]], c=real_samples[label_cols[0]] / 2) #, cmap='plasma')
    axarr[1].scatter(gen_samples[data_cols[0]], gen_samples[data_cols[1]], c=gen_samples[label_cols[0]] / 2) #, cmap='plasma')
    
    # For when there are multiple one-hot encoded label columns
    # for i in range(len(label_cols)):
        # temp = real_samples.loc[real_samples[label_cols[i]] == 1]
        # axarr[0].scatter(temp[data_cols[0]], temp[data_cols[1]], c='C'+str(i), label=i)
        # temp = gen_samples.loc[gen_samples[label_cols[i]] == 1]
        # axarr[1].scatter(temp[data_cols[0]], temp[data_cols[1]], c='C'+str(i), label=i)

    axarr[0].set_title('real')
    axarr[1].set_title('generated')   
    axarr[0].set_ylabel(data_cols[1])                                               # Only add y label to left plot
    for a in axarr: a.set_xlabel(data_cols[0])                                      # Add x label to both plots
    axarr[1].set_xlim(axarr[0].get_xlim()), axarr[1].set_ylim(axarr[0].get_ylim())  # Use axes ranges from real data for generated data
    
    if save:
        plt.savefig(prefix + '.xgb_check.png')
    plt.show()


#### Functions to define the layers of the networks used in the 'define_models' functions below
    
def generator_network_w_label(x, labels, data_dim, label_dim, base_n_count): 
    x = layers.concatenate([x, labels])
    x = layers.Dense(base_n_count*1, activation='relu')(x) # 1
    x = layers.Dense(base_n_count*2, activation='relu')(x) # 2
    x = layers.Dense(base_n_count*4, activation='relu')(x)
    x = layers.Dense(data_dim)(x)
    x = layers.concatenate([x, labels])
    return x
    
def critic_network(x, data_dim, base_n_count):
    x = layers.Dense(base_n_count*4, activation='relu')(x)
    x = layers.Dense(base_n_count*2, activation='relu')(x) # 2
    x = layers.Dense(base_n_count*1, activation='relu')(x) # 1
    x = layers.Dense(1)(x)
    return x    
    
#### Functions to define the keras network models    

def define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count, type=None):

    # Adversarial training of the generator network Gθ and discriminator network Dφ.
    generator_input_tensor = layers.Input(shape=(rand_dim, ))
    labels_tensor = layers.Input(shape=(label_dim,))
    generated_image_tensor = generator_network_w_label(generator_input_tensor, labels_tensor, data_dim, label_dim, base_n_count)
    generated_or_real_image_tensor = layers.Input(shape=(data_dim + label_dim,))
    
    discriminator_output = critic_network(generated_or_real_image_tensor, data_dim + label_dim, base_n_count)

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

    [cache_prefix, with_class, starting_step, train, data_cols, data_dim, 
     label_cols, label_dim, generator_model, discriminator_model, combined_model,
     rand_dim, nb_steps, batch_size, k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
     data_dir, generator_model_path, discriminator_model_path, sess, _z, _x, 
     _labels, _g_z, epsilon, x_hat, gradients, _gradient_penalty,
     _disc_loss_generated, _disc_loss_real, _disc_loss, disc_optimizer, 
     show, combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = model_components

    d_l_g, d_l_r, _ = sess.run([_disc_loss_generated, _disc_loss_real, disc_optimizer], feed_dict={
        _z: np.random.normal(size=(batch_size, rand_dim)),
        _x: get_data_batch(train, batch_size, seed=seed),
        _labels: get_label_batch(train, batch_size, seed=seed), # .reshape(-1, label_dim),
        epsilon: np.random.uniform(low=0.0, high=1.0, size=(batch_size, 1))
    })
    return d_l_g, d_l_r

def training_steps_WGAN(model_components):
    
    [cache_prefix, with_class, starting_step, train, data_cols, data_dim,
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
            labels = get_label_batch(train, batch_size, seed=i+j)
            loss = combined_model.train_on_batch([z, labels], [-np.ones(batch_size)])
        combined_loss.append(loss)

        # Determine xgb loss each step, after training generator and discriminator
        if not i % 10:                      # 2x faster than testing each step...
            K.set_learning_phase(0)         # 0 = test
            test_size = 1000                 # test using all of the actual fraud data
            z = np.random.normal(size=(test_size, rand_dim))
            x = get_data_batch(train, test_size, seed=i)
            labels = get_label_batch(train, test_size, seed=i)
            g_z = generator_model.predict([z, labels])
            xgb_loss = CheckAccuracy(x, g_z, data_cols, label_cols, seed=0, with_class=with_class, data_dim=data_dim)
            xgb_losses = np.append(xgb_losses, xgb_loss)
        
        if not i % log_interval:
            print('Step: {} of {}.'.format(i, starting_step + nb_steps))
            # K.set_learning_phase(0) # 0 = test
                        
            # loss summaries   
            print('Losses: G, D Gen, D Real, Xgb: {:.4f}, {:.4f}, {:.4f}, {:.4f}'
                .format(combined_loss[-1], disc_loss_generated[-1], disc_loss_real[-1], xgb_losses[-1]))
            print('D Real - D Gen: {:.4f}'.format(disc_loss_real[-1] - disc_loss_generated[-1]))
            
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

def adversarial_training_WGAN(arguments, train, data_cols, label_cols=[], seed=0, starting_step=0):

    [rand_dim, nb_steps, batch_size, k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
     data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show] = arguments
            
    np.random.seed(seed)                # set random seed
    data_dim = len(data_cols)
    label_dim = len(label_cols)
    with_class = True
    
    # define network models  
    K.set_learning_phase(1)             # 1 = train

    cache_prefix = 'WCGAN'
    generator_model, discriminator_model, combined_model = define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count, type='Wasserstein')

    # construct computation graph for calculating the gradient penalty (improved WGAN) and training the discriminator
    # sample a batch of noise (generator input)
    _z = tf.placeholder(tf.float32, shape=(batch_size, rand_dim))

    # sample a batch of real data
    _x = tf.placeholder(tf.float32, shape=(batch_size, data_dim + label_dim))    
    _labels = tf.placeholder(tf.float32, shape=(batch_size, label_dim))

    # generate a batch of data with the current generator
    _g_z = generator_model(inputs=[_z, _labels])
    
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
    
    model_components = [cache_prefix, with_class, starting_step, train, data_cols, data_dim,
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
        [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = pickle.load(open(loss_pickle_path, 'rb'))
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

    model_components = [cache_prefix, with_class, starting_step, train, data_cols, data_dim, 
                        label_cols, label_dim, generator_model, discriminator_model, combined_model,
                        rand_dim, nb_steps, batch_size, k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
                        data_dir, generator_model_path, discriminator_model_path, sess, _z, _x, 
                        _labels, _g_z, epsilon, x_hat, gradients, _gradient_penalty,
                        _disc_loss_generated, _disc_loss_real, _disc_loss, disc_optimizer, 
                        show, combined_loss, disc_loss_generated, disc_loss_real, xgb_losses]

    [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = training_steps_WGAN(model_components)

# End of function list