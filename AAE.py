# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:50:32 2017
ref: https://github.com/Naresh1318/Adversarial_Autoencoder
@author: mtodaka

:latent, flower, semi-supervised 
:latent, unsupervised
:latent, semi-supervised
:latent, semi-supervised round shape

: learning rate schedule
: early stopping

:tying weights?
:CNN, descriminator
:CNN, more layers on encoder
:CNN, decoder
: l2 
: max-norm regularization
: BN, heavy and did not work. want to try again?
: ae train by layers
: ae pretrain, then semi-supervised?
:pre train layer by layers, then final training.
:group the dense layer naming
:feed mode to run ./file
:param tuning, batch size?
"""

"""
modes:
1: Discriminator Training. Just to understand the dynamics.
2: Latent regulation. train generator to fool Descriminator with reconstruction constraint.
0: Showing latest model results. InOut, true dist, discriminator, latent dist.
"""
# you may want to change these to control your experiments
exptitle =  '10Lf' #experiment title that goes in tensorflow folder name

moderestore = '20171210131913_2_10Lf_realbs40'
mode= 0
flg_graph = True # showing graphs or not during the training. Showing graphs significantly slows down the training.
n_leaves = 10 # number of leaves in the mixed 2D Gaussian
OoTWeight = 0.01 # out of target weight in generator
DtTWeight = 0.001 # distance to target weight
n_latent_sample = 5000 # latent code visualization sample
n_step_dc = 0*n_leaves # mode 2, descriminator training steps
n_epochs_ge = 18*n_leaves # mode 3, generator training epochs
ac_batch_size = 100  # autoencoder training batch size
tb_batch_size = 800  # x_inputs batch size for tb
tb_log_step = 200 # tb logging step
flg_console_log = False # console logging swtich
import numpy as np
blanket_resolution = 10*int(np.sqrt(n_leaves)) # blanket resoliution for descriminator or its contour plot
dc_real_batch_size = int(blanket_resolution*blanket_resolution/40) # descriminator training real dist samplling batch size

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from datetime import datetime
import os
import matplotlib
#matplotlib.use('GTKAgg') # to speed up graphing
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import random
from math import cos,sin
from tqdm import tqdm

input_dim = 784
xLU = [-10,10] # blanket x axis lower and upper
yLU = [-10,10] # blanket y axis lower and upper
n_l1 = 1000
n_l2 = 1000
z_dim = 2
results_path = './Results/Adversarial_Autoencoder'

# reset graph
tf.reset_default_graph()

# Get the MNIST data
mnist = input_data.read_data_sets('./Data', one_hot=True)

# Placeholders for input data and the targets
x_input = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='Input')
real_distribution = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='Real_distribution')
unif_distribution = tf.placeholder(dtype=tf.float32, shape=[blanket_resolution*blanket_resolution, z_dim], name='Uniform_distribution')
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
"""
Functions
"""
def form_results():
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    folder_name = "/{0}_{1}_{2}". \
        format(datetime.now().strftime("%Y%m%d%H%M%S"), mode,exptitle)
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.makedirs(results_path + folder_name)
        os.makedirs(tensorboard_path)
        os.makedirs(saved_model_path)
        os.makedirs(log_path)
    return tensorboard_path, saved_model_path, log_path

def show_inout(sess,op):
    """
    Shows input MNIST image and reconstracted image.
    Randomly select 10 images from training dataset.
    Parameters. seess:TF session. op: autoencoder operation
    No return. Displays image.
    """
    if not flg_graph:
        return
    idx = random.sample(range(mnist[0].num_examples),ac_batch_size)
    img_in = mnist[0].images[idx,:]
    img_out = sess.run(op, feed_dict={x_input: img_in, is_training:False})
    img_out_s = img_out.reshape(ac_batch_size,28,28)
    img_in_s = img_in.reshape(ac_batch_size,28,28)
    #.reshape(10,28,28)
    plt.rc('figure', figsize=(15, 3))
    plt.tight_layout()
    for i in range(10):
        plt.subplot(2,10,i+1)
        plt.imshow(img_in_s[i],cmap="gray")
        plt.axis('off')
        plt.subplot(2,10,10+i+1)
        plt.imshow(img_out_s[i],cmap="gray")
        plt.axis('off')
    
    plt.suptitle("Original(1st row) and Decoded(2nd row)")
    plt.show()
    plt.close()

def show_latent_code(sess,spc):
    """
    Shows latent codes distribution based on all MNIST training images, with lables as color.
    Parameters. seess:TF session.
                spc: sample per class
    No return. Displays image.
    """
    if not flg_graph:
        return
    plt.rc('figure', figsize=(5, 5))
    plt.tight_layout()
    
    with tf.variable_scope("Encoder"):
        test_zs = sess.run(encoder(mnist[0].images, reuse=True),feed_dict={is_training:False})
    ytrain = np.argmax(mnist[0].labels,axis=1)
    
    cm = plt.cm.get_cmap('tab10')
    fig, ax = plt.subplots(1)
    
    for i in range(10):
        y=test_zs[np.where(ytrain==i),1][0,0:spc]
        x=test_zs[np.where(ytrain==i),0][0,0:spc]
        color = cm(i)
        ax.scatter(x, y, label=str(i), alpha=0.9, facecolor=color, linewidth=0.02, s = 2)
    
    ax.legend(loc='center left', markerscale = 8, bbox_to_anchor=(1, 0.5))
    ax.set_title('2D latent code')    
    plt.show()
    plt.close()
    
def show_discriminator(sess):
    """
    Shows discriminator activation contour plot. Close to 1 means estimated as positive (true dist).
    Parameters. seess:TF session.
    No return. Displays image.
    """
    if not flg_graph:
        return
    xlist, ylist, blanket = get_blanket(blanket_resolution)
    
    plt.rc('figure', figsize=(6, 5))
    plt.tight_layout()
    
    X, Y = np.meshgrid(xlist, ylist)    
    
    with tf.variable_scope("Discriminator"):
        desc_result = sess.run(tf.nn.sigmoid(discriminator(blanket, reuse=True)),feed_dict={is_training:False})

    Z = np.empty((blanket_resolution,blanket_resolution), dtype="float32")    
    for i in range(blanket_resolution):
        for j in range(blanket_resolution):
            Z[i][j]=desc_result[i*blanket_resolution+j]

    fig, ax = plt.subplots(1)
    cp = ax.contourf(X, Y, Z)
    plt.colorbar(cp)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Descriminator Contour')    
    plt.show()   
    plt.close()
    
def get_blanket(resolution):
    resolution = resolution
    xlist = np.linspace(xLU[0], xLU[1], resolution,dtype="float32")
    ylist = np.linspace(yLU[0], yLU[1], resolution,dtype="float32")
    blanket = np.empty((resolution*resolution,2), dtype="float32")
    for i in range(resolution):
        for j in range(resolution):
            blanket[i*resolution+j]=[xlist[j],ylist[i]]
    return xlist,ylist,blanket
            
def show_real_dist(z_real_dist):
    """
    Shows real distribution
    Parameters. z_real_dist:(batch_size,2) numpy array
    No return. Displays image.
    """
    if not flg_graph:
        return
    plt.rc('figure', figsize=(5, 5))
    plt.tight_layout()
    fig, ax = plt.subplots(1)
    ax.scatter(z_real_dist[:,0], z_real_dist[:,1], alpha=0.9, linewidth=0.15)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Real Distribution')   
    plt.xlim(xLU[0],xLU[1])
    plt.ylim(yLU[0],yLU[1])
    plt.show()
    plt.close()

def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

def mlp_enc(x): # multi layer perceptron
    with tf.contrib.framework.arg_scope(
            [fully_connected],
            weights_initializer=he_init):
        elu1 = fully_connected(x, n_l1,scope='elu1')
        elu2 = fully_connected(elu1, n_l2,scope='elu2')
    return elu2

def mlp_dec(x): # multi layer perceptron
    with tf.contrib.framework.arg_scope(
            [fully_connected],
            weights_initializer=he_init):
        elu2 = fully_connected(x, n_l2,scope='elu2')
        elu1 = fully_connected(elu2, n_l1,scope='elu1')
    return elu1

def mlp_descriminator(x): # multi layer perceptron
    with tf.contrib.framework.arg_scope(
            [fully_connected],
            weights_initializer=he_init):
        relu1 = fully_connected(x, 50,scope='relu1')
        relu2 = fully_connected(relu1, 100,scope='relu2')
        relu3 = fully_connected(relu2, 150,scope='relu3')
        relu4 = fully_connected(relu3, 200,scope='relu4')
        relu5 = fully_connected(relu4, 250,scope='relu5')
    return relu5

def encoder(x, reuse=False):
    """
    Encoder part of the autoencoder.
    :param x: input to the autoencoder
    :param reuse: True -> Reuse the encoder variables, False -> Create the variables
    :return: tensor which is the hidden latent variable of the autoencoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    last_layer = mlp_enc(x)
    output = fully_connected(last_layer, z_dim,weights_initializer=he_init, scope='None7',activation_fn=None)
    return output

def decoder(x, reuse=False):
    """
    Decoder part of the autoencoder.
    :param x: input to the decoder
    :param reuse: True -> Reuse the decoder variables, False -> Create the variables
    :return: tensor which should ideally be the input given to the encoder.
    tf.sigmoid
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    last_layer = mlp_dec(x)
    output = fully_connected(last_layer, input_dim, weights_initializer=he_init, scope='Sigmoid7', activation_fn=tf.sigmoid)
    return output

def discriminator(x, reuse=False):
    """
    Discriminator that leanes to activate at true distribution and not for the others.
    :param x: tensor of shape [batch_size, z_dim]
    :param reuse: True -> Reuse the discriminator variables, False -> Create the variables
    :return: tensor of shape [batch_size, 1]. I think it's better to take sigmoid here.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    last_layer = mlp_dec(x)
    output = fully_connected(last_layer, 1, weights_initializer=he_init, scope='None7',activation_fn=None)
    return output

def gaussian_mixture(batchsize, ndim, num_leaves):
    """
    Crate true distribution with num_leaves 2D Gaussian
    :batch_size: number of data points to generate
    :ndim: latet code dimention =2
    :return: tensor of shape [batch_size, 2]. I think it's better to take sigmoid here.
    """
    if ndim % 2 != 0:
        raise Exception("ndim must be a multiple of 2.")
    def sample(x, y, label, num_leaves):
        shift = 1.7
        r = 2.0 * np.pi / float(num_leaves) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.07
    x = np.random.normal(0, x_var, (batchsize, ndim // 2))
    y = np.random.normal(0, y_var, (batchsize, ndim // 2))
    z = np.empty((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], random.randint(0, num_leaves - 1), num_leaves)
    return z

def model_restore(saver,pmode,mname=''):
    if pmode == -1 or pmode == 0: # running all or show results -> get the specified model or ese latest one
        if len(mname) > 0:
            all_results = [mname]
        else:
            all_results = [path for path in os.listdir(results_path)] 
    else: # get previous mode
        all_results = [path for path in os.listdir(results_path) if '_'+str(pmode-1)+'_' in path or '_-1_' in path] 
    all_results.sort()
    saver.restore(sess, save_path=tf.train.latest_checkpoint(results_path + '/' + all_results[-1] + '/Saved_models/'))


def cdist(A,B,agg):
    # combinatory distance
    # for each row in A, measure the euclidian distance with all the other rows in B, take min or avg
    Bmod = tf.tile(tf.expand_dims(B,0),[tf.shape(A)[0],1,1])
    Amod = tf.tile(tf.expand_dims(A,1),[1,tf.shape(B)[0],1])
    sdif = tf.squared_difference(Amod,Bmod)
    if agg=='min':
        rtn = tf.reduce_min(tf.sqrt(tf.reduce_sum(sdif,axis=2)),axis=1)
    if agg=='mean':
        rtn = tf.reduce_mean(tf.reduce_sum(sdif,axis=2),axis=1) # optimizer generate nan if this take sqrt
    return rtn
            
"""
Defining key operations, Loess, Optimizer and other necessary operations
"""
with tf.variable_scope('Encoder'):
    encoder_output = encoder(x_input)
    
with tf.variable_scope('Decoder'):
    decoder_output = decoder(encoder_output)

with tf.variable_scope('Discriminator'):
    d_real = discriminator(real_distribution)
    d_blanket = discriminator(unif_distribution, reuse=True)
    d_fake = discriminator(encoder_output, reuse=True)
    
# loss
with tf.name_scope("ae_loss"):
    autoencoder_loss = tf.reduce_mean(tf.square(x_input - decoder_output))
    
with tf.name_scope("dc_loss"):
    dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
    dc_loss_blanket = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_blanket), logits=d_blanket))
    dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
    dc_loss = dc_loss_blanket + dc_loss_real+dc_loss_fake

with tf.name_scope("ge_loss"):
    #Out of Target penalty
    OoT_penalty =OoTWeight*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))
    dist_to_target = DtTWeight*tf.reduce_max(cdist(encoder_output, real_distribution,agg='min'))
    generator_loss = autoencoder_loss+OoT_penalty #not sure why it averages out

#optimizer
all_variables = tf.trainable_variables()
dc_var = [var for var in all_variables if 'Discriminator/' in var.name]
ae_var = [var for var in all_variables if ('Encoder/' in var.name or 'Decoder/' in var.name)]

with tf.name_scope("AE_optimizer"):
    autoencoder_optimizer = tf.train.AdamOptimizer().minimize(autoencoder_loss)

with tf.name_scope("DC_optimizer"):
    discriminator_optimizer = tf.train.AdamOptimizer().minimize(dc_loss, var_list=dc_var)
    
with tf.name_scope("GE_optimizer"):
    generator_optimizer = tf.train.AdamOptimizer().minimize(generator_loss, var_list=ae_var)

init = tf.global_variables_initializer()

# Reshape immages to display them
input_images = tf.reshape(x_input, [-1, 28, 28, 1])
generated_images = tf.reshape(decoder_output, [-1, 28, 28, 1])

# Tensorboard visualization
ae_sm = tf.summary.scalar(name='Autoencoder_Loss', tensor=autoencoder_loss)
dc_sm = tf.summary.scalar(name='Discriminator_Loss', tensor=dc_loss)
ge_sm = tf.summary.scalar(name='Generator_Loss', tensor=generator_loss)
oot_sm = tf.summary.scalar(name='OoT_penalty', tensor=OoT_penalty)
dtt_sm = tf.summary.scalar(name='dist_to_target', tensor=dist_to_target)
summary_op = tf.summary.merge_all()

# Creating saver and get ready
saver = tf.train.Saver()
step = 0

"""
Executing with a session based on mode specification
"""

def tb_init(sess): # create tb path, model path and return tb writer and saved model path
    tensorboard_path, saved_model_path, log_path = form_results()
    sess.run(init)
    writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)  
    return writer, saved_model_path

def tb_write(sess):
    batch_x, _ = mnist.train.next_batch(tb_batch_size)
    idx = random.sample(range(mnist[0].num_examples),dc_real_batch_size)
    z_real_batch = z_real_dist[idx,:]
#    if flg_console_log:
#        dc_loss_v,ge_loss_v,ae_loss_v,OoT_pen_v,dtt_v,sm \
#            = sess.run([dc_loss,generator_loss,autoencoder_loss,OoT_penalty,dist_to_target,summary_op] \
#            ,feed_dict={is_training:False, x_input: batch_x,real_distribution:z_real_batch,unif_distribution:blanket})
#        tqdm.write("ae_loss:{0:.5f},dc_loss:{1:.5f},Generator Loss:{2:.5f},OoT_pen:{3:.5f},DTT:{4:.5f}"\
#               .format(ae_loss_v,dc_loss_v,ge_loss_v,OoT_pen_v,dtt_v))
#    else:
    sm = sess.run(summary_op,feed_dict={is_training:False, x_input: batch_x,real_distribution:z_real_batch,unif_distribution:blanket})
    writer.add_summary(sm, global_step=step)

with tf.Session() as sess:
    if mode != 0: # some kind of training
        writer,saved_model_path = tb_init(sess)   
        _,_,blanket = get_blanket(blanket_resolution)
        z_real_dist = gaussian_mixture(55000, z_dim, n_leaves)
    if mode==1: # Descriminator training mode
        print("-------Descriminator Training--------")    
        show_inout(sess, op=decoder_output)  
        show_latent_code(sess,n_latent_sample)     
        show_real_dist(z_real_dist)
        show_discriminator(sess)
        batch_x, _ = mnist.train.next_batch(0)
        for i in tqdm(range(n_step_dc)):
            idx = random.sample(range(mnist[0].num_examples),dc_real_batch_size)
            z_real_batch = z_real_dist[idx,:]
            sess.run(discriminator_optimizer, feed_dict={is_training:True ,x_input: batch_x,real_distribution:z_real_batch,unif_distribution:blanket })
            if i % tb_log_step == 0:
                dc_loss_v,dc_summary = sess.run([dc_loss, dc_sm],
                                            feed_dict={is_training:False, x_input: batch_x,unif_distribution:blanket,real_distribution:z_real_batch})
                writer.add_summary(dc_summary, global_step=step)
                show_discriminator(sess)
                tqdm.write("Batch {} / {}, Descriminator Loss:{}".format(i,n_step_dc,dc_loss_v)) 
            step += 1
        show_discriminator(sess)
        saver.save(sess, save_path=saved_model_path, global_step=step, write_meta_graph = True)
    if mode==2: # Latent regulation
        n_batches = int(mnist.train.num_examples / ac_batch_size)
        for i in range(n_epochs_ge):
            print("------------------Epoch {}/{} ------------------".format(i, n_epochs_ge))
            for b in tqdm(range(n_batches)):    
                #Autoencoder
                    
                #Discriminator
                batch_x, _ = mnist.train.next_batch(ac_batch_size)    
                idx = random.sample(range(mnist[0].num_examples),dc_real_batch_size)
                z_real_batch = z_real_dist[idx,:]
                sess.run([discriminator_optimizer],feed_dict=\
                 {is_training:True, x_input: batch_x, real_distribution:z_real_batch,unif_distribution:blanket })
                #Generator
                sess.run([generator_optimizer],feed_dict=\
                 {is_training:True, x_input: batch_x,real_distribution:z_real_batch})
                if b % tb_log_step == 0:
                    show_discriminator(sess)
                    show_latent_code(sess,n_latent_sample)
                    tb_write(sess)
                step += 1
        saver.save(sess, save_path=saved_model_path, global_step=step, write_meta_graph = True)
    if mode != 0: # some kind of training
        writer.close()
    if mode==0: # showing the latest model result. InOut, true dist, discriminator, latent dist.
        model_restore(saver,mode,moderestore)
        show_inout(sess, op=decoder_output) 
        z_real_dist = gaussian_mixture(55000, z_dim, n_leaves)
        show_real_dist(z_real_dist)
        show_discriminator(sess)    
        show_latent_code(sess,n_latent_sample)
    
        