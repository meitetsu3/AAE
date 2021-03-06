# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:50:32 2017
ref: https://github.com/Naresh1318/Adversarial_Autoencoder
@author: mtodaka
: one combined discriminator with categorical
: hold off Y regulation for a while with maxnorm?
: discriminator vis and initial training
: stop when autoencoder loss increase on validation set
: separate? reconstruction, descriminator, generater, semi-supervised
: learning rate schedule
:feed mode to run ./file
"""

"""
modes:
1: Latent regulation. train generator to fool Descriminator with reconstruction constraint.
0: Showing latest model results. InOut, true dist, discriminator, latent dist.
"""
exptitle =  'lbl100base1' #experiment title that goes in tensorflow folder name
mode = 1
flg_graph = False # showing graphs or not during the training. Showing graphs significantly slows down the training.
model_folder = '' # name of the model to be restored. white space means most recent.
n_label = 100 # number of labels used in semi-supervised training
bs_ae = 2000  # autoencoder training batch size
bs_ss = 16 # semi-supervised training batch size
keep_prob = 1.00 # keep probability of drop out
w_zfool = 0.01 # weight on z fooling
w_yfool = 0.01 # weight on y fooling
w_classfication = 0.01 #classification weight in generator
w_VtxReg = 0.01 # Training distance to vertex weight
w_ae_loss = 1.00 # weight on autoencoding reconstuction loss
jit_std = 0.05 # Y real jittering stdev
n_leaves = 10 # number of leaves in the mixed 2D Gaussian
n_epochs_ge = 5*n_leaves # mode 3, generator training epochs
n_pretrain = 0 # pre supervised training step

import numpy as np
res_blanket = 10*int(np.sqrt(n_leaves)) # blanket resoliution for descriminator or its contour plot
bs_z_real = int(res_blanket*res_blanket/15) # descriminator training z real dist samplling batch size
bs_ae_tb = 800  # x_inputs batch size for tb
step_tb_log = 800  # tb logging step
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2 # can run up to 4 threads on main GPU, and 5 on others.
#config.log_device_placement = True
#config.allow_soft_placement=True
x_blanket_vis = 5 # x to the blanket resolution for descriminator contour plot
myColor = ['black','orange', 'red', 'blue','gray','green','pink','cyan','lime','magenta']
input_dim = 784
xLU = [-10,10] # blanket x axis lower and upper
yLU = [-10,10] # blanket y axis lower and upper
n_l1 = 1000
n_l2 = 1000
z_dim = 2
results_path = './Results/Adversarial_Autoencoder'

from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import dropout
from datetime import datetime
import os
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import random
from tqdm import tqdm

# reset graph
tf.reset_default_graph()

# Get the MNIST data
mnist = input_data.read_data_sets('./Data', one_hot=True)

# Placeholders for input data and the targets
x_auto = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x_auto')
y_test = tf.placeholder(dtype=tf.float32, shape=[None,10],name = 'y_test')
x_train = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x_train')
y_train = tf.placeholder(dtype=tf.float32, shape=[None,10],name = 'y_train')
z_real = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='z_real')
z_blanket = tf.placeholder(dtype=tf.float32, shape=[res_blanket*res_blanket, z_dim], name='z_blanket')
y_Zblanket = tf.placeholder(dtype=tf.float32, shape=[None,10],name = 'y_Zblanket')
y_Zreal = tf.placeholder(dtype=tf.float32, shape=[None,10],name = 'y_Zreal')
y_real = tf.placeholder(dtype=tf.float32, shape=[None,10],name = 'y_real') # jittering
y_blanket = tf.placeholder(dtype=tf.float32, shape=[None,10],name = 'y_blanket') # not integers

is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

"""
Util Functions
"""

def form_results():
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    folder_name = "/{0}_{1}_{2}".format(datetime.now().strftime("%Y%m%d%H%M%S"), mode,exptitle)
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.makedirs(results_path + folder_name)
        os.makedirs(tensorboard_path)
        os.makedirs(saved_model_path)
        os.makedirs(log_path)
    return tensorboard_path, saved_model_path, log_path

def get_blanket(resolution):
    resolution = resolution
    xlist = np.linspace(xLU[0], xLU[1], resolution,dtype="float32")
    ylist = np.linspace(yLU[0], yLU[1], resolution,dtype="float32")
    blanket = np.empty((resolution*resolution,2), dtype="float32")
    for i in range(resolution):
        for j in range(resolution):
            blanket[i*resolution+j]=[xlist[j],ylist[i]]
    return xlist,ylist,blanket

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

def next_batch(x, y, batch_size):
    random_index = np.random.permutation(np.arange(len(x)))[:batch_size]
    return x[random_index], y[random_index]  
       
"""
Vis Functions
"""
def show_inout(sess,op):
    """
    Shows input MNIST image and reconstracted image.
    Randomly select 10 images from training dataset.
    Parameters. seess:TF session. op: autoencoder operation
    No return. Displays image.
    """
    if not flg_graph:
        return
    idx = random.sample(range(mnist[0].num_examples),bs_ae)
    img_in = mnist[0].images[idx,:]
    img_out = sess.run(op, feed_dict={x_auto: img_in,is_training:False})
    img_out_s = img_out.reshape(bs_ae,28,28)
    img_in_s = img_in.reshape(bs_ae,28,28)
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

def show_latent_code(sess):
    """
    Shows latent codes distribution (y and z) based on all MNIST test images, and print accuracy
    Parameters. seess:TF session.
    """
    if not flg_graph:
        return
    plt.rc('figure', figsize=(8, 8))
    plt.tight_layout()
    
    with tf.variable_scope("Encoder"):
        train_zs,train_ys = sess.run(encoder(mnist[2].images, reuse=True), feed_dict={is_training:False}) #2 is test, 10k images
    train_yi = np.argmax(train_ys, axis=1) 
    zm = gaussian_mixture(train_zs, n_leaves, train_yi)
    train_lbl = mnist[2].labels
    train_lbli = np.argmax(train_lbl, axis=1)
    cm = matplotlib.colors.ListedColormap(myColor)
    fig, ax = plt.subplots(1)
    
    for i in range(n_leaves):
        y=zm[np.where(train_lbl[:,i]==1),1][0,:]
        x=zm[np.where(train_lbl[:,i]==1),0][0,:]
        color = cm(i)
        ax.scatter(x, y, label=str(i), alpha=0.9, facecolor=color, linewidth=0.02, s = 10)
    
    ax.legend(loc='center left', markerscale = 3, bbox_to_anchor=(1, 0.5))
    ax.set_title('2D latent code')    
    plt.show()
    plt.close()
    
    test_accuracy = 100.0*np.mean(np.equal(train_lbli, train_yi))
    
    np.set_printoptions(precision=1)
    print("Test Accuracy:{}%".format(test_accuracy))
    print("Y logits of top 5 test images :")
    print(train_ys[0:4])
    
def show_z_discriminator(sess,digit):
    """
    Shows z discriminator activation contour plot. Close to 1 means estimated as positive (true dist).
    Parameters. seess:TF session.
    No return. Displays image.
    """
    if not flg_graph:
        return
    br = x_blanket_vis*res_blanket
    xlist, ylist, blanket = get_blanket(br)
    
    if digit==-1:
        #y_input = (np.random.uniform(-100,100,10*br*br)).astype('float32').reshape(br*br,10)
        y_input = np.full([br*br,10],0.05,dtype="float32")
    else:
        y_input = np.eye(10,dtype="float32")[np.full([br*br],digit)]
    
    plt.rc('figure', figsize=(6, 5))
    plt.tight_layout()
    
    X, Y = np.meshgrid(xlist, ylist)    
    
    with tf.variable_scope("DiscriminatorZ"):
        desc_result = sess.run(tf.nn.sigmoid(discriminator_z(blanket,y_input, reuse=True)),\
                               feed_dict={is_training:False})

    Z = np.empty((br,br), dtype="float32")    
    for i in range(br):
        for j in range(br):
            Z[i][j]=desc_result[i*br+j]

    fig, ax = plt.subplots(1)
    cp = ax.contourf(X, Y, Z)
    plt.colorbar(cp)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Z Descriminator Contour for {}'.format(digit))    
    plt.show()   
    plt.close()
    
def show_z_dist(z_real_dist):
    """
    Shows z distribution
    Parameters. z_real_dist:(batch_size,2) numpy array
    No return. Displays image.
    """
    if not flg_graph:
        return
    plt.rc('figure', figsize=(5, 5))
    plt.tight_layout()
    fig, ax = plt.subplots(1)

    ax.scatter(z_real_dist[:,0],z_real_dist[:,1], alpha=0.9, linewidth=0.15,s = 5)

    ax.legend(loc='center left',  markerscale = 5, bbox_to_anchor=(1, 0.5))
    ax.set_title('Real Z Distribution')
    
    plt.xlim(xLU[0],xLU[1])
    plt.ylim(yLU[0],yLU[1])
    plt.show()
    plt.close()

"""
model Functions
"""

def mlp_enc(x): # multi layer perceptron
    with tf.contrib.framework.arg_scope(
            [fully_connected],
            weights_initializer=he_init):
        X_drop = dropout(x, keep_prob, is_training = is_training)
        relu1 = fully_connected(X_drop, n_l1,scope='relu1')
        relu1_drop = dropout(relu1, keep_prob, is_training=is_training)
        relu2 = fully_connected(relu1_drop, n_l2,scope='relu2')
        relu2_drop = dropout(relu2, keep_prob, is_training=is_training)
    return relu2_drop

def mlp_dec(x): # multi layer perceptron
    with tf.contrib.framework.arg_scope(
            [fully_connected],
            weights_initializer=he_init):
        X_drop = dropout(x, keep_prob, is_training = is_training)
        relu2 = fully_connected(X_drop, n_l2,scope='relu2')
        relu2_drop = dropout(relu2, keep_prob, is_training=is_training)
        relu1 = fully_connected(relu2_drop, n_l1,scope='relu1')
        relu1_drop = dropout(relu1, keep_prob, is_training=is_training)
    return relu1_drop

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
    outputZ = fully_connected(last_layer, z_dim,weights_initializer=he_init, scope='linZ',activation_fn=None)
    outputY = fully_connected(last_layer, n_leaves, weights_initializer=he_init,\
                              scope='linY',activation_fn=None)
    
    return outputZ,outputY

def decoder(z,y, reuse=False):
    """
    Decoder part of the autoencoder.
    :param x: input to the decoder
    :param reuse: True -> Reuse the decoder variables, False -> Create the variables
    :return: tensor which should ideally be the input given to the encoder.
    tf.sigmoid
    """
    x = tf.concat([z,y],axis=1)
    if reuse:
        tf.get_variable_scope().reuse_variables()
    last_layer = mlp_dec(x)
    output = fully_connected(last_layer, input_dim, weights_initializer=he_init,scope='Sigmoid',\
                             activation_fn=tf.sigmoid)
    return output

def discriminator_z(x,y, reuse=False):
    """
    Discriminator that leanes to activate at true distribution and not for the others.
    :param x: tensor of shape [batch_size, z_dim]
    :param y: predicted class. We need to feed this to uncorrelate 
    :param reuse: True -> Reuse the discriminator variables, False -> Create the variables
    :return: tensor of shape [batch_size, 1]. I think it's better to take sigmoid here.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    last_layer = mlp_dec(tf.concat([x,y],1))
    output = fully_connected(last_layer, 1, weights_initializer=he_init, scope='None',activation_fn=None)
    return output

def discriminator_y(y, reuse=False):
    """
    Discriminator that leanes to activate at true distribution and not for the others.
    :param y: predicted class.
    :param reuse: True -> Reuse the discriminator variables, False -> Create the variables
    :return: tensor of shape [batch_size, 1]. I think it's better to take sigmoid here.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    last_layer = mlp_dec(y)
    output = fully_connected(last_layer, 1, weights_initializer=he_init, scope='None',activation_fn=None)
    return output


def gaussian_mixture(z, num_leaves, selector):
    """
    selector to distribute cluster faceFalse
    from the cluseter face, place z, which is 2D latent distribution
    """
    def clulster_face(num_l, sel):
        shift = 9
        r = (2.0 * np.pi / float(num_l)) * sel
        new_x = shift * np.cos(r)
        new_y = shift * np.sin(r)
        return np.transpose(np.array([new_x, new_y])).reshape((len(sel),2))

    z = z+clulster_face(num_leaves,selector)
    
    return z

def gaussian(batchsize):
    """
    Crate true z, 2D standard normal distribution
    """
    return np.random.normal(0, 1, (batchsize, 2))
 
"""
Defining key operations, Loess, Optimizer and other necessary operations
"""
with tf.variable_scope('Encoder'):
    encoder_outputZ,encoder_outputYlogits = encoder(x_auto)
    _,trainer_ylogits = encoder(x_train, reuse=True)
    
with tf.variable_scope('Decoder'):
    decoder_output = decoder(encoder_outputZ,tf.one_hot(tf.argmax(encoder_outputYlogits, dimension = 1), \
                                                        depth = n_leaves))
    
with tf.variable_scope('DiscriminatorZ'):
    d_Zreal = discriminator_z(z_real, y_Zreal)
    d_Zblanket = discriminator_z(z_blanket, y_Zblanket,reuse=True)
    d_Zfake = discriminator_z(encoder_outputZ,tf.one_hot(tf.argmax(encoder_outputYlogits, dimension = 1), \
                                                        depth = n_leaves),reuse=True)

with tf.variable_scope('DiscriminatorY'):
    d_Yreal = discriminator_y(y_real)
    d_Yblanket = discriminator_y(y_blanket,reuse=True)
    d_Yfake = discriminator_y(encoder_outputYlogits,reuse=True)
    
# loss 
with tf.name_scope('Y_regulation'):
    with tf.name_scope('unsupervised_simplex_distToVertex'):
        Yonehot = tf.one_hot(tf.argmax(encoder_outputYlogits, dimension = 1), depth = n_leaves)
        dist_to_vertexT = w_VtxReg*tf.reduce_mean(tf.reduce_sum(tf.square( y_train- trainer_ylogits),axis=1))
        dist_to_vertex = tf.reduce_mean(tf.reduce_sum(tf.square( Yonehot- encoder_outputYlogits),axis=1))
        maxlogit = tf.reduce_mean(tf.reduce_max(encoder_outputYlogits,axis=1))
        minlogit = tf.reduce_mean(tf.reduce_min(encoder_outputYlogits,axis=1))
        sumlogit = tf.reduce_mean(tf.reduce_sum(encoder_outputYlogits,axis=1))
        
with tf.name_scope("dc_loss"):
    DCloss_Zreal = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
            labels=tf.ones_like(d_Zreal), logits=d_Zreal))
    DCloss_Zblanket = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
            labels=tf.zeros_like(d_Zblanket), logits=d_Zblanket))
    DCloss_Zfake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
            labels=tf.zeros_like(d_Zfake), logits=d_Zfake))
    
    DCloss_Yreal = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
            labels=tf.ones_like(d_Yreal), logits=d_Yreal))
    DCloss_Yblanket = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
            labels=tf.zeros_like(d_Yblanket), logits=d_Yblanket))
    DCloss_Yfake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
            labels=tf.zeros_like(d_Yfake), logits=d_Yfake))
    
    dc_Zloss = DCloss_Zblanket + DCloss_Zreal+DCloss_Zfake
    dc_Yloss = DCloss_Yblanket + DCloss_Yreal+DCloss_Yfake
    dc_loss = dc_Zloss+dc_Yloss

with tf.name_scope("ge_loss"):
    autoencoder_loss = w_ae_loss*tf.reduce_mean(tf.square(x_auto - decoder_output))
    classification_loss = w_classfication*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
            labels=y_train,logits=trainer_ylogits, name="classification_loss"))
    d_Zfooling =w_zfool*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
            labels=tf.ones_like(d_Zfake), logits=d_Zfake))
    d_Yfooling =w_yfool*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
            labels=tf.ones_like(d_Yfake), logits=d_Yfake))
    generator_loss = autoencoder_loss+d_Zfooling+d_Yfooling+classification_loss+dist_to_vertexT

# metrics
with tf.name_scope('accuracy'):
    with tf.name_scope('train_accuracy'):
        t_correct_prediction = tf.equal(tf.argmax(y_train, 1), tf.argmax(trainer_ylogits, 1))
        t_accuracy = tf.reduce_mean(tf.cast(t_correct_prediction, tf.float32))
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_test, 1), tf.argmax(encoder_outputYlogits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#optimizer
all_variables = tf.trainable_variables()
dc_var = [var for var in all_variables if ('DiscriminatorZ/' in var.name or 'DiscriminatorY/' in var.name)]
ae_var = [var for var in all_variables if ('Encoder/' in var.name or 'Decoder/' in var.name)]
a_var = [var for var in all_variables if ('Encoder/' in var.name)]

with tf.name_scope("AE_optimizer"):
    autoencoder_optimizer = tf.train.AdamOptimizer().minimize(autoencoder_loss)

with tf.name_scope("DC_optimizer"):
    discriminator_optimizer = tf.train.AdamOptimizer().minimize(dc_loss, var_list=dc_var)

with tf.name_scope("Y_optimizer"):
    Y_optimizer = tf.train.AdamOptimizer().minimize(dist_to_vertex, var_list=a_var)    
    
with tf.name_scope("GE_optimizer"):
    generator_optimizer = tf.train.AdamOptimizer().minimize(generator_loss, var_list=ae_var)


init = tf.global_variables_initializer()

# Reshape immages to display them
input_images = tf.reshape(x_auto, [-1, 28, 28, 1])
generated_images = tf.reshape(decoder_output, [-1, 28, 28, 1])

# Tensorboard visualizationdegit_veye
tf.summary.scalar(name='Autoencoder_Loss', tensor=autoencoder_loss)
tf.summary.scalar(name='dc_Zloss', tensor=dc_Zloss)
tf.summary.scalar(name='dc_Yloss', tensor=dc_Yloss)
tf.summary.scalar(name='dc_loss', tensor=dc_loss)
tf.summary.scalar(name='Generator_Loss', tensor=generator_loss)
tf.summary.scalar(name='d_Zfooling', tensor=d_Zfooling)
tf.summary.scalar(name='d_Yfooling', tensor=d_Yfooling)
tf.summary.scalar(name='Classification_Loss', tensor=classification_loss)
tf.summary.scalar(name='Accuracy', tensor=accuracy)
tf.summary.scalar(name='Training_Accuracy', tensor=t_accuracy)
tf.summary.scalar(name='Distance_to_Vertex', tensor=dist_to_vertex)
tf.summary.scalar(name='Training_Distance_to_Vertex', tensor=dist_to_vertexT)
tf.summary.scalar(name='maxlogit', tensor=maxlogit)
tf.summary.scalar(name='minlogit', tensor=minlogit)
tf.summary.scalar(name='sumlogit', tensor=sumlogit)

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
    test_x, test_y = mnist.test.next_batch(bs_ae_tb) # to evalute accuracy with unseen data.
    #test_x, test_y = mnist.train.next_batch(tb_batch_size) # debuging. want to see the training stats
    # use the priviousely generated data for others
    sm = sess.run(summary_op,feed_dict={is_training:False, x_auto:test_x, y_test:test_y,y_real:real_y, \
            x_train:alltrain_x, y_train:alltrain_y, z_real:real_z,y_Zblanket:Zblanket_y,y_Zreal:Zreal_y\
            ,z_blanket:blanket_z, y_blanket:blanket_y})
    writer.add_summary(sm, global_step=step)

with tf.Session(config=config) as sess:
    if mode==1: # Latent regulation
        writer,saved_model_path = tb_init(sess)   
        _,_,blanket_z = get_blanket(res_blanket)
        n_batches = int(mnist.train.num_examples / bs_ae)
        alltrain_x, alltrain_y = mnist.train.next_batch(n_label)
        for i in range(n_epochs_ge):
            print("------------------Epoch {}/{} ------------------".format(i, n_epochs_ge))
            for b in tqdm(range(n_batches)):    
                #Discriminator
                auto_x, _ = mnist.train.next_batch(bs_ae)
                Zreal_y = np.eye(10)[np.random.randint(0,n_leaves, size=bs_z_real)]
                real_y = np.eye(10)[np.random.randint(0,n_leaves, size=bs_z_real)]+np.random.normal(0.0,jit_std,(bs_z_real,10))
                Zblanket_y = np.eye(10)[np.random.randint(0,n_leaves, size=res_blanket*res_blanket)]
                real_z = gaussian(bs_z_real)
          
                blanket_y = (np.random.uniform(-3,3,10*res_blanket*res_blanket)).astype('float32').reshape(res_blanket*res_blanket,10)
      
                sess.run([discriminator_optimizer],feed_dict={is_training:True,\
                        x_auto:auto_x, y_Zreal:Zreal_y, y_Zblanket:Zblanket_y,y_real:real_y,\
                        z_real:real_z, z_blanket:blanket_z, y_blanket:blanket_y})
    
                #Generator - autoencoder, fooling descriminator, and y semi-supervised classification
                train_xb, train_yb = next_batch(alltrain_x,alltrain_y,bs_ss)
                
                sess.run([generator_optimizer],feed_dict={is_training:True,\
                         x_auto:auto_x, x_train:train_xb, y_train:train_yb})
                if b % step_tb_log == 0:
                    show_z_discriminator(sess,1)  
                    show_z_discriminator(sess,4)  # [0,0,0,0,1,0,0,0,0,0]
                    show_z_discriminator(sess,-1) 
                    show_latent_code(sess)
                    tb_write(sess)
                step += 1
        saver.save(sess, save_path=saved_model_path, global_step=step, write_meta_graph = True)
        writer.close()
    if mode==0: # showing the latest model result. InOut, true dist, discriminator, latent dist.
        model_restore(saver,mode,model_folder)
        show_inout(sess, op=decoder_output)         
        real_z= gaussian(5000)
        show_z_dist(real_z)
        show_z_discriminator(sess,1)  
        show_z_discriminator(sess,4)  # [0,0,0,0,1,0,0,0,0,0]
        show_z_discriminator(sess,-1) 
        vtx = np.array([[1,0,0,0,0,0,0,0,0,0]
                        ,[0,1,0,0,0,0,0,0,0,0]
                        ,[0,0,1,0,0,0,0,0,0,0]
                        ,[0,0,0,1,0,0,0,0,0,0]
                        ,[0,0,0,0,1,0,0,0,0,0]
                        ,[0,0,0,0,0,1,0,0,0,0]
                        ,[0,0,0,0,0,0,1,0,0,0]
                        ,[0,0,0,0,0,0,0,1,0,0]
                        ,[0,0,0,0,0,0,0,0,1,0]
                        ,[0,0,0,0,0,0,0,0,0,1]],dtype='float32')
        nvtx = np.array([[0,0,0,0,0,0,0,0,0,0]
                        ,[0.1,0,0,0,0,0,0,0,0,0]
                        ,[5,5,5,5,5,5,5,5,5,5]
                        ,[0,0,0,0.3,0,0,0,0,0,0]
                        ,[0,0,0,0,0.4,0,0,0,0,0]
                        ,[0,0,0,0,0,0.5,0,0,0,0]
                        ,[0,0,0,0,0,0,0.9,0,0,0]
                        ,[0,0,0,0,0,0,0,0.95,0,0]
                        ,[0,0,0,0,0,0,0,0,0.99,0]
                        ,[1,0,0,0,0,0,0,0,0,0.99]],dtype='float32')
        with tf.variable_scope("DiscriminatorY"):
            v = sess.run(tf.nn.sigmoid(discriminator_y(vtx, reuse=True)),feed_dict={is_training:False})
            nv = sess.run(tf.nn.sigmoid(discriminator_y(nvtx, reuse=True)),feed_dict={is_training:False})
        print(v)
        print(nv)
        show_latent_code(sess)
        
            
        
        