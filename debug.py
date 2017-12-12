
# you may want to change these to control your experiments
mode= -1
flg_graph = False # showing graphs or not during the training. Showing graphs significantly slows down the training.
n_leaves = 3 # number of leaves in the mixed 2D Gaussian

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
#from tensorflow.contrib.layers import batch_norm
import numpy as np
import matplotlib
matplotlib.use('GTKAgg') # to speed up graphing
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import random
from math import cos,sin
from tqdm import tqdm

n_epochs_ae = 2*n_leaves # mode 1, autoencoder training epochs
n_step_dc = 50*n_leaves # mode 2, descriminator training steps
n_epochs_ge = 70*n_leaves # mode 3, generator training epochs
ge_OoT_weight = 0.0001 # Out of terget weight to make reconstraction const more effective. 
                        #Dynamically update to the same scalse as autoencoder_loss 
ge_Dst_weight = 0.0001 # distance weight. to give a direction to on target.
ac_batch_size = 100  # autoencoder training batch size
ge_batch_size = ac_batch_size*n_leaves # generator training batch size
blanket_resolution = 100*int(np.sqrt(n_leaves)) # blanket resoliution for descriminator or its contour plot
dc_batch_size = blanket_resolution*blanket_resolution # descriminator training real dist samplling batch size

input_dim = 784
xLU = [-10,10] # blanket x axis lower and upper
yLU = [-10,10] # blanket y axis lower and upper
n_l1 = 500
n_l2 = 450
n_l3 = 400
n_l4 = 350
n_l5 = 300
n_l6 = 250
n_l7 = 200
n_l8 = 150
n_l9 = 100
n_l10 = 50
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

bn_params={
        'is_training': is_training,
        'decay': 0.99,
        'updates_collections': None
        }

he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")


def show_latent_code(sess):
    """
    Shows latent codes distribution based on all MNIST training images, with lables as color.
    Parameters. seess:TF session.
    No return. Displays image.
    """
    if not flg_graph:
        return
    plt.rc('figure', figsize=(5, 5))
    plt.tight_layout()
    
    with tf.variable_scope("Encoder"):
        test_zs = sess.run(encoder(mnist[0].images, reuse=True))
    ytrain = np.argmax(mnist[0].labels,axis=1)
    
    cm = plt.cm.get_cmap('tab10')
    fig, ax = plt.subplots(1)
    
    for i in range(10):
        y=test_zs[np.where(ytrain==i),1]
        x=test_zs[np.where(ytrain==i),0]
        color = cm(i)
        ax.scatter(x, y, label=str(i), alpha=0.9, facecolor=color, linewidth=0.15)
        
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('2D latent code')    
    plt.show()
    plt.close()
    
def mlp(x): # multi layer perceptron
    with tf.contrib.framework.arg_scope(
            [fully_connected],
            #normalizer_fn=batch_norm,
            #normalizer_params=bn_params,
            #weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
            activation_fn=tf.nn.elu,
            weights_initializer=he_init):
        elu1 = fully_connected(x, n_l1,scope='elu1')
        elu2 = fully_connected(elu1, n_l2,scope='elu2')
        elu3 = fully_connected(elu2, n_l3,scope='elu3')
        elu4 = fully_connected(elu3, n_l4,scope='elu4')
        elu5 = fully_connected(elu4, n_l5,scope='elu5')
        elu6 = fully_connected(elu5, n_l6,scope='elu6')
        elu7 = fully_connected(elu6, n_l7,scope='elu7')
        elu8 = fully_connected(elu7, n_l8,scope='elu8')
        elu9 = fully_connected(elu8, n_l9,scope='elu9')
        elu10 = fully_connected(elu9, n_l10,scope='elu10')
    return elu10

def encoder(x, reuse=False):
    """
    Encoder part of the autoencoder.
    :param x: input to the autoencoder
    :param reuse: True -> Reuse the encoder variables, False -> Create the variables
    :return: tensor which is the hidden latent variable of the autoencoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    last_layer = mlp(x)
    output = fully_connected(last_layer, z_dim,weights_initializer=he_init, scope='None7',activation_fn=None)
    
    return output

def decoder(x, reuse=False):
    """
    Decoder part of the autoencoder.
    :param x: input to the decoder
    :param reuse: True -> Reuse the decoder variables, False -> Create the variables
    :return: tensor which should ideally be the input given to the encoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    
    last_layer = mlp(x)
    output = fully_connected(last_layer, input_dim, weights_initializer=he_init, scope='Sigmoid7', activation_fn=tf.sigmoid)
    return output

def disttest(A,B):
    Bmod = tf.tile(tf.expand_dims(B,0),[tf.shape(A)[0],1,1])
    Amod = tf.tile(tf.expand_dims(A,1),[1,tf.shape(B)[0],1])
    sdif = tf.squared_difference(Amod,Bmod)
    #rtn = tf.reduce_min(tf.reduce_sum(sdif,axis=2),axis=1)
    rtn = tf.reduce_min(tf.reduce_sum(sdif,axis=2),axis=1)
    return rtn

def cdist(A,B,agg):
    Bmod = tf.tile(tf.expand_dims(B,0),[tf.shape(A)[0],1,1])
    Amod = tf.tile(tf.expand_dims(A,1),[1,tf.shape(B)[0],1])
    sdif = tf.squared_difference(Amod,Bmod)
    if agg=='min':
        rtn = tf.reduce_min(tf.sqrt(tf.reduce_sum(sdif,axis=2)),axis=1)
    if agg=='mean':
        rtn = tf.reduce_mean(tf.sqrt(tf.reduce_sum(sdif,axis=2)),axis=1)
    return rtn

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
        shift = 2.5
        r = 2.0 * np.pi / float(num_leaves) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batchsize, ndim // 2))
    y = np.random.normal(0, y_var, (batchsize, ndim // 2))
    z = np.empty((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], random.randint(0, 10 - 1), num_leaves)
    return z

def density(A,ep):
    # for each row in A, count how many data points exists in ep distance
    # return average density
    Bmod = tf.tile(tf.expand_dims(A,0),[tf.shape(A)[0],1,1])
    Amod = tf.tile(tf.expand_dims(A,1),[1,tf.shape(A)[0],1])
    sdif = tf.squared_difference(Amod,Bmod)
    dist = tf.sqrt(tf.reduce_sum(sdif,axis=2))
    dens = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.less(dist,ep),tf.float32),axis=1))
    return dens

"""
Defining key operations, Loess, Optimizer and other necessary operations
"""
with tf.variable_scope('Encoder'):
    encoder_output = encoder(x_input)
    
with tf.variable_scope('Decoder'):
    decoder_output = decoder(encoder_output)
    
# loss
#reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

with tf.name_scope("ae_loss"):
    autoencoder_loss = tf.reduce_mean(tf.square(x_input - decoder_output))

#optimizer
all_variables = tf.trainable_variables()

dc_var = [var for var in all_variables if 'Discriminator/' in var.name]
ae_var = [var for var in all_variables if ('Encoder/' in var.name or 'Decoder/' in var.name)]

with tf.name_scope("AE_optimizer"):
    autoencoder_optimizer = tf.train.AdamOptimizer().minimize(autoencoder_loss)

init = tf.global_variables_initializer()

# Reshape immages to display them
input_images = tf.reshape(x_input, [-1, 28, 28, 1])
generated_images = tf.reshape(decoder_output, [-1, 28, 28, 1])

# Tensorboard visualization
ae_sm = tf.summary.scalar(name='Autoencoder_Loss', tensor=autoencoder_loss)

summary_op = tf.summary.merge_all()

# Creating saver and get ready
saver = tf.train.Saver()
step = 0

"""
Executing with a session based on mode specification
"""

with tf.Session() as sess:
    sess.run(init)
    A = np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]])
    bx = sess.run([density(A,1.1)])
    print(bx)
    print(np.array(bx).shape)

    
        