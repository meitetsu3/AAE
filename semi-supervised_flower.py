# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:50:32 2017
ref: https://github.com/Naresh1318/Adversarial_Autoencoder
@author: mtodaka
: ck operations on devices, meomry load inc
: stop when autoencoder loss increase on validation set
: drop out
: categorical descriminator
: separate? reconstruction, descriminator, generater, semi-supervised
: learning rate schedule
:feed mode to run ./file
"""

"""
modes:
1: Latent regulation. train generator to fool Descriminator with reconstruction constraint.
0: Showing latest model results. InOut, true dist, discriminator, latent dist.
"""
exptitle =  'lbl1000base' #experiment title that goes in tensorflow folder name
mode = 1
flg_graph = False # showing graphs or not during the training. Showing graphs significantly slows down the training.
model_folder = '' # name of the model to be restored. white space means most recent.
n_leaves = 10 # number of leaves in the mixed 2D Gaussian
n_epochs_ge = 20*n_leaves # mode 3, generator training epochs
ac_batch_size = 500  # autoencoder training batch size
semi_sup_batch_size = 32 # semi-supervised training batch size
import numpy as np
blanket_resolution = 10*int(np.sqrt(n_leaves)) # blanket resoliution for descriminator or its contour plot
dc_real_batch_size = int(blanket_resolution*blanket_resolution/15) # descriminator training real dist samplling batch size
n_label = 1000 # number of labels used in semi-supervised training
OoTWeight = 0.01 # out of target weight in generator
ClassificationWeight = 0.01 #classification weight in generator
Yreg_weight = 0.01 # Y regulation or distance to vertex weight
tb_batch_size = 800  # x_inputs batch size for tb
tb_log_step = 200  # tb logging step
import tensorflow as tf
config = tf.ConfigProto()
config.log_device_placement = True
config.allow_soft_placement=True
config.gpu_options.per_process_gpu_memory_fraction = 0.2 # can run up to 4 threads on main GPU, and 5 on others.

dc_contour_res_x = 5 # x to the blanket resolution for descriminator contour plot
myColor = ['black','orange', 'red', 'blue','gray','green','pink','cyan','lime','magenta']
input_dim = 784
xLU = [-10,10] # blanket x axis lower and upper
yLU = [-10,10] # blanket y axis lower and upper
n_l1 = 1000
n_l2 = 1000
z_dim = 2
results_path = './Results/Adversarial_Autoencoder'


from tensorflow.contrib.layers import fully_connected
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
labels_fixedcopy = mnist.train.labels # keeping original label

# Placeholders for input data and the targets

x_input = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='Input')
real_distribution = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='Real_distribution')
unif_z = tf.placeholder(dtype=tf.float32, shape=[blanket_resolution*blanket_resolution, z_dim], name='Uniform_z')
fake_lbl = tf.placeholder(dtype=tf.float32, shape=[None,10],name = 'fake_lbl')
trainer_lbl = tf.placeholder(dtype=tf.float32, shape=[None,10],name = 'trainer_lbl')
trainer_input = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='trainer_input')

he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

"""
Util Functions
"""

def variables_on_cpu(op):
    """
    Placing variables on CPU, partially to save GPU memory and utilize the main system RAM
    Not sure if this is actually impacting any. Need to check with TB.
    """
    if op.type not in ['Variable', 'VariableV2', 'VarHandleOp']:
        return '/gpu:0'
    else:
        return '/cpu:0'

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
    idx = random.sample(range(mnist[0].num_examples),ac_batch_size)
    img_in = mnist[0].images[idx,:]
    img_out = sess.run(op, feed_dict={x_input: img_in})
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
        train_zs,train_ys = sess.run(encoder(mnist[2].images, reuse=True)) #2 is test, 10k images
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
    print("Test Accuracy:{}%".format(test_accuracy))
    
def show_z_discriminator(sess):
    """
    Shows z discriminator activation contour plot. Close to 1 means estimated as positive (true dist).
    Parameters. seess:TF session.
    No return. Displays image.
    """
    if not flg_graph:
        return
    br = dc_contour_res_x*blanket_resolution
    xlist, ylist, blanket = get_blanket(br)

    plt.rc('figure', figsize=(6, 5))
    plt.tight_layout()
    
    X, Y = np.meshgrid(xlist, ylist)    
    
    with tf.variable_scope("Discriminator"):
        desc_result = sess.run(tf.nn.sigmoid(discriminator_z(blanket, reuse=True)))

    Z = np.empty((br,br), dtype="float32")    
    for i in range(br):
        for j in range(br):
            Z[i][j]=desc_result[i*br+j]

    fig, ax = plt.subplots(1)
    cp = ax.contourf(X, Y, Z)
    plt.colorbar(cp)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Z Descriminator Contour')    
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
    output = fully_connected(last_layer, input_dim, weights_initializer=he_init,scope='Sigmoid', activation_fn=tf.sigmoid)
    return output

def discriminator_z(x, reuse=False):
    """encoder
    Discriminator that leanes to activate at true distribution and not for the others.
    :param x: tensor of shape [batch_size, z_dim]
    :param reuse: True -> Reuse the discriminator variables, False -> Create the variables
    :return: tensor of shape [batch_size, 1]. I think it's better to take sigmoid here.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    last_layer = mlp_dec(x)
    output = fully_connected(last_layer, 1, weights_initializer=he_init, scope='None',activation_fn=None)
    return output

def gaussian_mixture(z, num_leaves, selector):
    """
    selector to distribute cluster face
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
    encoder_outputZ,encoder_outputYlogits = encoder(x_input)
    _,trainer_ylogits = encoder(trainer_input, reuse=True)
    
with tf.variable_scope('Decoder'):
    decoder_output = decoder(encoder_outputZ,tf.one_hot(tf.argmax(encoder_outputYlogits, dimension = 1), depth = n_leaves))
    
with tf.variable_scope('Discriminator'):
    d_real = discriminator_z(real_distribution)
    d_blanket = discriminator_z(unif_z, reuse=True)
    d_fake = discriminator_z(encoder_outputZ,reuse=True)
    
# loss 
with tf.name_scope('Y_regulation'):
    with tf.name_scope('unsupervised_simplex_distToVertex'):
        Yonehot = tf.one_hot(tf.argmax(encoder_outputYlogits, dimension = 1), depth = n_leaves)
        Ysoftmax = tf.nn.softmax(encoder_outputYlogits)
        dist_to_vertex = Yreg_weight*tf.reduce_mean(tf.reduce_sum(tf.square( Yonehot- encoder_outputYlogits),axis=1))

with tf.name_scope("dc_loss"):
    dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
    dc_loss_blanket = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_blanket), logits=d_blanket))
    dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
    dc_loss = dc_loss_blanket + dc_loss_real+dc_loss_fake

with tf.name_scope("ge_loss"):
    autoencoder_loss = tf.reduce_mean(tf.square(x_input - decoder_output))
    classification_loss = ClassificationWeight*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=trainer_lbl,logits=trainer_ylogits, name="classification_loss"))
    #Out of Target penaltyreal_lbl
    OoT_penalty =OoTWeight*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))
    generator_loss = autoencoder_loss+classification_loss+OoT_penalty+dist_to_vertex #not sure why it averages out

# metrics
with tf.name_scope('accuracy'):
    with tf.name_scope('train_accuracy'):
        t_correct_prediction = tf.equal(tf.argmax(trainer_lbl, 1), tf.argmax(trainer_ylogits, 1))
        t_accuracy = tf.reduce_mean(tf.cast(t_correct_prediction, tf.float32))
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(fake_lbl, 1), tf.argmax(encoder_outputYlogits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#optimizer
all_variables = tf.trainable_variables()
dc_var = [var for var in all_variables if 'Discriminator/' in var.name]
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
input_images = tf.reshape(x_input, [-1, 28, 28, 1])
generated_images = tf.reshape(decoder_output, [-1, 28, 28, 1])

# Tensorboard visualizationdegit_v
ae_sm = tf.summary.scalar(name='Autoencoder_Loss', tensor=autoencoder_loss)
dc_sm = tf.summary.scalar(name='Discriminator_Loss', tensor=dc_loss)
ge_sm = tf.summary.scalar(name='Generator_Loss', tensor=generator_loss)
oot_sm = tf.summary.scalar(name='OoT_penalty', tensor=OoT_penalty)
cll_sm = tf.summary.scalar(name='Classification_Loss', tensor=classification_loss)
acc_sm = tf.summary.scalar(name='Accuracy', tensor=accuracy)
tac_sm = tf.summary.scalar(name='Training_Accuracy', tensor=t_accuracy)
dtv_sm = tf.summary.scalar(name='Distance_to_Vertex', tensor=dist_to_vertex)
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
    #batch_x, batch_y = mnist.train.next_batch(tb_batch_size) #bigger batch to see ae_loss with stability
    test_x, test_y = mnist.test.next_batch(tb_batch_size) # to evalute accuracy with unseen data.
    # use the priviousely generated data for others
    sm = sess.run(summary_op,feed_dict={x_input: test_x, real_distribution:dc_real_z\
            ,unif_z:blanket, trainer_input:train_x,trainer_lbl:train_y,fake_lbl:test_y})
    writer.add_summary(sm, global_step=step)

with tf.Session(config=config) as sess:
    if mode==1: # Latent regulation
        writer,saved_model_path = tb_init(sess)   
        _,_,blanket = get_blanket(blanket_resolution)
        n_batches = int(mnist.train.num_examples / ac_batch_size)
        train_x, train_y = mnist.train.next_batch(n_label)
        test_x, test_y = mnist.test.next_batch(n_label)
        for i in range(n_epochs_ge):
            print("------------------Epoch {}/{} ------------------".format(i, n_epochs_ge))
            for b in tqdm(range(n_batches)):    
                #Discriminator
                batch_x, _ = mnist.train.next_batch(ac_batch_size)
                #random label as a selector to train z descriminator for flower graph 
                dc_real_lbl = np.eye(10)[np.array(np.random.randint(0,n_leaves, size=dc_real_batch_size)).reshape(-1)]
                dc_real_z = gaussian(dc_real_batch_size)

                sess.run([discriminator_optimizer],feed_dict={x_input: batch_x, \
                         real_distribution:dc_real_z,unif_z:blanket})
                #Generator - autoencoder, fooling descriminator, and y semi-supervised classification
                train_xb, train_yb = next_batch(train_x,train_y,semi_sup_batch_size)
                sess.run([generator_optimizer],feed_dict={x_input: batch_x,trainer_input:train_xb, trainer_lbl:train_yb})
                if b % tb_log_step == 0:
                    show_z_discriminator(sess) #shows others like 3, 7 -1 ?
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
        show_z_discriminator(sess)    
        show_latent_code(sess)
        
            
        
        