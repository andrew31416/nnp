"""
networks for per-atom attributes
"""

import tensorflow as tf
import numpy as np


def train(train_x,train_y,test_x,test_y,train_lim,test_lim,num_nodes,output_ypred,opt_method,\
        activation='sigmoid',reg_const=0.001):
    """
    Input
    -----
    """

    # feature dimensionality
    D = train_x.shape[1]

    # number of hidden layers in network
    num_hidden_layers = len(num_nodes)

    #---------------#
    # place holders #
    #---------------#

    # dimension-D input
    x = tf.placeholder(tf.float64, [None, D])

    # total configuration energy
    Etot_in = tf.placeholder(tf.float64,shape=[None,1])

    # borders between structures in 1-d array of atoms
    slice_lim = tf.placeholder(tf.int32,[None,2])

    # number of structures
    Nconf = tf.placeholder(dtype=tf.int32,shape=())

    hidden_weights = []
    hidden_biases = []
    hidden_output = []

    #----------------------#
    # initial hidden layer #
    #----------------------#

    hidden_weights = [tf.Variable(tf.random_normal([D,num_nodes[0]],dtype=tf.float64))]
    hidden_biases = [tf.Variable(tf.zeros([num_nodes[0]],dtype=tf.float64))]
    if activation == 'sigmoid':
        hidden_output = [tf.nn.sigmoid(tf.matmul(x, hidden_weights[0]) + hidden_biases[0])]
    elif activation == 'tanh':
        hidden_output = [tf.nn.tanh(tf.matmul(x, hidden_weights[0]) + hidden_biases[0])]
    else:
        raise NotImplementedError

    #-------------------------#
    # remaining hidden layers #
    #-------------------------#

    for ii,_width in enumerate(num_nodes):
        if ii==0:
            continue
        hidden_weights.append( tf.Variable(tf.random_normal([num_nodes[ii-1],_width],dtype=tf.float64)) )
        hidden_biases.append( tf.Variable(tf.zeros([_width],dtype=tf.float64)) )
        
        if activation == 'sigmoid':
            hidden_output.append( tf.nn.sigmoid(tf.matmul(hidden_output[ii-1], hidden_weights[ii]) + \
                    hidden_biases[ii]) )
        elif activation == 'tanh':
            hidden_output.append( tf.nn.tanh(tf.matmul(hidden_output[ii-1], hidden_weights[ii]) + \
                    hidden_biases[ii]) )

    #--------------#
    # output layer #
    #--------------#

    # final output layer weights
    final_weights = tf.Variable(tf.random_normal([num_nodes[-1], 1],dtype=tf.float64))

    # final output layer offsets
    final_bias = tf.Variable(tf.zeros([1],dtype=tf.float64))

    # energy per atom [Nconf,Natm]
    output_y = tf.matmul(hidden_output[-1], final_weights) + final_bias

    # total configuration energy
    #Etot_out = tf.Variable(tf.zeros([Nconf,1],dtype=tf.float64))
    Etot_out = tf.zeros([Nconf,1],dtype=tf.float64)

    for ii in range(Nconf):
        tf.scatter_add(ref=Etot_out,indices=[ii],\
                updates=tf.reduce_sum( tf.slice(output_y,begin=slice_lim[ii,0],size=slice_lim[ii,1]) ) )

    #-----------------#
    # fitness measure #
    #-----------------#

    regularizer = tf.nn.l2_loss(final_weights) + tf.nn.l2_loss(final_bias) 
    for ii in range(num_hidden_layers):
        regularizer += tf.nn.l2_loss(hidden_weights[ii]) + tf.nn.l2_loss(hidden_biases[ii])

    # convergence measure
    loss_train = tf.sqrt(tf.reduce_mean(tf.squared_difference(Etot_out, Etot_in))) + \
            reg_const*tf.reduce_mean(regularizer)
   
    # rmse
    loss_test = tf.sqrt(tf.reduce_mean(tf.squared_difference(Etot_out, Etot_in))) 

    with tf.Session() as sess:
        # do initialisation
        sess.run(tf.global_variables_initializer())

        if opt_method == "batch":
            # optimiser of convergence measure
            train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss_train)
            
            store_loss = {"x":[],"y":[]}

            for ii,_batch_size in enumerate(np.hstack((\
                    np.linspace(int(0.1*train_x.shape[0]),train_x.shape[0],1500,dtype=int),\
                    np.ones(1000)*train_x.shape[0]))):
                # pick a subset for training each epoch
                idx = np.random.choice(train_x.shape[0],_batch_size,replace=False)
                batch_x = train_x[idx]
                batch_y = train_y[idx]
                
                # could take a random sub sample at each pass here
                _,calculated_loss = sess.run([train_step,loss_train],feed_dict={x:np.array(batch_x),\
                        Etot_in:np.array(batch_y)})
                if np.mod(ii,20)==0:
                    print('ii = {} loss = {}'.format(ii,calculated_loss))
                    store_loss["y"].append(calculated_loss)
                    store_loss["x"].append(ii)
        else:
            # local optimisation
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss_train,method=opt_method)
            optimizer.minimize(session=sess,feed_dict={x:np.array(train_x),Etot_in:np.array(train_y),\
                    slice_lim:train_lim,Nconf:slice_lim.shape[0]})


        #-------------------------------#
        # compute measure for whole set #
        #-------------------------------#
        
        # try to evaluate model on test set
        test_measure = sess.run(loss_test,feed_dict={x:np.array(test_x),Etot_in:np.array(test_y),\
                slice_lim:test_lim})
        
        # train measure
        train_measure = sess.run(loss_test,feed_dict={x:np.array(train_x),Etot_in:np.array(train_y),\
                slice_lim:train_lim})

        #--------------------#
        # output predicted y #
        #--------------------#

        if output_ypred:
            ypred_train = sess.run(output_y,feed_dict={x:np.array(train_x),Etot_in:np.array(train_y),\
                    slice_lim:train_lim})
            ypred_test = sess.run(output_y,feed_dict={x:np.array(test_x),Etot_in:np.array(test_y),\
                    slice_lim:test_lim})
        else:
            ypred_train = None
            ypred_test = None
    
    return train_measure,test_measure,ypred_train,ypred_test
