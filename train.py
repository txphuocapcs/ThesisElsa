import tensorflow as tf
import numpy as np
import time
import data
num_features=20
num_hidden=512
num_layers=3
num_classes = data.voca_size+12

batch_size = 15*40
learning_rate = 0.01
num_epochs=50
num_examples = 16
#num_batches_per_epoch = int(num_examples / batch_size)
num_batches_per_epoch=230


graph= tf.Graph()
with graph.as_default():
    #mfcc is used as features
    inputs= tf.placeholder(tf.float32, [None, None, num_features], name='inpu')
    #inputs_reshaped= tf.reshape(inputs, [1,-1,num_features, 1])
    #Sparpse_placeholder
    targets= tf.sparse_placeholder(tf.int32, name='targ')

    #seq_len= [batch_size]
    seq_len= tf.placeholder(tf.int32, [None], name='seqlen')

    #network definition

    #convolution block 1
    #conv1= tf.layers.conv2d(inputs_reshaped, 120, kernel_size=(3,3), activation=tf.nn.relu)
    #pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # convolution block 2
    #conv2 = tf.layers.conv2d(pool1, 120, kernel_size=[3, 3], activation=tf.nn.relu)
    #pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=3)


    #pool2= tf.reshape(pool2, [1,-1, num_features])
    #rnn layers
    cell1 = tf.contrib.rnn.GRUCell(num_hidden, activation=tf.tanh)
    cell2 = tf.contrib.rnn.GRUCell(num_hidden, activation=tf.tanh)
    cell3 = tf.contrib.rnn.GRUCell(num_hidden, activation=tf.tanh)
    cell4 = tf.contrib.rnn.GRUCell(num_hidden, activation=tf.tanh)
    #stack rnn
    stack= tf.contrib.rnn.MultiRNNCell([cell1]+[cell2]+[cell3]+ [cell4], state_is_tuple=True)

    #second output= last state, omitted
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

    shape=tf.shape(inputs)

    batch_s, max_time_steps= shape[0], shape[1]

    #reshaping to apply the same weights over the timesteps
    outputs= tf.reshape(outputs, [-1, num_hidden])

    #zero init
    W=tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))

    b= tf.Variable(tf.constant(0., shape=[num_classes]))
    #Output*W + b, softmax
    logits= tf.matmul(outputs, W)+b

    #reshape to original
    logits= tf.reshape(logits, [batch_s, -1, num_classes])

    #tranpose to time major
    logits= tf.transpose(logits, (1,0,2), name='final')

    #ctc loss definition
    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)
    optimizer= tf.train.AdamOptimizer().minimize(cost)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)

    with tf.Session(graph=graph) as session:
        saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        #re-init dataset
        for curr_epoch in range(num_epochs):
            print('Epoch '+ str(curr_epoch)+':')
            train_cost=train_ler=0
            start=time.time()
            dataset = data.Dataset(1)
            epoch_cost=0
            num_batches_per_epoch= dataset.total_samples()
            for batch in range(num_batches_per_epoch):
                train_inputs, train_targets, train_seq_len= dataset.feed()
                #tmp=list()
                #for target in train_targets:
                sparserow= data.sparse_tuple_from([train_targets])
                    #if (tmp==None):
                        #tmp=sparserow
                    #else:
                #tmp.append(sparserow)
                feed = {inputs: train_inputs, targets: data.sparse_tuple_from([train_targets]), seq_len: train_seq_len}
                costval, __ =session.run([cost, optimizer], feed_dict=feed)
                epoch_cost+=costval/num_batches_per_epoch
                print(str(batch*100/ num_batches_per_epoch)+'%' +'    Cost:' + str(epoch_cost)+ '              Total time: '+ str(time.time()-start))
            print ('Congrats! Nya!')
            #if (curr_epoch%2==0):
            saver.save(session, 'training/model.'+str(epoch_cost)[:5]+'.ckpt')


        #predict
        mfcc= np.load('p225_001.wav.npy').transpose()
        mfcc= mfcc.reshape(1,mfcc.shape[0], mfcc.shape[1])
        out = decoded
        seqlen=[mfcc.shape[1]]
        d= session.run(decoded[0], feed_dict={inputs:mfcc, seq_len:seqlen})
        str_decoded = ''.join([data.index2byte[x] for x in np.asarray(d[1])])
        print(str_decoded)
        tmp=0

