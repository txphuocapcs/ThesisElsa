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
num_epochs=1
num_examples = 16
#num_batches_per_epoch = int(num_examples / batch_size)
num_batches_per_epoch=230

tf.reset_default_graph()
graph= tf.Graph()

with graph.as_default():
    with tf.Session(graph=graph) as session:
        #init graph
        new_saver = tf.train.import_meta_graph('training/model.121.9.ckpt.meta')
        new_saver.restore(session, 'training/model.121.9.ckpt')

        #reload tensors


        inputs = graph.get_tensor_by_name('inpu:0')

        targetsind= graph.get_tensor_by_name('targ/indices:0')
        targetsval= graph.get_tensor_by_name('targ/values:0')
        targetsshape = graph.get_tensor_by_name('targ/shape:0')
        targets = tf.SparseTensor(indices= targetsind, values= targetsval, dense_shape= targetsshape)
        seq_len = graph.get_tensor_by_name('seqlen:0')
        logits = graph.get_tensor_by_name('final:0')


        #ctc loss definition
        loss = tf.nn.ctc_loss(targets, logits, seq_len)
        cost = tf.reduce_mean(loss)
        optimizer= graph.get_operation_by_name('optimizer')
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)



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
                feed = {inputs: train_inputs,
                        targetsind: sparserow[0],
                        targetsval: sparserow[1],
                        targetsshape: sparserow[2],
                        seq_len: train_seq_len}
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

