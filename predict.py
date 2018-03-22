import tensorflow as tf
import librosa as rosa
import data
import numpy as np

with tf.Graph().as_default():
    with tf.Session() as session:
        new_saver = tf.train.import_meta_graph('training/model.13.74.ckpt.meta')
        new_saver.restore(session, 'training/model.13.74.ckpt')
        # predict
        #mfcc = np.load('p225_001.wav.npy').transpose()
        wave,sr=rosa.load('p225_002.wav', mono=True, sr=16000)
        rosa.output.write_wav('16k',wave,sr=16000)
        mfcc=rosa.feature.mfcc(wave).transpose()
        mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1])
        seqlen = [mfcc.shape[1]]
        graph=tf.get_default_graph()
        inputs = graph.get_tensor_by_name('inpu:0')
        seq_len= graph.get_tensor_by_name('seqlen:0')
        logits= graph.get_tensor_by_name('final:0')
        decoded, log_prob= tf.nn.ctc_beam_search_decoder(logits, seq_len)
        d = session.run(decoded[0], feed_dict={inputs: mfcc, seq_len:seqlen})
        str_decoded = ''.join([data.index2byte[x] for x in np.asarray(d[1])])
        print(str_decoded)
        tmp = 0