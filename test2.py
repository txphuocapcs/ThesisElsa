import tensorflow as tf
import librosa as rosa
import datapron as data
import numpy as np
import os
import subprocess
import pronunciation as pron
import pandas as pd
import glob
import csv
import align
from Bio.pairwise2 import format_alignment
_data_path = "data/"

false_pron=np.zeros([41,41])
total_pron= np.zeros(41)
with tf.Graph().as_default():
    with tf.Session() as session:

        new_saver = tf.train.import_meta_graph('training/model.0.542.ckpt.meta')
        new_saver.restore(session, 'training/model.0.542.ckpt')
        graph = tf.get_default_graph()
        inputs = graph.get_tensor_by_name('inpu:0')
        seq_len = graph.get_tensor_by_name('seqlen:0')
        logits = graph.get_tensor_by_name('final:0')
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)
        # predict
        #mfcc = np.load('p225_001.wav.npy').transpose()
        df = pd.read_table(_data_path + 'VCTK-Corpus/speaker-info.txt', usecols=['ID'],
                           index_col=False, delim_whitespace=True)

        # read file IDs
        file_ids = []
        for d in [_data_path + 'VCTK-Corpus/txt/p%d/' % uid for uid in df.ID.values]:
            if (int(d[-4:-1]) <=335):
                continue
            file_ids.extend([f[-12:-4] for f in sorted(glob.glob(d + '*.txt'))])

        for i, f in enumerate(file_ids):

            # wave file name
            wave_file = _data_path + 'VCTK-Corpus/wav48/%s/' % f[:4] + f + '.wav'
            fn = wave_file.split('/')[-1]
            target_filename = 'asset/data/preprocess/mfcc/' + fn + '.npy'
            # if os.path.exists( target_filename ):
            # continue
            # print info
            print("VCTK corpus preprocessing (%d / %d) - '%s']" % (i, len(file_ids), wave_file))

            # load wave file
            wave, sr = rosa.load(wave_file, mono=True, sr=16000)
            mfcc = rosa.feature.mfcc(wave, sr=16000).transpose()
            mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1])
            seqlen = [mfcc.shape[1]]

            pronlabel = pron.w2pro(open(_data_path + 'VCTK-Corpus/txt/%s/' % f[:4] + f + '.txt').read().split())
            print(pronlabel)
            if (pronlabel.size == 0):
                continue
            # get label index
            label = data.str2index(pronlabel)
            print(label)

            # save result ( exclude small mfcc data to prevent ctc loss )


            d = session.run(decoded[0], feed_dict={inputs: mfcc, seq_len:seqlen})
            for x in np.asarray(d[1]):
                tmp= data.index2byte[x]
                i=1
            str_decoded = ''.join([data.index2byte[x]+' ' for x in np.asarray(d[1])])
            print(str_decoded)
            a=align.algn(pronlabel, str_decoded)
            if (len(a)==0):
                continue
            print(format_alignment(*a[0]))
            for i in range(len(a[0][0])):
                tmp1= a[0][0][i]
                tmp2= a[0][1][i]
                if (a[0][1][i]!='<EMP>'):
                    if (a[0][0][i]=='-'):
                        total_pron[40]=total_pron[40]+1
                        if (a[0][1][i]!='-'):
                            false_pron[40][data.byte2index[a[0][1][i]]]=false_pron[40][data.byte2index[a[0][1][i]]]+1
                    else:
                        total_pron[data.byte2index[a[0][0][i]]]=total_pron[data.byte2index[a[0][0][i]]]+1
                        if (a[0][1][i]=='-'):
                            false_pron[data.byte2index[a[0][0][i]]][40]= false_pron[data.byte2index[a[0][0][i]]][40]+1
                        elif (a[0][0][i]!=a[0][1][i]):
                            false_pron[data.byte2index[a[0][0][i]]][data.byte2index[a[0][1][i]]]=false_pron[data.byte2index[a[0][0][i]]][data.byte2index[a[0][1][i]]]+1
                tmp=0
            np.save('total.np' ,total_pron,allow_pickle=False)
            np.save('false.np', false_pron, allow_pickle=False)
