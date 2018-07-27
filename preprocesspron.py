import numpy as np
import pandas as pd
import glob
import csv
import librosa
import datapron as data
import os
import subprocess
import pronunciation as pron
import re

__author__ = 'namju.kim@kakaobrain.com'


# data path
_data_path = "data/"


#
# process VCTK corpus
#

def process_vctk(csv_file):

    # create csv writer
    writer = csv.writer(csv_file, delimiter=',')

    # read label-info
    df = pd.read_table(_data_path + 'VCTK-Corpus/speaker-info.txt', usecols=['ID'],
                       index_col=False, delim_whitespace=True)

    # read file IDs
    file_ids = []
    for d in [_data_path + 'VCTK-Corpus/txt/p%d/' % uid for uid in df.ID.values]:
        if (int(d[-4:-1])>335):
            continue
        file_ids.extend([f[-12:-4] for f in sorted(glob.glob(d + '*.txt'))])

    for i, f in enumerate(file_ids):

        # wave file name
        wave_file = _data_path + 'VCTK-Corpus/wav48/%s/' % f[:4] + f + '.wav'
        fn = wave_file.split('/')[-1]
        target_filename = 'asset/data/preprocess/mfcc/' + fn + '.npy'
        #if os.path.exists( target_filename ):
            #continue
        # print info
        print("VCTK corpus preprocessing (%d / %d) - '%s']" % (i, len(file_ids), wave_file))

        # load wave file
        wave, sr = librosa.load(wave_file, mono=True, sr=16000)

        # re-sample ( 48K -> 16K )
        #wave = wave[::3]

        # get mfcc feature
        mfcc = librosa.feature.mfcc(wave, sr=16000)
        truthlbl= open(_data_path + 'VCTK-Corpus/txt/%s/' % f[:4] + f + '.txt').read()
        pronlabel= pron.w2pro(re.split(r'(\s+)', truthlbl))
        #print (pronlabel)
        if (pronlabel.size==0):
            continue
        # get label index
        label = data.str2index(pronlabel)
        #print (label)

        # save result ( exclude small mfcc data to prevent ctc loss )
        if len(label) < mfcc.shape[1]:
            # save meta info
            writer.writerow(np.append(fn, label))
            # save mfcc
            np.save(target_filename, mfcc, allow_pickle=False)


def process_libri(csv_file, category):
    parent_path = _data_path + 'train-clean-360/LibriSpeech/' + category + '/'
    labels, wave_files = [], []

    # create csv writer
    writer = csv.writer(csv_file, delimiter=',')

    # read directory list by speaker
    speaker_list = glob.glob(parent_path + '*')
    for spk in speaker_list:

        # read directory list by chapter
        chapter_list = glob.glob(spk + '/*/')
        for chap in chapter_list:

            # read label text file list
            txt_list = glob.glob(chap + '/*.txt')
            for txt in txt_list:
                with open(txt, 'rt') as f:
                    records = f.readlines()
                    for record in records:
                        # parsing record
                        field = record.split('-')  # split by '-'
                        speaker = field[0]
                        chapter = field[1]
                        field = field[2].split()  # split field[2] by ' '
                        utterance = field[0]  # first column is utterance id

                        # wave file name
                        wave_file = parent_path + '%s/%s/%s-%s-%s.flac' % \
                                    (speaker, chapter, speaker, chapter, utterance)
                        wave_files.append(wave_file)

                        # label index
                        truthlbl=np.array(field[1:])
                        truthlblws= ''
                        init=False
                        for word in truthlbl:
                            if (init==False):
                                truthlblws= np.array([word.copy()])
                                truthlblws = np.append(truthlblws, ' ')
                                init=True
                            else:
                                truthlblws= np.append(truthlblws, np.array(word))
                                truthlblws= np.append(truthlblws,' ')
                        pronlabel=pron.w2pro(truthlblws)
                        labels.append(data.str2index(pronlabel))  # last column is text label
                        tmp=0

    for i, (wave_file, label) in enumerate(zip(wave_files, labels)):
        fn = wave_file.split('/')[-1]
        target_filename = 'asset/data/preprocess/mfcc/' + fn + '.npy'
        if os.path.exists(target_filename):
            continue
        # print info
        print("LibriSpeech corpus preprocessing (%d / %d) - '%s']" % (i, len(wave_files), wave_file))

        # load flac file
        #wave, sr, _ = scikits.audiolab.flacread(wave_file)]
        wave, sr= librosa.load(wave_file, mono=True, sr=16000)

        # get mfcc feature
        mfcc = librosa.feature.mfcc(wave, sr=16000)

        #pronlabel = pron.w2pro(open(_data_path + 'VCTK-Corpus/txt/%s/' % f[:4] + f + '.txt').read().split())
        #print(pronlabel)
        #if (pronlabel.size == 0):
            #continue
        # get label index
        #label = data.str2index(pronlabel)
        #print(label)
        try:
            a= len(label)
        except:
            continue

        # save result ( exclude small mfcc data to prevent ctc loss )
        if len(label) < mfcc.shape[1]:
            # save meta info
            writer.writerow(np.append(fn, label))
            # save mfcc
            np.save(target_filename, mfcc, allow_pickle=False)


#
# Create directories
#
if not os.path.exists('asset/data/preprocess'):
    os.makedirs('asset/data/preprocess')
if not os.path.exists('asset/data/preprocess/meta'):
    os.makedirs('asset/data/preprocess/meta')
if not os.path.exists('asset/data/preprocess/mfcc'):
    os.makedirs('asset/data/preprocess/mfcc')


#
# Run pre-processing for training
#

# VCTK corpus
csv_f = open('asset/data/preprocess/meta/train.csv', 'w')
process_libri(csv_f, 'train-clean-360')
process_vctk(csv_f)


csv_f.close()