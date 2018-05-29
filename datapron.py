import numpy as np
import csv
import string
import librosa as rosa

_coeff=20

# default data path
_data_path = 'asset/data/'

#
# vocabulary table
#

# index to byte mapping
index2byte = ['<EMP>', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY',
            'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G','HH',
            'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY',
            'P',  'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']

# byte to index mapping
byte2index = {}
for i, ch in enumerate(index2byte):
    byte2index[ch] = i

# vocabulary size
voca_size = len(index2byte)


# convert sentence to index list
def str2index(str_):

    # clean white space
    #str_ = ' '.join(str_.split())
    # remove punctuation and make lower case
    #str_ = str_.translate(string.punctuation).lower()

    res = np.array(0)
    for ch in str_:
        try:
            ch = ''.join(i for i in ch if not i.isdigit())
            res=np.append(res, byte2index[ch])
        except KeyError:
            # drop OOV
            pass
    return res


# convert index list to string
def index2str(index_list):
    # transform label index to character
    str_ = ''
    for ch in index_list:
        if ch > 0:
            str_ += index2byte[ch]
        elif ch == 0:  # <EOS>
            break
    return str_


# print list of index list
def print_index(indices):
    for index_list in indices:
        print(index2str(index_list))


#load mfcc
def _load_mfcc(src_list):

    # label, wave_file
    label, mfcc_file = src_list

    # decode string to integer
    label = np.fromstring(label, np.int)

    # load mfcc
    mfcc = np.load(mfcc_file, allow_pickle=False)

    # speed perturbation augmenting
    mfcc = _augment_speech(mfcc)

    return label, mfcc


def _augment_speech(mfcc):

    # random frequency shift ( == speed perturbation effect on MFCC )
    r = np.random.randint(-2, 2)

    # shifting mfcc
    mfcc = np.roll(mfcc, r, axis=0)

    # zero padding
    if r > 0:
        mfcc[:r, :] = 0
    elif r < 0:
        mfcc[r:, :] = 0

    return mfcc

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

class Dataset(object):

    def __init__(self, batch_size=1, set_name='train'):

        # load meta file
        self.batch_size=batch_size
        self.label, self.mfcc_file = [], []
        sum_row=0
        self.feed_count=0
        with open(_data_path + 'preprocess/meta/%s.csv' % set_name) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                # mfcc file
                if (len(row)==0):
                    continue
                self.mfcc_file.append(_data_path + 'preprocess/mfcc/' + row[0] + '.npy')
                # label info ( convert to string object for variable-length support )
                self.label.append(np.asarray(row[1:], dtype=np.int))
                sum_row+=1
        #generate queue of files
        self.queue= np.arange(sum_row)
        #shuffle input queue
        #np.random.shuffle(self.queue)
    def shuffle(self):
        self.queue = np.arange(self.queue.size)
        np.random.shuffle(self.queue)
        self.feed_count=0
    def total_samples(self):
        return self.queue.size
    def feed(self):
        #train_X = np.zeros((self.batch_size,1, 15*40, _coeff))
        #train_Y = np.zeros((self.batch_size, 256))
        batch_f_count=0
        #load files into batch
        mfcc=np.load(self.mfcc_file[self.feed_count]).transpose().copy()
        #mfcc.resize(240*20)
        #mfcc=mfcc.reshape([240,20])
        #mfcc=mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1])
        label= self.label[self.feed_count]
        #label_one_hot=np.zeros(256)
        #label_one_hot[:label.shape[0]]=label
        #train_X[batch_f_count][0][:mfcc.shape[0]]=mfcc
        #train_Y[batch_f_count][:label_one_hot.shape[0]]=label_one_hot
        self.feed_count+=1
        return mfcc, label, [mfcc.shape[0]]
        # print info

    def feedBatch(self):
        bsCount = 0
        ret_mfcc = None
        ret_label = None
        ret_seq_len = None
        init=False
        while (bsCount < self.batch_size and self.feed_count < self.queue.size):
            mfcc, label, seq_len = self.feed()
            if (init == False):
                ret_mfcc = np.array([mfcc.copy()])
                ret_label = [label]
                ret_seq_len = seq_len
                init=True
            else:
                if (mfcc.shape[0] > ret_mfcc.shape[1]):
                    tmpret = np.zeros((ret_mfcc.shape[0], mfcc.shape[0] - ret_mfcc.shape[1], 20))
                    ret_mfcc = np.concatenate((ret_mfcc, tmpret), axis=1)
                elif (mfcc.shape[0] < ret_mfcc.shape[1]):
                    tmpret = np.zeros((-mfcc.shape[0] + ret_mfcc.shape[1], 20))
                    mfcc= np.concatenate((mfcc, tmpret), axis=0)
                ret_mfcc= np.concatenate((ret_mfcc,[mfcc]), axis=0)
                ret_label.append(label)
                ret_seq_len = np.append(ret_seq_len, seq_len)
            bsCount += 1
        return ret_mfcc, ret_label, ret_seq_len