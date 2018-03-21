from __future__ import print_function
import numpy as np

from keras.optimizers import SGD
np.random.seed(1337)
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Reshape, Convolution1D, MaxPooling1D, AtrousConv1D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import LeakyReLU
import utils
import data
import tensorflow as tf
from keras import backend as K

'exception_verbosity = high'
batch_size = 32
hidden_units = 100
nb_classes = data.voca_size


def ctc_loss(y_true,y_pred):
    y_true_sparse=K.ctc_label_dense_to_sparse(y_true, 64)
    y_pred_sparse=K.ctc_label_dense_to_sparse(y_pred, 64)
    return(tf.nn.ctc_loss(y_pred_sparse, 64,
         preprocess_collapse_repeated=False, ctc_merge_repeated=False,
         time_major=True))


print('Loading data...')
#X_train, y_train = utils.load_spectro()

print('Build model...')
model = Sequential()

#cnn block 1
#model.add(TimeDistributed(Convolution1D(128, 7, activation='relu',dilation_rate=1, padding='same'), input_shape=X_train.shape[1:]))
#model.add(TimeDistributed(Convolution1D(128, 7, activation='relu',dilation_rate=2, padding='same')))
#model.add(TimeDistributed(Convolution1D(128, 7, activation='relu',dilation_rate=4, padding='same')))
#model.add(TimeDistributed(Convolution1D(128, 7, activation='relu',dilation_rate=8, padding='same')))




#lstm block 1
#model.add(LSTM(output_dim=128, init='uniform', inner_init='uniform',
               #forget_bias_init='one', activation='relu', inner_activation='sigmoid', return_sequences=True, input_shape=X_train.shape[1:]))
#model.add(Dropout(0.2))

#lstm block 2
#model.add(LSTM(output_dim=64, init='uniform', inner_init='uniform',
               #forget_bias_init='one', activation='relu', inner_activation='sigmoid', return_sequences=True))
#model.add(Dropout(0.2))

#lstm block 3
#model.add(LSTM(output_dim=32, init='uniform', inner_init='uniform',
               #forget_bias_init='one', activation='relu', inner_activation='sigmoid'))
#model.add(Dropout(0.2))

#gru unit
model.add(GRU(128, return_sequences=True, activation='tanh', input_shape=(64,64)))
model.add(Dropout(0.2))
model.add(GRU(128, return_sequences=False, activation='tanh'))
model.add(Dropout(0.2))



model.add(Dense(units=nb_classes, activation='softmax'))
model.compile(loss=ctc_loss, optimizer='adam', metrics=['accuracy'])