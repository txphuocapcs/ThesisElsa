import numpy
import librosa as rosa

wave,sr= rosa.load('glasstest.wav', mono= True, sr=16000)
numpy.save('test.npy',wave, allow_pickle= False)