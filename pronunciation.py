import nltk
import numpy as np

dict = nltk.corpus.cmudict.dict()

def w2pro(words):
    init=False
    #print (words)
    for word in words:
        if (word!=' '):
            iword= word.lower()
            iword= iword.replace(".", "")
            iword = iword.replace(",", "")
            iword = iword.replace(";", "")
            iword = iword.replace("?", "")
            iword = iword.replace("!", "")
            iword = iword.replace("'", "")
            iword = iword.replace("\n", "")
        else:
            iword=' '
        pron=None
        if (len(iword)==0):
            continue
        if (iword==' '):
            pron=' '
        elif (iword in dict):
            pron= dict[iword][0]
        else:
            return np.array([])
        if (init==False):
            ret=pron.copy()
            init=True
        else:
            ret= np.append(ret, pron)
    if (ret[-1]!=' '):
        ret=np.append(ret,' ')
    return ret