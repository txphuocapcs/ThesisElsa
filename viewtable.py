import numpy as np
total_pron=np.load('total.np.npy', allow_pickle=False)
false_pron= np.load('false.np.npy', allow_pickle=False)
print(np.sum(false_pron)/np.sum(total_pron))
for i in range(1,41):
    for j in range(1,41):
        false_pron[i][j]=false_pron[i][j]/total_pron[i]
tmp=0
np.savetxt("foo.csv", false_pron, delimiter=",")
