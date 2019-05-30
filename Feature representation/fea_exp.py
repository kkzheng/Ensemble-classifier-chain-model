import numpy as np
def fea_exp(matrix,protein):
    # alfabeto = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    alfabeto = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    res = np.zeros((57,len(protein)))
    le = len(protein)
    for i in range(len(protein)):
        index = [j for j,x in enumerate(alfabeto) if x == protein[i]]
        if index == []:
            res[:,i] = 0
        else:
            res[:,i] = matrix[:,index[0]]
    return res
