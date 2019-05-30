import PYH_6
import AutoCov
import BPF
import TBF
import AAC
import ASDC
import fea_exp
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.svm import SVC
import scipy.io as sio

pd.set_option('display.width',1000)

# load physicochemical properties
# filename = 'phy_6.mat'
# filename = 'energy_20.mat'
filename = 'all_whsx_list.mat'
# filename = 'pssm.mat'
matrix = sio.loadmat(filename)
# sio.savemat('filename',{'data':dataa})
# matrix = matrix['phy_6']
matrix = matrix['all_whsx']
# matrix = matrix['energy_20']
# matrix = matrix['pssm']
print(matrix.shape)
print(matrix)
# ================================================================================
filename = r'F:\testtesttest\Anti-inflammatory\processed\pos_validation.txt'
proteins = []
with open(filename,'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        line = line.split()
        proteins.append(line)
dirbname = r'F:\testtesttest\Anti-inflammatory_repeate\temp_result\pssm\validation\pos'
for i in range(len(proteins)):
    s = proteins[i][0]
    protein = []
    for p in range(len(s)):
        protein.append(s[p])
    res = fea_exp.fifty_seven(matrix,protein)
    df = pd.DataFrame(res)
    data = df.iloc[:,:].values
    data = data.reshape(-1,57)
    # print(data)
    # save to .mat
    # sio.savemat(dirbname+"\\"+str(i)+".mat",{'data'+str(i)+'':data})
    # sio.savemat(dirbname + "\\" + str(i) + ".mat", {'data': data})
    # ==================================================================================

    # savetxt to .txt
    # fh = open(dirbname + "\\" + str(i) + '.txt', 'w')
    # for j in range(len(res)):
    #     l = ""
    #     for t in range(len(res[j])):
    #         l = l + str(res[j][t]) + " "
    #     fh.write(l)
    #     fh.write('\n')
    # fh.close()


# load sample
# sample = pd.read_csv('sample.csv')
# proteins = sample.iloc[:,1].values

# AC
# for i in range(len(proteins)):
#     protein = []
#     for j in range(2,len(proteins[i])-2):
#         protein.append(proteins[i][j])
#     print('\n',protein)
#     p = PYH_6.transform(protein,matrix)
#     ac = AutoCov.autocov(p,len(protein))
#     p = p.ravel()
#     print('p\n', p)
#     print('ac\n', ac)

# BPF
# for i in range(len(proteins)):
#     protein = []
#     for j in range(2,len(proteins[i])-2):
#         protein.append(proteins[i][j])
#     print('\n',protein)
#     bpf = BPF.bpf(protein,len(protein))
#     print('BPF\n', bpf)

# TBF
# for i in range(len(proteins)):
#     protein = []
#     for j in range(2,len(proteins[i])-2):
#         protein.append(proteins[i][j])
#     print('\n',protein)
#     tbf = TBF.tbf(protein,len(protein))
#     print('TBF\n', tbf)

#AAC
# protein = ['A','A','C','D','B']
# p = AAC.aac(protein)
# print(p)

# ASDC
# for i in range(len(proteins)):
#     protein = []
#     for j in range(2,len(proteins[i])-2):
#         protein.append(proteins[i][j])
#     print('\n',protein)
#     asdc = ASDC.asdc(protein)
#     print('ASDC\n', asdc.reshape(-1,20))
# pca = PCA(n_components= 2 )
# pca.fit()
# data = pca.transform()