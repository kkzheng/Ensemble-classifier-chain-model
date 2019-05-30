import fea_exp
import pandas as pd
import scipy.io as sio

# load ".mat" file
# filename = 'energy_20.mat'
filename = 'all_whsx_list.mat'
# filename = 'pssm.mat'
matrix = sio.loadmat(filename)
# sio.savemat('filename',{'data':dataa})
matrix = matrix['all_whsx']
# matrix = matrix['energy_20']
# matrix = matrix['pssm']

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
    print(data)
    save to .mat
    sio.savemat(dirbname+"\\"+str(i)+".mat",{'data'+str(i)+'':data})
    sio.savemat(dirbname + "\\" + str(i) + ".mat", {'data': data})
