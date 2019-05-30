import numpy as np
import scipy.io as sio
from ECC.fea_extract import protein_exp
from ECC.fea_sel import extra_PSE,extra_DWT

def read_file(dirname):
    proteins = []
    with open(dirname,'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.split()[0]
            proteins.append(line)
    return proteins

def load_model(filename):
    import pickle
    with open(filename,'rb') as f:
        model = pickle.load(f)
    return model

def encoder(dictMats,dictLabels,ccModel,ccIndex):
    labelPres = np.array([])
    for i in range(len(dictMats)):
        for j in range(len(dictMats)):
            clf = ccModel[i][j]
            index = ccIndex[i][j]
            if j == 0:
                labelPre = clf.predict(dictMats[int(index)])
                labelPres = labelPre[:,np.newaxis]
            else:
                labelPre = clf.predict(np.hstack([dictMats[int(index)],labelPres]))
                labelPres = np.hstack([labelPres,labelPre[:,np.newaxis]])
        labelPre = labelPre.reshape(-1,1)
        if i == 0:
            pEncoder = labelPre
        else:
            pEncoder = np.hstack([pEncoder,labelPre])
    # 返回链数\链数乘分类器数
    return pEncoder

def caucalate_metrics(y_true,y_pred):
    TP = np.sum(y_true[y_pred==1]==1)
    TN = np.sum(y_true[y_pred==0]==0)
    FN = np.sum(y_true[y_pred==0]==1)
    FP = np.sum(y_true[y_pred==1]==0)
    se = round(TP / (TP + FN),4)
    sp = round(TN / (FP + TN),4)
    acc = round((TP + TN) / len(y_true),4)
    mcc = round(((TP * TN) - (FP * FN)) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)),4)
    a = [se,sp,acc,mcc]
    return a

def mutilTest(dictMats, dictLabels):
    filename = r'E:\tensorLearn\ECC\train_model\CCmodel.pickle'
    indexfile = r'E:\tensorLearn\ECC\train_model\index.txt'
    modelfile = r'E:\tensorLearn\ECC\train_model\clf.pickle'
    ccModel = load_model(filename)
    ccIndex = load_model(indexfile)
    clf = load_model(modelfile)
    X_test = encoder(dictMats, dictLabels, ccModel, ccIndex)
    print(X_test.shape)
    y = np.hstack([np.ones((1, 173)), np.zeros((1, 252))])[0]
    y_pred = clf.predict(X_test)
    print("accuracy_score: ", caucalate_metrics(y,y_pred))

if __name__ == '__main__':
    dirname = r'./pos_validation.txt'
    proteins = read_file(dirname)
    dirname = r'./neg_validation.txt'
    proteins_neg = read_file(dirname)
    proteins.extend(proteins_neg)
    print(len(proteins))
    dictMats = {}
    dictLabels = {}
    for i in range(len(proteins)):
        print("------------------------sample "+str(i))
        fets = protein_exp.fea_exp(proteins[i])
        DWT_feas = extra_DWT.DWT_fea(fets)
        PSE_feas = extra_PSE.PSE_fea(fets)

        if i == 0:
            # dictMats[0] = DWT_feas[0].reshape(1, -1)
            # dictMats[1] = DWT_feas[1].reshape(1, -1)
            # dictMats[2] = DWT_feas[2].reshape(1, -1)
            dictMats[0] = PSE_feas[0].reshape(1, -1)
            dictMats[1] = PSE_feas[1].reshape(1, -1)
            dictMats[2] = PSE_feas[2].reshape(1, -1)
        else:
            # dictMats[0] = np.vstack([dictMats[0], DWT_feas[0].reshape(1, -1)])
            # dictMats[1] = np.vstack([dictMats[1], DWT_feas[1].reshape(1, -1)])
            # dictMats[2] = np.vstack([dictMats[2], DWT_feas[2].reshape(1, -1)])
            dictMats[0] = np.vstack([dictMats[0], PSE_feas[0].reshape(1, -1)])
            dictMats[1] = np.vstack([dictMats[1], PSE_feas[1].reshape(1, -1)])
            dictMats[2] = np.vstack([dictMats[2], PSE_feas[2].reshape(1, -1)])

    for i in range(3):
        lablePos = np.ones((173, 1))
        labelNeg = np.zeros((252, 1))
        label = np.vstack([lablePos, labelNeg])
        dictLabels[i] = label

    mutilTest(dictMats,dictLabels)