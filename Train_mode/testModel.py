import numpy as np
import scipy.io as sio

def load_data(dirname):
    dictMats = {}
    dictLabels = {}
    for i in range(6):

        matrix = sio.loadmat(dirname + '\\' + 'sample' + str(i) + '.mat')
        matrix = matrix['feature']
        lablePos = np.ones((173, 1))
        labelNeg = np.zeros((252, 1))
        label = np.vstack([lablePos, labelNeg])

        dictMats[i] = matrix
        dictLabels[i] = label
    return dictMats,dictLabels

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
    dirname = "test"
    dictMats,dictLabels = load_data(dirname)
    mutilTest(dictMats,dictLabels)
