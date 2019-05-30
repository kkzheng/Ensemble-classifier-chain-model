import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from ECC.fea_extract import protein_exp
from ECC.fea_sel import extra_PSE,extra_DWT
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

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

def save_model(model,filename):
    import pickle
    with open(filename,'wb') as f:
        pickle.dump(model,f)

kfold = StratifiedKFold(n_splits=5)
def fit_model(classifier,params,X,y):
    scorer = make_scorer(roc_auc_score)
    tuning = GridSearchCV(classifier,param_grid=params,scoring=scorer,cv=kfold)
    tuning.fit(X,y)
    print(tuning.best_params_)
    print(tuning.best_score_)
    return tuning.best_estimator_

def train(X,y):
    y = y.reshape(1,-1)[0]
    clf_rf = RandomForestClassifier(random_state=1)
    param_rf = {"n_estimators":[50,70,80,100,120,130,140,150,160,180,200,220,230,240,250,260,300,310,320,330,340,350,360,380,400,420,500,550,600,650,700]}
    clf_rf_best = fit_model(clf_rf,param_rf,X,y)
    labelPre = clf_rf_best.predict(X)
    return clf_rf_best,labelPre[:,np.newaxis]

def randSelect(dataIndex,i):
    j = dataIndex.index(i)
    while i == dataIndex[j]:
            j = int(np.random.uniform(0,len(dataIndex)))
    return j

def oneChain(dictMats,dictLabels,last):
    l = len(dictMats)
    dataIndex = list(range(l))
    chains = []
    save_index = []
    labelPres = np.array([])
    for i in range(l-1):
        randomIndex = randSelect(dataIndex,last)
        index = dataIndex[randomIndex]
        if i == 0:
            model,labelPre = train(dictMats[index],dictLabels[index])
            labelPres = labelPre
        else:
            model,labelPre = train(np.hstack([dictMats[index],labelPres]),dictLabels[index])
            labelPres = np.hstack([labelPres,labelPre])
        chains.append(model)
        del (dataIndex[randomIndex])
        save_index.append(str(index))
    model, labelPre = train(np.hstack([dictMats[last], labelPres]), dictLabels[last])
    chains.append(model)
    save_index.append(str(last))
    print('randIndex_list :',save_index)
    labelPres = np.hstack([labelPres, labelPre])
    return chains,save_index,labelPre

def multiCC(dictMats,dictLabels):
    multiClissiferChain = []
    multiSaveIndex = []
    for i in range(len(dictMats)):
        print("Chain -----------------------------------------> ",i)
        chain,save_index,labelOfOneChian = oneChain(dictMats,dictLabels,i)
        multiClissiferChain.append(chain)
        multiSaveIndex.append(save_index)
        if i == 0:
            pEncoder = labelOfOneChian
        else:
            pEncoder = np.hstack([pEncoder,labelOfOneChian])
    return pEncoder,multiClissiferChain,multiSaveIndex

def train_model(dictMats, dictLabels):
    X,ccModel,ccIndex = multiCC(dictMats, dictLabels)
    y = np.hstack([np.ones((1, 690)), np.zeros((1, 1009))])[0]
    print(X.shape)
    rf = RandomForestClassifier(random_state=1)
    param_rf = {"min_samples_split":[2,3],
                "n_estimators": [50,60,70,80,100,120,150,160,200,220,300,350]}
    model = fit_model(rf,param_rf,X,y)
    save_model(model, r'./clf.pickle')
    save_model(ccModel,r'./CCmodel.pickle')
    save_model(ccIndex,r'./index.txt')
    print("Train Finish!")

if __name__ == '__main__':
    dirname = ""
    dictMats,ductLabels = load_data(dirname)
    train_model(dictMats,dictLabels)
