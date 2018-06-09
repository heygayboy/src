#using q_table to select trail
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
import scipy.io as scio  
import sklearn.metrics as skmetrics
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import imp 
import multiprocessing as mp
from scipy.stats import ttest_ind 


'''-------------------
       total config
-------------------'''
dataNew = 'D:/BCI_dataNew.mat' 
file_directory = 'D:/BCI_q_table.csv'
trail_num = 3703 #num of selected trails less than 7938L
epsilon = 0.00001


def not_empty(s):
    return s and s.strip()

def load_data(datafile=dataNew):
    data = scio.loadmat(dataNew) 
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    return X_train, X_test, y_train, y_test

def train_cls(cls, xtrain, ytrain):
    cls.fit(xtrain, ytrain)
     
def cal_accuracy(cls, xtest, ytest):
#     with open('save/clf.pickle', 'rb') as f:
#         cls = pickle.load(f)
#     X = np.array(env.x_test, np.float64)#.reshape(300,432)
#     y = np.concatenate ( np.array(env.y_test).reshape(env.y_test.shape[0],1) )
    Y_pred= cls.predict( xtest )
#     print(Y_pred)
#     print(ytest)
    score2= skmetrics.accuracy_score(list(ytest), list(Y_pred))
    #score2 = cls.score(X, y)
    return score2   

if __name__ == "__main__":
    cfg_module  = "config.py"
    if cfg_module[-3:]=='.py':
        cfg_module= cfg_module[:-3]
    cfg= imp.load_source(cfg_module, "./config.py")
    X_train, X_test, y_train, y_test = load_data(dataNew)
    print(X_train.shape, X_test.shape)
    '''
    read q_value from file into an array and form a new state_value array
    '''
    df = pd.read_csv(file_directory, header = 0)
    array = df.values #q_value
    
    value = np.max(array[:, 1: ], axis = 1)
    state = array[:, 0]
    state_value = np.vstack((state, value)).T
    #print(state_value.shape)
    '''
    sorted state_value array (ascending = False)
    '''
    sorted_value = np.argsort(-state_value[:, 1], axis=0)
    sorted_state_value = state_value[sorted_value]
    #print(sorted_state_value[:,1])
    
    assert trail_num < X_train.shape[0]
    feature = sorted_state_value[0: trail_num, 0]
    print("Notice: number of trial to be chosen must less than number of observation space.")
    print("number of observation space = {0}".format(feature.shape[0]))
    trail_feature = feature[0]
    #print (trail_feature)
    t = trail_feature.split(' ')  #final element is [']]] 
    t = filter(not_empty, t)
    #print(t)
    #t.remove()
    xdata=np.zeros( (trail_num, len(t)) )
    for trail in range(trail_num):  #trail_num
        trail_feature = feature[trail]
        #print(trail_feature)
        t = trail_feature.split(' ')
        t[0] = t[0][2:] #delete [[
        t[-1] = t[-1][:-2]
        t = filter(not_empty, t)
        #results = list(map(int,t))
        #print(t[1])
        for feature_num in range(len(t)):
            if(t[feature_num] != ''):
                print(t[feature_num])
                xdata[trail][feature_num] = float(t[feature_num])
    #print(xdata[0]) # (98L, 432L)
    
    
    trail_index = []
    for i in range(xdata.shape[0]):
        #print(xdata[i,:])
        temp = np.where(np.sum(np.abs(xdata[i,:] - X_train), axis = 1) < epsilon)
        trail_index.append(temp)
    print (trail_index)
    
    print(y_train.shape)
    x_new_train = np.array(np.squeeze(X_train[trail_index]), np.float64)
    y_new_train = np.array(np.squeeze(y_train[trail_index,:]), np.float64)
    y_new_test = np.array(np.squeeze(y_test), np.float64)
    X_train = np.array(X_train, np.float64)
    y_train = np.array(y_train, np.float64)
#     print(x_new_train.shape)
#     print(y_new_train.shape)
    #cls = SGDClassifier(loss="hinge", penalty="l2")
    
    '''
    compare two kinds of data samples in training cls
    (1) x_new_train and y_new_train (after selected)----accuracy_new
    (2) X_train and y_train-------accuracy_old
    test data are same
    ''' 
    cls= RandomForestClassifier(n_estimators=cfg.RF['trees'], max_features='auto',\
                                    max_depth=cfg.RF['maxdepth'], n_jobs=mp.cpu_count(),class_weight='balanced' )
    accuracy_old = []
    accuracy_new = []
    for i in range(1):
        print('----------The {0} times training began:---------------').format(i)
        train_cls(cls, x_new_train, y_new_train)
        accuracy = cal_accuracy(cls, X_test, y_new_test)
        print('clf accuracy trained by selected trails: accuracy = {0}').format(accuracy)
        accuracy_new.append(accuracy)
#         print(X_train.shape)
#         print(y_train.shape)
        train_cls(cls, X_train, y_train)
        accuracy = cal_accuracy(cls, X_test, y_new_test)
        print('clf accuracy trained by original trails: accuracy = {0}').format(accuracy)
        accuracy_old.append(accuracy)
    print(np.mean(accuracy_old), np.mean(accuracy_new) )
    
    '''
    t-test
    '''
    print(ttest_ind(accuracy_old, accuracy_new))
    
        
        
    
#     for i in range(X_train.shape[0]):
#         if ((np.abs(xdata[0,0] - X_train[i, 0] )) < epsilon):
#             print(i)
#             print(xdata[0,:] , X_train[i, :])

    
        #print(X_train[np.where(xdata == X_train )])
    #X_train, X_test, y_train, y_test = xdata_split(x_data, y_data, test_size=.3)
    
    

