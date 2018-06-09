import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
import sklearn
import imp 
import multiprocessing as mp
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import sklearn.metrics as skmetrics
import Acquire_data
from sklearn import svm 

'''
df1 = pd.read_csv("D:/psd_data.csv")
psd_data = df1.values[:, 1:]
#print(psd_data)
x_data = psd_data[:, :-1]
print(x_data.shape)  #(300L, 432L)
y_data = psd_data[:, -1]
print(y_data.shape)  #(300L,)
'''

class Env(object):
  
    def __init__(self, x_data, y_data, x_test, y_test):
        self.x_data = x_data
        self.y_data = y_data
        self.x_test = x_test
        self.y_test = y_test
        self.sigma = 0.1
        self.iteration = 2000
        self.nstep = 0
        
        self.action_space = []
        for i in range(self.x_data.shape[0]): 
            self.action_space.append(i)
        self.n_actions = len(self.action_space)
        
        #initial cls
        cfg_module  = "config.py"
        if cfg_module[-3:]=='.py':
            cfg_module= cfg_module[:-3]
        self.cfg= imp.load_source(cfg_module, "./config.py")
        
        #self.cls = SGDClassifier(loss="hinge", penalty="l2")
        self.cls = svm.SVC()
        
        
#         self.cls= RandomForestClassifier(n_estimators=self.cfg.RF['trees'], max_features='auto',\
#             max_depth=self.cfg.RF['maxdepth'], n_jobs=mp.cpu_count(), class_weight='balanced' )
    
    
    
        
    def after_action(self, action):
        next_state =   np.array( self.x_data[action, :].reshape(1,-1)) 
        y_value = np.array( self.y_data[action] ).reshape(1,) 
        return next_state, y_value
     
    def reset(self):
        self.cls = SGDClassifier(loss="hinge", penalty="l2")
        self.nstep = 0
        init = random.randint(0, self.x_data.shape[0]-1)
        initial_state, initial_y = self.after_action(init)
        self.cls.partial_fit( initial_state, initial_y , classes = np.unique(np.array(self.y_data, np.float64)))
        '''
        x = np.array(self.x_data, np.float64)
        y = np.array(self.y_data , np.float64)
        self.cls.fit( x, y )
        '''
        return initial_state
           
    def step(self, action):
        self.nstep = self.nstep + 1
        print("step {0}".format(self.nstep))
        s_, Y = self.after_action( action )
        #print(s_[1])

        # reward function
        Y_pred= self.cls.predict( s_ )
        if (np.square(Y_pred-Y) < self.sigma):
            reward = 1
            
        else:
            reward = 0
            self.cls.partial_fit( s_, Y )
            #self.cls.fit( np.array(self.x_data).reshape(300,432), np.array(self.y_data) )
            
        
        if(self.nstep > self.iteration):
            done = True
            with open('save/clf.pickle', 'wb') as f:
                pickle.dump(self.cls, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            done = False
        
        return s_, reward, done  

def xdata_split(x_data, y_data, test_size=.3):
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.3, random_state=0)    
    return  X_train, X_test, y_train, y_test
    
def preprocessing(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)  # Don't cheat - fit only on training data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

if __name__ == "__main__":
    #load file and data
    x_data, y_data = Acquire_data.data_aquire()
    X_train, X_test, y_train, y_test = xdata_split(x_data, y_data, test_size=.3)
    
    #preprocessing
    X_train, X_test = preprocessing(X_train, X_test)
    print(X_train.shape, X_test.shape)
    
    '''
    #SGD
    clf = SGDClassifier(loss="hinge", penalty="l2")
    X_train = np.array(X_train, np.float64)
    y_train = np.concatenate ( np.array(y_train, np.float64).reshape(y_train.shape[0],1))
    #y = np.concatenate ( np.array(env.y_test).reshape(env.y_test.shape[0],1) )
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    score= skmetrics.accuracy_score(list(y_test), list(y_predict))
    print(score)
    '''
    clf = svm.SVC()
    X_train = np.array(X_train, np.float64)
    y_train = np.concatenate ( np.array(y_train, np.float64).reshape(y_train.shape[0],1))
    #y = np.concatenate ( np.array(env.y_test).reshape(env.y_test.shape[0],1) )
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    score= skmetrics.accuracy_score(list(y_test), list(y_predict))
    print(score)
    
    
   
    
    
    
    #bulid an env for testing
    test_env = Env(X_train, y_train, X_test, y_test)
    
    
    #(1) test reset
    for i in range(20):
        observation = test_env.reset()
        print(observation.shape)
    
    
    
    '''
    #(2) test s_
    test_env = Env(X_train, y_train, X_test, y_test)
    observation = test_env.reset() 
    for i in range(20):
        action = np.random.choice(test_env.action_space)
        print (action)
        test_env.step(action)
    '''
    
    
    '''
    #(3) test reward
    test_env = Env(X_train, y_train, X_test, y_test)
    observation = test_env.reset() 
    while (True):
        action = np.random.choice(test_env.action_space)
        print (action)
        observation_, reward, done = test_env.step(action) 
        observation = observation_
        
        if done:
            break   
    '''
    

    
    
        
            
