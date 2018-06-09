"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from environment import Env
from RL_model import QLearningTable
import pickle
import numpy as np
#from sklearn.cross_validation import StratifiedShuffleSplit, LeaveOneOut
from sklearn.model_selection import train_test_split 
import sklearn.metrics as skmetrics
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio  
import Acquire_data

dataNew = 'D:/BCI_dataNew.mat'  #SAVE PATH
QTable_path = 'D:/BCI_q_table.csv'

def update(env):
    accuracy_all = []
    for episode in range(5):
        accuracy_episode = []
        print("{0} episode begin to train:".format(episode))
        # initial observation
        observation = env.reset()

        while True:
            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            print("Take action: Choose Trail {0}".format(action))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            print("Get a reward: {0}".format(reward))
            
            acuracy = cal_accuracy(env.cls)
            accuracy_episode.append(acuracy)
            print("accuracy = ", acuracy)
            
            #cross_validation(env.cls)
            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))
            # swap observation
            observation = observation_
            
            # break while loop when end of this episode
            if done:
                accuracy_all.append(accuracy_episode)
                break

    # end of game
    print('train over')
    return accuracy_all

def cross_validation(cls):
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
    X = np.array(env.x_data, np.float64)#.reshape(300,432)
    y = np.concatenate ( np.array(env.y_data, np.int32).reshape(300,1) )
#     print(type(X[1][1]))
#     print(type(y[1]))
#     print(X.shape)
#     print(y.shape)
    scores = cross_val_score(cls, X, y, cv=5, scoring='accuracy')
    print("cross_validation score: {0}".format(scores))
    print("mean cross_validation score(cv = 5) is {0}".format(scores.mean()))



def cal_accuracy(cls):
#     with open('save/clf.pickle', 'rb') as f:
#         cls = pickle.load(f)
    X = np.array(env.x_test, np.float64)#.reshape(300,432)
    y = np.concatenate ( np.array(env.y_test).reshape(env.y_test.shape[0],1) )
    Y_pred= cls.predict( X )
    #print(Y_pred)
    #print(X_test)
    #score1 = cls.score(X, y)
    score2= skmetrics.accuracy_score(list(y), list(Y_pred))
    #score2 = cls.score(X, y)
    return score2

def xdata_split(x_data, y_data, test_size=.3):
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.3, random_state=0)    
    return  X_train, X_test, y_train, y_test
    
def preprocessing(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)  # Don't cheat - fit only on training data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def load_env():
    #load file and data
    x_data, y_data = Acquire_data.data_aquire()
    X_train, X_test, y_train, y_test = xdata_split(x_data, y_data, test_size=.3)
    #preprocessing
    X_train, X_test = preprocessing(X_train, X_test)
    print(X_train.shape, X_test.shape)
    #save data to mat
    scio.savemat(dataNew, {'X_train':X_train, 'y_train': y_train, 'X_test':X_test, 'y_test': y_test})  
    #bulid an env for testing
    env = Env(X_train, y_train, X_test, y_test)
    return env

def sort_value(column):  
    new_column = column.sort_values(ascending = False)  
    return new_column.iloc[0]  
      

if __name__ == "__main__":
    
    env = load_env()
    cfg = env.cfg
    print( list(range(env.n_actions)) )
    RL = QLearningTable(actions=list(range(env.n_actions)))

    accuracy_all = update(env)
#     print(accuracy_all)
    accuracy_index = []
    for i in range(len(accuracy_all)): 
        accuracy_index.append('episode_' +  str(i+1))
    accuracy_csv=pd.DataFrame(accuracy_all, index = accuracy_index)
    accuracy_csv.to_csv('D:/accuracy_csv2.csv')
    #plot MSE-iteration 
    for i in range(len(accuracy_all)):
        x_ = range(len(accuracy_all[i]))
        plt.plot(x_, accuracy_all[i],'--', label = ('episode = %d' % (i))) 
            #plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
    

    with open('save/clf.pickle', 'rb') as f:
        cls = pickle.load(f)
    #cross_validation(cls)
    RL.q_table.to_csv(QTable_path)
#     print(RL.q_table.index[0])
#     scio.savemat(QTable_path, {'Index':RL.q_table.index, 'value': RL.q_table.values, 'column':RL.q_table.columns})  
    acuracy = cal_accuracy(cls)
    print(acuracy)

    


    
    
    
    
    
    