import scipy.io as scio  
import numpy as np

Left_datapath = 'D:/lpsdL.mat' 
Right_datapath = 'D:/lpsdR.mat' 


def data_aquire():
    Left_data = scio.loadmat(Left_datapath)  
    Right_data = scio.loadmat(Right_datapath) 
    
    each_trial = Left_data['lpsdL'].shape[0]
    time_num = Left_data['lpsdL'].shape[1]
    chanel_num = Left_data['lpsdL'].shape[2]
    frequency_num = Left_data['lpsdL'].shape[3]
    
    Left_data = Left_data['lpsdL'].reshape(each_trial*time_num, chanel_num*frequency_num)
    Right_data = Right_data['lpsdR'].reshape(each_trial*time_num, chanel_num*frequency_num)
    #print(Right_data.shape, Left_data.shape)
    
    x_data = np.vstack((Left_data, Right_data))
    x_data = np.array(x_data, np.float64)
    y_data = np.array([0]*each_trial*time_num + [1]*each_trial*time_num, np.float64)
    y_data = y_data.reshape(each_trial*2*time_num, -1)
    #print(x_data.shape, y_data.shape)
    
    return x_data, y_data

if __name__ == "__main__":
    x_data, y_data = data_aquire()
    print(x_data.shape, y_data.shape)

