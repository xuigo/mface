import os
import glob
import numpy as np
import pickle
import h5py
import pandas as pd
import glob
from datetime import datetime
import feature_axis as feature_axis

latent_dir='../dataset/dlatent'
data_dir='./data'

os.makedirs(data_dir,exist_ok=True)

def get_data():
    y=[]
    z=[]
    y_name=[]
    label=[-1 for x in range(0,len(os.listdir(latent_dir)))]
    for index,sub_dir in enumerate(os.listdir(latent_dir)):
        y_name.append(sub_dir)
        label[index]=1
        sub_dir=os.path.join(latent_dir,sub_dir)
        for filename in glob.glob(os.path.join(sub_dir,'*.npy')):
            # print('filename',filename)
            # filepath=os.path.join(sub_dir,filename)
            latent_code=np.load(filename)
            latent_code=latent_code.ravel()
            z.append(latent_code)
            label_=np.array(label)
            y.append(label_)
    pathfile_y = os.path.join(data_dir, 'y.h5')
    with h5py.File(pathfile_y, 'w') as f:
        f.create_dataset('y', data=y)
    pathfile_z = os.path.join(data_dir, 'z.h5')
    with h5py.File(pathfile_z, 'w') as f:
        f.create_dataset('z', data=z)
    return y,z,y_name

def feature_direction():
    '''path_celeba_att = './list_attr_celeba.txt'
    df_attr = pd.read_csv(path_celeba_att, sep='\s+', header=1, index_col=0)
    y_name = df_attr.columns.values.tolist()'''
    y,z,y_name=get_data() 
    z=np.array(z)
    y=np.array(y)
    print('z-shape:{}'.format(z.shape))
    print('y-shape:{}'.format(y.shape))
    feature_slope = feature_axis.find_feature_axis(z, y, method='tanh')
    #feature_slope,bias = feature_axis.Ridge_model(z,y)
    #print(bias)
    yn_normalize_feature_direction = False
    if yn_normalize_feature_direction:
        feature_direction = feature_axis.normalize_feature_axis(feature_slope)
    else:
        feature_direction = feature_slope
    pathfile_feature_direction = os.path.join(data_dir, 'feature_direction_{}.pkl'.format(datetime.now().strftime('%H_%M')))
    dict_to_save = {'direction': feature_direction, 'name': y_name}
    with open(pathfile_feature_direction, 'wb') as f:
        pickle.dump(dict_to_save, f)
        
if __name__=='__main__':
    feature_direction()