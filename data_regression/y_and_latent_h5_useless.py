import os
import glob
import numpy as np
import PIL.Image
import h5py

path='/home/xushaohui/FaceEdit_/data/RaFD/RaFD_class_npy'

list_z=[]
labels=[]

for index,subdir in enumerate(os.listdir(path)):
    
    label=[-1,-1,-1,-1,-1,-1]
    label[index]=1
    print('{} :{}'.format(subdir,label))
    newdir=os.path.join(path,subdir)
    for file in os.listdir(newdir):
        filepath=os.path.join(newdir,file)
        
        latent_code=np.load(filepath)
        latent_code=latent_code.ravel()
        
        list_z.append(latent_code)
        
        label = np.array(label)
        labels.append(label)
        
pathfile_sample_y = os.path.join('./', 'label_y_emotion.h5')
with h5py.File(pathfile_sample_y, 'w') as f:
    f.create_dataset('y', data=labels)
pathfile_sample_z = os.path.join('./', 'latent_code_emtion.h5')
with h5py.File(pathfile_sample_z, 'w') as f:
    f.create_dataset('z', data=list_z)
        
        
            
    
