import argparse
import glob
import os

import scipy.misc


import cv2
import imageio

img_dir_2='./1'
img_dir='./2'


files = glob.glob(os.path.join(img_dir,'*'))
files.sort()

handle=[]
for i in range(1,16):
    file1=os.path.join(img_dir,'%d.jpeg'%i)
    file2=os.path.join(img_dir_2,'%d.jpeg'%i)

    src1=cv2.resize(cv2.imread(file1),(480,480))
    src2=cv2.resize(cv2.imread(file2),(480,480))


    dst=cv2.hconcat([src1,src2])

    dst=cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
    handle.append(dst)

    # print(file)
    # handle.append(imageio.imread(file))
imageio.mimsave('example1.gif', handle,duration=0.3)