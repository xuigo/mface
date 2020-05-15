import os
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import glob
import time

import time
from face_alignment.face_align import face_align

raw_path='../dataset/test'
face_path='../dataset/face_area'
model_path='model/shape_predictor_68_face_landmarks.dat'

os.makedirs(face_path,exist_ok=True)

class mface(object):
    def __init__(self):
        self.aligntor=face_align(raw_path,face_path,model_path)    
        self.dirCheck()
    def dirCheck(self):       
        os.makedirs(face_path,exist_ok=True)
        self.face_ext()
    def face_ext(self):
        self.aligntor.aligntor()
        self.ref_images = [os.path.join(face_path, x) for x in os.listdir(face_path)]
        self.ref_images = list(filter(os.path.isfile, self.ref_images))
        if len(self.ref_images) == 0:
            raise Exception('%s is empty' % raw_path)
if __name__ == "__main__":
    mface_=mface()