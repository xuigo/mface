import os
import bz2
import argparse
import tqdm 

from keras.utils import get_file
from face_alignment.face_alignment import image_align
from face_alignment.landmarks_detector import LandmarksDetector

import multiprocessing
import PIL.Image
import numpy as np

LANDMARKS_MODEL_URL='http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    
class face_align(object):
    
    def __init__(self,raw_dir,align_dir,model_path,output_size=1024):
        self.url=LANDMARKS_MODEL_URL
        
        self.raw_dir=raw_dir
        self.align_dir=align_dir
        
        self.model_path=model_path
        self.output_size=output_size
        self.x_scale=1.
        self.y_scale=1.
        self.em_scale=0.1
        self.use_alpha=False
        self.find_faces=True
        self.landmarks_model_path=model_path
        #self.landmarks_model_path=self.unpack_bz2(get_file(self.model_path,self.url,cache_subdir='temp'))
        
        os.makedirs(self.align_dir,exist_ok=True)
        self.detector_=LandmarksDetector(self.landmarks_model_path)
        
    def unpack_bz2(self,src_path):
        data=bz2.BZ2File(src_path).read()
        dst_path=src_path[:-4]
        with open(dst_path,'wb') as fp:
            fp.write(data)
        return dst_path 
    
    def aligntor(self):
        for img_name in tqdm.tqdm(os.listdir(self.raw_dir)):
            try:
                raw_img_path=os.path.join(self.raw_dir,img_name)
                fn=face_img_name='%s.png'%(os.path.splitext(img_name)[0])
                
                '''if os.path.isfile(fn):                    
                    continue'''
                if self.find_faces:
                    landmarks=list(self.detector_.get_landmarks(raw_img_path))
                    assert len(landmarks)==1
                else:
                    landmarks=[[(89, 230), (90, 258), (91, 287), (93, 317), (104, 344), (122, 368), (144, 387), (171, 406),
                     (203, 414), (236, 409), (262, 392), (284, 370), (302, 345), (310, 317), (312, 289), (312, 260),
                     (311, 233), (114, 214), (129, 199), (149, 192), (170, 193), (190, 202), (228, 201), (248, 192),
                     (268, 190), (287, 196), (299, 210), (210, 222), (211, 241), (212, 260), (212, 280), (184, 290),
                     (197, 294), (211, 300), (225, 294), (238, 288), (144, 227), (155, 223), (167, 222), (179, 228),
                     (167, 232), (154, 231), (241, 227), (251, 222), (264, 221), (275, 226), (265, 230), (252, 230),
                     (153, 323), (174, 321), (194, 320), (211, 323), (226, 319), (243, 320), (261, 323), (244, 344),
                     (227, 350), (211, 352), (194, 350), (173, 343), (159, 324), (195, 326), (211, 327), (226, 326),
                     (255, 324), (226, 340), (211, 342), (194, 341)]]
                for i, landmarks_ in enumerate(landmarks,start=1):
                    try:
                        face_img_name='%s.png'%(os.path.splitext(img_name)[0])
                        aligned_face_path=os.path.join(self.align_dir,face_img_name)
                        
                        image_align(raw_img_path,aligned_face_path,landmarks_,output_size=self.output_size,x_scale=self.x_scale,y_scale=self.y_scale,em_scale=self.em_scale,alpha=self.use_alpha,find_faces=self.find_faces)
                    except Exception as e:
                        print(' [!] Exception in face alignment!',str(e))
            except Exception as e:
                print(' [!] Exception in landmark detection!',str(e))
                        
if __name__=='__main__':
    align=face_align('test','result')
    align.aligntor()
                        
                     
            
        
    