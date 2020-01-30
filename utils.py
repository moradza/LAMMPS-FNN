#!/usr/bin/env python
__author__ = "Alireza Moradzadeh"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Alireza Moradzadeh"
__email__ = "moradza2@illinois.edu"

import numpy as np
import os, sys
from numpy.linalg import norm
import time
import csv


class data_tool():
    def __init__(self, data_path, file_name='file_train.dat', R_max=6., normalize=True):
        self.data_path = os.path.join(data_path)
        self.order_num = 0
        self.batch_num = 0
        self.dataset_order = []
        
        os.chdir(self.data_path)
        with open(file_name, 'r') as f:
            self.dataset_order = f.read().split('\n')[:-1]
            
        self.dataset_size = len(self.dataset_order)
        self.R_max = R_max
        self.number = 0
        self.normalize = normalize
         
    def next_batch(self, batch_size):
        num = self.dataset_order[self.number]
        R2_file = 'R2_'+num+'.npy'
        R3_jik_file = 'R3_jik_'+num+'.npy'
        R3_ijk_file = 'R3_ijk_'+num+'.npy'
        R3_jki_file = 'R3_jki_'+num+'.npy'
        F_file = 'F_'+num+'.npy'
        os.chdir(self.data_path)
        R2 = np.load(R2_file)
        R3_jik = np.load(R3_jik_file)
        R3_ijk = np.load(R3_ijk_file) 
        R3_jki = np.load(R3_jki_file)
        F = np.load(F_file)
        
        Ni2_size = R2.shape[1]
        Ni3_size = R3_jik.shape[1]
        batch_num = int(F.shape[0]/batch_size)
        
        if self.batch_num >= batch_num:    
            self.number += 1
            self.batch_num = 0
        if self.number >= len(self.dataset_order):
            self.number = 0
        self.batch_num += 1
        indx_start = (self.batch_num - 1)*batch_size
        indx_end = self.batch_num * batch_size
        
        # 2 body features
        R2_output = norm(R2[indx_start:indx_end], axis= -1) # (batch_size, Ni2_size, 3) --> (batch_size, Ni2_size)
        R2_output = R2_output.reshape((batch_size, R2_output.shape[1], 1)) # (batch_size, Ni2_size) --> (batch_size, Ni2_size, 1)
        Z2_output = R2[indx_start:indx_end]/R2_output # (batch_size, Ni2_size,3)
        R2_output = R2_output.reshape((batch_size, R2_output.shape[1], 1,1)) # (batch_size, Ni2_size,1) --> (batch_size, Ni2_size,1, 1)
        
        # 3 body features jik
        R3_output_jik = norm(R3_jik[indx_start:indx_end], axis= -1) #  (batch_size, Ni3_size,3, 3) --> (batch_size, Ni3_size,3)
        R3_output_jik = R3_output_jik.reshape((batch_size,R3_output_jik.shape[1],1,3)) #  (batch_size, Ni3_size,3) --> (batch_size, Ni3_size,1, 3)
        Z3_output_jik =R3_jik[indx_start:indx_end,:,:2,:]/R3_output_jik[:,:,:,:2].reshape((batch_size,Ni3_size,2,1))  #  (batch_size, Ni3_size,2, 3) 
        
        # 3 body features ijk
        R3_output_ijk = norm(R3_ijk[indx_start:indx_end], axis= -1) #  (batch_size, Ni3_size,3, 3) --> (batch_size, Ni3_size,3)
        R3_output_ijk = R3_output_ijk.reshape((batch_size,R3_output_ijk.shape[1],1,3)) #  (batch_size, Ni3_size,3) --> (batch_size, Ni3_size,1, 3)
        Z3_output_ijk =R3_ijk[indx_start:indx_end,:,:2,:]/R3_output_ijk[:,:,:,:2].reshape((batch_size,Ni3_size,2,1))  #  (batch_size, Ni3_size,2, 3) 
        
        # 3 body features jki
        R3_output_jki = norm(R3_jki[indx_start:indx_end], axis= -1) #  (batch_size, Ni3_size,3, 3) --> (batch_size, Ni3_size,3)
        R3_output_jki = R3_output_jki.reshape((batch_size,R3_output_jki.shape[1],1,3)) #  (batch_size, Ni3_size,3) --> (batch_size, Ni3_size,1, 3)
        Z3_output_jki =R3_jki[indx_start:indx_end,:,:2,:]/R3_output_jki[:,:,:,:2].reshape((batch_size,Ni3_size,2,1))  #  (batch_size, Ni3_size,2, 3) 
        
        # force target
        F_output = F[indx_start:indx_end,:]
        if self.normalize:
            return R2_output/self.R_max, Z2_output, R3_output_jik/self.R_max, Z3_output_jik, R3_output_ijk/self.R_max, Z3_output_ijk, R3_output_jki/self.R_max, Z3_output_jki, F_output
        else:
            return R2_output, Z2_output, R3_output_jik, Z3_output_jik, R3_output_ijk, Z3_output_ijk, R3_output_jki, Z3_output_jki, F_output
