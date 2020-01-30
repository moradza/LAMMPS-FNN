#!/usr/bin/env python
__author__ = "Alireza Moradzadeh"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Alireza Moradzadeh"
__email__ = "moradza2@illinois.edu"


import os
from model import *
from utils import *
from train import *
Ni2_size = 20
Ni3_size = 180
load_dir = os.getcwd()
print("load_dir:", load_dir)
data_dir = os.path.join('/u/sciteam/moradzad/scratch/DNN_FF/SW/CNN/data.300')


model = ForceFeildNN(load_dir=load_dir, Ni2_size=Ni2_size, Ni3_size=Ni3_size)

data_train = data_tool(data_dir, file_name='train.dat')
data_val =  data_tool(data_dir, file_name='validation.dat')

train(model, data_train, data_val, learning_rate=0.0005,
          batch_size=102, num_steps=700001, step_save=1000, step_show=200,
         pre_train=False, working_dir=load_dir)
