#!/usr/bin/env python
__author__ = "Alireza Moradzadeh"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Alireza Moradzadeh"
__email__ = "moradza2@illinois.edu"


import numpy as np
import os, sys
import time
import tensorflow as tf
import csv

def train(model, data_set_train, data_set_val, learning_rate=0.0005,
          batch_size=16, num_steps=5000, step_save=1000, step_show=1000,
         pre_train=False, working_dir=None):
    """
    Training loop of mini-batch gradient descent.
    Performs mini-batch gradient descent with the indicated batch_size and
    learning_rate.
    ----------------------------------------------------------------------
    Inputs:
        model(ForceFieldNN): Initialized a NN Force Field model.
        data_set: MD dataset.
        learning_rate(float): Learning rate.
        batch_size(int): Batch size used for training.
        num_steps(int): Number of steps to run the update ops.
    """
    if pre_train:
        print("not implemented yet!")
    else:
        for step in range(0, num_steps):
            batch_r2, batch_z2, batch_r3_jik, batch_z3_jik, batch_r3_ijk, batch_z3_ijk, batch_r3_jki, batch_z3_jki, batch_force= data_set_train.next_batch(batch_size)
            model.session.run(model.update_op_tensor,
                    feed_dict={model.R2_ph: batch_r2, model.Z2_ph: batch_z2,
                               model.R3_jik_ph: batch_r3_jik, model.Z3_jik_ph: batch_z3_jik,
                               model.R3_ijk_ph: batch_r3_ijk, model.Z3_ijk_ph: batch_z3_ijk, 
                               model.R3_jki_ph: batch_r3_jki, model.Z3_jki_ph: batch_z3_jki, 
                               model.F_ph: batch_force,
                               model.lr_placeholder: learning_rate})

            if step % step_save == 0:
                model.save(step)
            if step % step_show == 0:
                loss_train = model.session.run(model.mae_loss,
                    feed_dict={model.R2_ph: batch_r2, model.Z2_ph: batch_z2,
                               model.R3_jik_ph: batch_r3_jik, model.Z3_jik_ph: batch_z3_jik,
                               model.R3_ijk_ph: batch_r3_ijk, model.Z3_ijk_ph: batch_z3_ijk, 
                               model.R3_jki_ph: batch_r3_jki, model.Z3_jki_ph: batch_z3_jki,
                               model.F_ph: batch_force})

                batch_r2, batch_z2, batch_r3_jik, batch_z3_jik, batch_r3_ijk, batch_z3_ijk, batch_r3_jki, batch_z3_jki, batch_force= data_set_val.next_batch(5*batch_size)

                loss_val = model.session.run(model.mae_loss,
                    feed_dict={model.R2_ph: batch_r2, model.Z2_ph: batch_z2,
                               model.R3_jik_ph: batch_r3_jik, model.Z3_jik_ph: batch_z3_jik,
                               model.R3_ijk_ph: batch_r3_ijk, model.Z3_ijk_ph: batch_z3_ijk, 
                               model.R3_jki_ph: batch_r3_jki, model.Z3_jki_ph: batch_z3_jki,
                               model.F_ph: batch_force})
                
                os.chdir(os.path.join(working_dir))
                with open('loss.dat','a') as f:
                    f.write(str(step+model.global_step )+' ')
                    f.write(str(loss_train)+' ')
                    f.write(str(loss_val)+' ')
                    f.write('\n')
                    f.close()
