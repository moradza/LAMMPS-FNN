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
import tensorflow as tf
import csv


class ForceFeildNN():
    def __init__(self, load_dir, Ni2_size=20, Ni3_size= 100, ndims=3, many_body=[2,3]):
        """
        Initialize a NN based force field
        ---------------------------------
        Inputs:
            Ni2_size = int size of neighbor list of two body interactions
            Ni3_size = int size of neighbor list of three body interactions
        """
        self.Ni3_size = Ni3_size
        self.Ni2_size = Ni2_size
        self.many_body = many_body
        self.ndims = ndims
        
        # Create session
        self.session = tf.Session()
        self.load_dir = os.path.join(load_dir, 'checkpoint')
        self.save_dir = os.path.join(self.load_dir,'checkpoint.ckpt')
        self.global_step = 0
        
        # define input placeholder
        self.R2_ph = tf.placeholder(tf.float32,[None, self.Ni2_size, 1, 1], name='R2') # norm of R_ij
        self.Z2_ph = tf.placeholder(tf.float32,[None, self.Ni2_size, self.ndims], name='Z2') # direction of R_ij
        
        self.R3_jik_ph = tf.placeholder(tf.float32,[None, self.Ni3_size, 1, self.ndims], name='R3_jik') # norm of r_ij , r_jk, r_jk
        self.Z3_jik_ph = tf.placeholder(tf.float32,[None, self.Ni3_size, 2, self.ndims], name='Z3_jik') # direction of e_ij, e_jk
        
        self.R3_ijk_ph = tf.placeholder(tf.float32,[None, self.Ni3_size, 1, self.ndims], name='R3_ijk')
        self.Z3_ijk_ph = tf.placeholder(tf.float32,[None, self.Ni3_size, 2, self.ndims], name='Z3_ijk')
        
        self.R3_jki_ph = tf.placeholder(tf.float32,[None, self.Ni3_size, 1, self.ndims], name='R3_jki')
        self.Z3_jki_ph = tf.placeholder(tf.float32,[None, self.Ni3_size, 2, self.ndims], name='Z3_jki')
        
        self.F_ph = tf.placeholder(tf.float32,[None, self.ndims], name='Force') # Ground truth force
        
        
        # Use model to get the forces from 2body 3body interactions
        self.F2_pred =  self.mlp2body(self.R2_ph, self.Z2_ph) # Total 2body force contribution
        self.F3_jik_pred =  self.mlp3body(self.R3_jik_ph, self.Z3_jik_ph, arange_type=1) # 3body force contribution with jik
        self.F3_ijk_pred =  self.mlp3body(self.R3_ijk_ph, self.Z3_ijk_ph, arange_type=2) # 3body force contribution with ijk
        self.F3_jki_pred =  self.mlp3body(self.R3_jki_ph, self.Z3_jki_ph, arange_type=3) # 3body force contribution with jik
        
        # Total 3body force claculation
        self.F3_pred = tf.add(tf.add(self.F3_jik_pred, self.F3_ijk_pred),  self.F3_jki_pred) # total 3body force
        
        # Total force claculation
        self.F_pred = tf.add(self.F2_pred, self.F3_pred) # total force
        
        # Define loss Mean Absolute Error
        self.mae_loss = tf.reduce_mean(tf.abs(tf.subtract(self.F_pred, self.F_ph ))) #loss function
        self.lr_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
        
        # Setup model parameters for resue, update_op_tensor
        self.model_param = tf.trainable_variables()
        
        self.mlp3_param = [p for p in self.model_param if p.name.startswith('MLP3')]
        
        self.update_op_tensor = self.update_op(self.mae_loss, self.lr_placeholder)
        
        # Save and load model checkpoints.
        # Saving the model
        self.saver = tf.train.Saver(max_to_keep=None)
        
        # Load the model
        self.loader = tf.train
        self.saver = tf.train.Saver()
        
        # Initialize all variables.
        self.session.run(tf.global_variables_initializer())
        
        # load save model and create save dir
        if os.path.isfile(self.load_dir+'/iteration.dat'):
            self.global_step = int(np.loadtxt(os.path.join(self.load_dir, 'iteration.dat')))
            self.load()
        if not os.path.exists(self.load_dir):
            os.makedirs(self.load_dir)
            self.save(step=0)
        
    def save(self, step):
        """
        Save model at specefic step of training
        ---------------------------------------
        Inputs:
            step = int training step
            save_dir = string path to the stored model
        """
        os.chdir(self.load_dir)
        self.saver.save(self.session, self.save_dir, step+self.global_step)
        np.savetxt('iteration.dat', np.array([step+self.global_step]))
        
    def load(self, ct = 'checkpoint.ckpt-0.meta', step=0):
        """
        Load model at specefic step of training
        ---------------------------------------
        Inputs:
            step = int training step
            load_dir = string path to the stored model
        """
        ct = 'checkpoint.ckpt-'+str(self.global_step)+'.meta'
        ct =os.path.join(self.load_dir, ct )
        saver = self.loader.import_meta_graph(ct)
        if step==0:
            saver.restore(self.session, self.loader.latest_checkpoint(self.load_dir))
        else:
            saver.restore(self.session, os.path.join(self.load_dir,'checkpoint.ckpt-'+str(step)))
        
    def mlp2body(self, r, z, total_force=True, max_2bodyforce=40.0):
        """
        Calculate force of 2body interaction
        ------------------------------------
        Inputs:
            r = placeholder(None, Ni2_size, 1)
            z = placeholder(None, Ni2_size, 3)
            total_force = bool
        Outputs:
            f = (None, 3)
            f_pair = (None, 3)
        Note:
            layer of 2body are named with lowcase initial name, and 3body are named with capcase initial name
        """
        conv1 = tf.layers.conv2d(r,     filters=3,  kernel_size=1, strides=(1, 1), activation= tf.nn.tanh, name='conv1')
        conv2 = tf.layers.conv2d(conv1, filters=9,  kernel_size=1, strides=(1, 1), activation= tf.nn.tanh, name='conv2')
        conv3 = tf.layers.conv2d(conv2, filters=27, kernel_size=1, strides=(1, 1), activation= tf.nn.tanh, name='conv3')
        conv4 = tf.layers.conv2d(conv3, filters=81, kernel_size=1, strides=(1, 1), activation= tf.nn.tanh, name='conv4')
        conv5 = tf.layers.conv2d(conv4, filters=27, kernel_size=1, strides=(1, 1), activation= tf.nn.tanh, name='conv5')
        conv6 = tf.layers.conv2d(conv5, filters=9,  kernel_size=1, strides=(1, 1), activation= tf.nn.tanh, name='conv6')
        conv7 = tf.layers.conv2d(conv6, filters=3,  kernel_size=1, strides=(1, 1), activation= tf.nn.tanh, name='conv7')
        conv8 = tf.layers.conv2d(conv7, filters=1,  kernel_size=1, strides=(1, 1), activation= tf.nn.tanh, name='conv8')
        
        flatted =tf.contrib.layers.flatten(conv8)
        
        output =  tf.reduce_sum(tf.expand_dims(flatted, -1) * z, axis=1)
        
        if total_force:
        
            return max_2bodyforce*output
        else:
            
            return max_2bodyforce*flatted
        
    def mlp3body(self, r, z, arange_type=1, max_3bodyforce=40.0):
        """
        Calculate force of 3body interaction
        ------------------------------------
        Inputs:
            r = placeholder(None, Ni3_size, 1, 3)
            z = placeholder(None, Ni3_size, 2, 3)
            arange_type = 1,2,3 corresponding to jik, ijk, jki
            total_force = bool
        Outputs:
            f = (None, 3)
            f_pair = (None, 3)
        Note:
            layer of 2body are named with lowcase initial name, and 3body are named with capcase initial name
        """
        with tf.variable_scope("MLP3", reuse=tf.AUTO_REUSE) as scope:
            z1 = tf.reshape(z[:,:,0,:], (-1, r.shape[1], 1, 3))
            z2 = tf.reshape(z[:,:,1,:], (-1, r.shape[1], 1, 3))
            conv1 = tf.layers.conv2d(r,      filters=12,  kernel_size=1, strides=(1,1), activation= tf.nn.tanh, name='MLP3Conv1')
            conv2 = tf.layers.conv2d(conv1,  filters=48,  kernel_size=1, strides=(1,1), activation= tf.nn.tanh, name='MLP3Conv2')
            conv3 = tf.layers.conv2d(conv2,  filters=192, kernel_size=1, strides=(1,1), activation= tf.nn.tanh, name='MLP3Conv3')
            conv4 = tf.layers.conv2d(conv3,  filters=384, kernel_size=1, strides=(1,1), activation= tf.nn.tanh, name='MLP3Conv4')
            conv5 = tf.layers.conv2d(conv4,  filters=192, kernel_size=1, strides=(1,1), activation= tf.nn.tanh, name='MLP3Conv5')
            conv6 = tf.layers.conv2d(conv5,  filters=96,  kernel_size=1, strides=(1,1), activation= tf.nn.tanh, name='MLP3Conv6')
            conv7 = tf.layers.conv2d(conv6,  filters=48,  kernel_size=1, strides=(1,1), activation= tf.nn.tanh, name='MLP3Conv7')
            conv8 = tf.layers.conv2d(conv7,  filters=24,  kernel_size=1, strides=(1,1), activation= tf.nn.tanh, name='MLP3Conv8')
            conv9 = tf.layers.conv2d(conv8,  filters=12,  kernel_size=1, strides=(1,1), activation= tf.nn.tanh, name='MLP3Conv9')
            conv10 =tf.layers.conv2d(conv9,  filters=6,   kernel_size=1, strides=(1,1), activation= tf.nn.tanh, name='MLP3Conv10')
            conv11 =tf.layers.conv2d(conv10, filters=3,   kernel_size=1, strides=(1,1), activation= tf.nn.tanh, name='MLP3Conv11')

            conv101= tf.layers.conv2d(conv11, filters=1,   kernel_size=1, strides=(1,1), activation= tf.nn.tanh, name='MLP3Conv101') # dot(f_j, e_ij)
            conv102= tf.layers.conv2d(conv11, filters=1,   kernel_size=1, strides=(1,1), activation= tf.nn.tanh, name='MLP3Conv102') # dot(f_j, e_ik)
            conv103= tf.layers.conv2d(conv11, filters=1,   kernel_size=1, strides=(1,1), activation= tf.nn.tanh, name='MLP3Conv103') # dot(f_k, e_ij)
            conv104= tf.layers.conv2d(conv11, filters=1,   kernel_size=1, strides=(1,1), activation= tf.nn.tanh, name='MLP3Conv104') # dot(f_k, e_ik)


            flatted1real = conv101 * z1
            flatted2real = conv102 * z2
            flatted3real = conv103 * z1
            flatted4real = conv104 * z2

            output1 =  tf.reduce_sum(flatted1real, axis=2) +tf.reduce_sum(flatted2real, axis=2)
            output2 =  tf.reduce_sum(flatted3real, axis=2) + tf.reduce_sum(flatted4real, axis=2)


            if arange_type == 1:
                output = tf.reduce_sum(output1, axis=1) + tf.reduce_sum(output2, axis=1)
                return max_3bodyforce*output

            elif arange_type == 2:
                output = tf.reduce_sum(-output1, axis=1)
                return max_3bodyforce*output
            
            elif arange_type == 3:
                output = tf.reduce_sum(-output2, axis=1)
                return max_3bodyforce * output
        

    def update_op(self, loss, learning_rate):
        """Creates the update optimizer.
        Use tf.train.AdamOptimizer to obtain the update op.
        -------------------------------------------------------
        Inputs:
            loss(tf.Tensor): Tensor of shape () containing the loss function.
            learning_rate(tf.Tensor): Tensor of shape (). Learning rate for
                gradient descent.
        Returns:
            train_op(tf.Operation): Update opt tensorflow operation.
        """
        train_op = tf.train.AdamOptimizer(learning_rate,name='Adam').minimize(loss)
        
        return train_op
