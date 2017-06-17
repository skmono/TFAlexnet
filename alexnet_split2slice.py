#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 02:58:55 2017

@author: monolese
"""

import numpy as np
import cv2
import time

import tensorflow as tf

def format_im(image, imSize=[227,227]):
    resized = cv2.resize(image, (imSize[0], imSize[1])).astype(np.float32)
    resized = resized - np.mean(resized)
    
    return np.reshape(resized, (1, imSize[0], imSize[1], 3))

def conv2d(name, x, w, b, s, group=0):
    print(name, group)
    with tf.variable_scope(name):
        if group==0:
            i = tf.nn.conv2d(x, w, strides=[1,s,s,1], padding='SAME')
            j = tf.nn.bias_add(i, b)
        else:
            print(x.get_shape())
            print(w.get_shape())
            xd = [int(i) for i in x.get_shape()]
            wd = [int(i) for i in w.get_shape()]
            
            xslice = [tf.slice(x, [0,0,0,0], xd[:3]+[int(xd[3]/2)]), tf.slice(x, [0,0,0,int(xd[3]/2)], xd[:3]+[int(xd[3]/2)])]
            wslice = [tf.slice(w, [0,0,0,0], wd[:3]+[int(wd[3]/2)]), tf.slice(w, [0,0,0,int(wd[3]/2)], wd[:3]+[int(wd[3]/2)])]
            
            print([i.get_shape() for i in xslice])
            print([i.get_shape() for i in wslice])
            
            i = tf.concat([tf.nn.conv2d(i,k, [1,s,s,1], padding='SAME') for i,k in zip(xslice, wslice)], 3)
            j = tf.reshape(tf.nn.bias_add(i,b), [-1]+i.get_shape().as_list()[1:])
        
    return tf.nn.relu(j, name=name)


def maxpool2d(name, x, k, s, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,s,s,1], padding=padding, name=name)

def norm(name, x, radius, alpha, beta, bias):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias,name=name)



im1 = format_im(cv2.imread('im1.jpg'))
im2 = format_im(cv2.imread('im2.jpg'))

net_data = np.load(open('bvlc_alexnet.npy', 'rb'), encoding='latin1').item()

iminput = tf.placeholder(tf.float32, [1, 227,227, 3], name='input')

#conv1
conv1W = tf.Variable(net_data['conv1'][0], name='conv1W')
conv1b = tf.Variable(net_data['conv1'][1], name='conv1b')
print(conv1W.get_shape(), conv1b.get_shape())
conv1 = conv2d('conv1', iminput, conv1W, conv1b, 4)
#norm1
norm1 = norm('norm1', conv1, radius=2, alpha=2e-05, beta=0.75, bias=1.0)
#pool1
pool1 = maxpool2d('pool1', norm1, 3,2, 'VALID')
print(pool1.get_shape())

#conv2
conv2W = tf.Variable(net_data['conv2'][0], name='conv2W')
conv2b = tf.Variable(net_data['conv2'][1], name='conv2b')
print(conv2W.get_shape(), conv2b.get_shape())
conv2 = conv2d('conv2', pool1, conv2W, conv2b, 1,2)
#norm2
norm2 = norm('norm2', conv2, radius=2, alpha=2e-05, beta=0.75, bias=1.0)
#pool2
pool2 = maxpool2d('pool2', norm2, 3, 2, 'VALID')

#conv3
conv3W = tf.Variable(net_data['conv3'][0], name='conv3W')
conv3b = tf.Variable(net_data['conv3'][1], name='conv3b')
conv3 = conv2d('conv3', pool2, conv3W, conv3b, 1)
print(conv3, conv3.get_shape())

#conv4
conv4W = tf.Variable(net_data['conv4'][0], name='conv4W')
conv4b = tf.Variable(net_data['conv4'][1], name='conv4b')
conv4 = conv2d('conv4', conv3, conv4W, conv4b, 1, 2)
print(conv4, conv4.get_shape())

#conv5
conv5W = tf.Variable(net_data["conv5"][0], name='conv5W')
conv5b = tf.Variable(net_data["conv5"][1], name='conv5b')
conv5 = conv2d('conv5', conv4, conv5W, conv5b, 1, 2)
print(conv5, conv5.get_shape())
#pool5
pool5 = maxpool2d('pool5', conv5, 3, 2, 'VALID')

#fc6
fc6W = tf.Variable(net_data["fc6"][0], name='fc6W')
fc6b = tf.Variable(net_data["fc6"][1], name='fc6b')
fc6 = tf.nn.relu_layer(tf.reshape(pool5, [-1, int(np.prod(pool5.get_shape()[1:]))]), fc6W, fc6b, name='fc6')

#fc7
fc7W = tf.Variable(net_data["fc7"][0], name='fc7W')
fc7b = tf.Variable(net_data["fc7"][1], name='fc7b')
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b, name='output')

#fc8
fc8W = tf.Variable(net_data["fc8"][0], name='fc8W')
fc8b = tf.Variable(net_data["fc8"][1], name='fc8b')
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b, name='prob')



init = tf.initialize_all_variables()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for i in range(5):
        st = time.process_time()
        
        if i%2==0:
            ex = sess.run(fc7, feed_dict={iminput:im1})
        else:
            ex = sess.run(fc7, feed_dict={iminput:im2})
        print('elapsed:',time.process_time()-st)
        print(ex.shape)
        print(ex[0][:10])
    train_writer = tf.summary.FileWriter('./log2', sess.graph)
    saver.save(sess, './model2/mynet')
    train_writer.close()
