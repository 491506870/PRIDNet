from __future__ import division
from scipy.io import loadmat
from scipy.io import savemat
import glob
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tflearn.layers.conv import global_avg_pool
from network import network


val_dir = './Validation/ValidationNoisyBlocksRaw.mat'
checkpoint_dir = './checkpoint/SIDD_Pyramid/model.ckpt'
result_dir = './res/'

mat = loadmat(val_dir)
# print(mat.keys)
val_img = mat['ValidationNoisyBlocksRaw'] #(40, 32, 256, 256)
# val_img = mat['ValidationNoisyBlocksRaw']
# val_img = np.expand_dims(val_img.reshape([1280, 256, 256]), axis=3)
val_img = val_img.reshape([1280, 256, 256])

ps = 256


ouput_blocks = [None] * 40 * 32

sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 1])
# gt_image = tf.placeholder(tf.float32, [None, None, None, 3])

out_image = network(in_image)

saver = tf.train.Saver(max_to_keep=15)
sess.run(tf.global_variables_initializer())

print('loaded ' + checkpoint_dir)
saver.restore(sess, checkpoint_dir)


if not os.path.isdir(result_dir):
    print('-----------------------------no existing path')
    os.makedirs(result_dir)

for i in range(len(val_img)):

    each_block = val_img[i] #(256, 256)
    each_block = np.expand_dims(np.expand_dims(each_block, axis=0), axis=3)
    

    st = time.time()
    output = sess.run(out_image, feed_dict={in_image: each_block})
    output = np.minimum(np.maximum(output, 0), 1)

    
    t_cost = time.time() - st
    ouput_blocks[i] = output
    print(ouput_blocks[i].shape)
    #scipy.misc.toimage(each_block[0,:,:,0] * 255, high=255, low=0, cmin=0, cmax=255).save(
    #            result_dir + '%04d_test_in.jpg' % (i))
    #scipy.misc.toimage(output[0,:,:,0] * 255, high=255, low=0, cmin=0, cmax=255).save(
    #            result_dir + '%04d_test_out.jpg' % (i))
    print('cleaning block %4d' % i)
    print('time_cost:', t_cost)
out_mat = np.squeeze(ouput_blocks)
out_mat = out_mat.reshape([40, 32, 256, 256])

savemat(result_dir + 'ValidationCleanBlocksRaw.mat', {'results': out_mat})
