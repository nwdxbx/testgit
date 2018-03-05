import os
import re
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import cv2

from dataset import face_dataset, face_dataset_test
from model import face_model
import random

with open('train-align.txt') as f:
    lines = f.readlines()
random.shuffle(lines)
with open('train-align.txt', 'w') as f:
    for l in lines:
        f.write(l)      
with open('test-align.txt') as f:
    lines = f.readlines()
random.shuffle(lines)
with open('test-align.txt', 'w') as f:
    for l in lines:
        f.write(l)

os.environ['CUDA_VISIBLE_DEVICES']="0"
def center_loss(features, label, alfa, nrof_classes):
    with tf.variable_scope("center_loss"):
        decay_rate = 0.1
        nrof_features = features.get_shape()[1]
        centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
            initializer=tf.constant_initializer(0), trainable=False)
        label = tf.reshape(label, [-1])
        centers_batch = tf.gather(centers, label)
        diff = (1 - alfa) * (centers_batch - features)*decay_rate
        centers = tf.scatter_sub(centers, label, diff)
        loss = tf.reduce_mean(tf.square(features - centers_batch))*decay_rate / 2
        return loss, centers

def get_center_loss(features, labels, alpha, num_classes):
    with tf.variable_scope("center_loss"):
        decay_rate = 0.0002 #0.0005
        len_features = features.get_shape()[1]
        centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
            initializer=tf.constant_initializer(0), trainable=False)
        labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)
        loss_o = tf.nn.l2_loss(features - centers_batch)/2
        loss = loss_o*decay_rate
        diff = (centers_batch - features)*decay_rate
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff
        centers_update_op = tf.scatter_sub(centers, labels, diff)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)
        return loss, centers, centers_update_op, loss_o

image_size=[224, 224]
classes=9131
imgs_p = tf.placeholder(tf.float32, shape=[None]+image_size+[3], name='imgs_placeholder')
labels_p = tf.placeholder(tf.int64, shape=[None], name='labels_placeholder')
imgs, labels = face_dataset(24, image_size, classes)
imgs_test, labels_test = face_dataset_test(24, image_size, classes)
one_hot_labels = tf.one_hot(labels_p, classes, name='one_hot_code')
y_pred, end_points = face_model(imgs_p, image_size, classes)


with tf.variable_scope("training_loss"):
    embeddings = end_points['feat'] #tf.nn.l2_normalize(end_points['feat'], 1, 1e-10, name='embeddings')
    center_loss, _, update_center, center_loss_without_decay = get_center_loss(embeddings, labels_p, 0.95, classes)

    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_labels, y_pred)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    l2loss = tf.losses.get_regularization_loss()

    loss = tf.add_n([l2loss, center_loss, cross_entropy_mean])
    
    
with tf.variable_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), labels_p)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    
#global_step = tf.Variable(0, trainable=False, name='global_step')
global_step = tf.placeholder(dtype=tf.int32, name='global_step')
lr = tf.constant(0.00005)#tf.train.exponential_decay(0.001, global_step=global_step, decay_rate = 0.9, decay_steps=10000, name='lr')


trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

#trainable_var = filter(lambda x: x.name.find(r'resnet_v1_50/logits')!=-1, trainable_var)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#print(update_ops)
with tf.control_dependencies(update_ops):
    opt = tf.train.AdamOptimizer(lr).minimize(loss, var_list=trainable_var)

checkpoint_file = './models/model.ckpt-65000'

trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

#trainable_var = list(filter(lambda x: x.name.find(r'resnet_v1_50/logits')==-1
#                and x.name.find(r'resnet_v1_50/feat')==-1, trainable_var))
#trainable_var = filter(lambda x: x.name.find(r'Adam')==-1, trainable_var)
                
loader = tf.train.Saver(trainable_var, name='loader')
saver = tf.train.Saver(name='saver')


tf.summary.scalar('loss', loss)
tf.summary.scalar('loss-l2', l2loss)
tf.summary.scalar('loss-crossentropy', cross_entropy_mean)
tf.summary.scalar('loss-center', center_loss_without_decay)
tf.summary.scalar('lr', lr)
tf.summary.scalar('accuracy', accuracy)
summaries = tf.summary.merge_all()
summaries_loger_train = tf.summary.FileWriter('log/train')
summaries_loger_test = tf.summary.FileWriter('log/test')
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    loader.restore(sess, checkpoint_file)
    summaries_loger_train.add_graph(sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    tf.get_default_graph().finalize()
    for i in range(257001, 10000000):
        imgs_r, labels_r= sess.run([imgs, labels])
        if i % 10 == 0:
            _, loss_r, cross_entropy_loss_r, center_loss_r, acc_r, lr_r, s_r= sess.run([opt, loss, cross_entropy_mean, center_loss_without_decay, accuracy, lr, summaries],
                                        feed_dict={imgs_p: imgs_r, labels_p: labels_r, global_step:i})
            summaries_loger_train.add_summary(s_r, i)                                
        else:
            _, loss_r, cross_entropy_loss_r, center_loss_r, acc_r, lr_r= sess.run([opt, loss, cross_entropy_mean, center_loss_without_decay, accuracy, lr],
                                        feed_dict={imgs_p: imgs_r, labels_p: labels_r, global_step:i})
        
        print(i, loss_r, cross_entropy_loss_r, center_loss_r, acc_r, lr_r)
        

        if i % 100 == 0:
            imgs_r, labels_r= sess.run([imgs_test, labels_test])
            cross_entropy_loss_r, acc_r, s_r= sess.run([cross_entropy_mean, accuracy, summaries],
                                        feed_dict={imgs_p: imgs_r, labels_p: labels_r, global_step:i})
            summaries_loger_test.add_summary(s_r, i)
            print('test', cross_entropy_loss_r, acc_r)
        if i % 1000 == 0:
            saver.save(sess, 'models/model.ckpt', global_step=i)

    coord.request_stop()
    coord.join(threads)





