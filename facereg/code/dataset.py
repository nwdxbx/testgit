import os
import re
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import cv2

def data_augment(img, image_size):

    img = tf.image.resize_images(img, [224, 224])
    img = tf.image.resize_image_with_crop_or_pad(img, int(image_size[0]*1.2), int(image_size[1]*1.2))
    
    
    r1 = tf.random_uniform([], -3.1415926/18, 3.1415926/18, dtype=tf.float32)
    img = tf.contrib.image.rotate(img, r1, interpolation='BILINEAR')
    
    
    r2 = tf.random_uniform([], 0.7, 1.1, dtype=tf.float32)
    r3 = tf.random_uniform([], 0.7, 1.1, dtype=tf.float32)
    img = tf.random_crop(img, [tf.cast(image_size[0]*r2, dtype=tf.int32), tf.cast(image_size[1]*r3, dtype=tf.int32), 3])    
    
    
    
    return img


def face_dataset(batch_size, image_size, classes): # h, w
    with tf.variable_scope("data"):
        filename_queue = tf.train.string_input_producer(['train-align.txt'], name='train-align.txt')
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue, name='csv_read')
        record_defaults = [[""], [1]]
        imgname, label = tf.decode_csv(value, record_defaults = record_defaults, field_delim=' ', name='csv_decode')

        label = tf.cast(label, tf.int64, name='cast_label')
        imgcontent = tf.read_file(imgname, name='image_read')
        img = tf.image.decode_png(imgcontent, channels = 3, name='image_decode')
        img = tf.cast(img, tf.float32, name='image_cast')
        
        r = tf.random_uniform([], 0, 1, dtype=tf.float32)
        
        img = tf.cond(r>0.5, lambda :data_augment(img, image_size), lambda : img)
        
        
        
        img = tf.image.resize_images(img, image_size)
        img = tf.image.random_flip_left_right(img)
        
        
        

        img = img/127.5 - 1
        #img = tf.image.random_brightness(img, max_delta = 0.2)
        
        img.set_shape([image_size[0], image_size[1], 3])
        
        [imgs, labels] = tf.train.shuffle_batch([img, label], batch_size=batch_size, 
            num_threads=16, capacity=2048, min_after_dequeue=1024, name='batch')
        
        return [imgs, labels]

def face_dataset_test(batch_size, image_size, classes): # h, w
    with tf.variable_scope("data_test"):
        filename_queue = tf.train.string_input_producer(['test-align.txt'], name='test-align.txt')
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue, name='csv_read')
        record_defaults = [[""], [1]]
        imgname, label = tf.decode_csv(value, record_defaults = record_defaults, field_delim=' ', name='csv_decode')

        label = tf.cast(label, tf.int64, name='cast_label')
        imgcontent = tf.read_file(imgname, name='image_read')
        img = tf.image.decode_png(imgcontent, channels = 3, name='image_decode')
        img = tf.cast(img, tf.float32, name='image_cast')
        
        
        img = tf.image.resize_images(img, image_size)
        img = tf.image.random_flip_left_right(img)
        

        img = img/127.5 - 1
        
        img.set_shape([image_size[0], image_size[1], 3])
        
        [imgs, labels] = tf.train.shuffle_batch([img, label], batch_size=batch_size, 
            num_threads=16, capacity=2048, min_after_dequeue=1024, name='batch')
        
        return [imgs, labels]

if __name__ == '__main__':
    imgs, labels = face_dataset(1, [224, 224], 9131)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        while (True):
            I= sess.run([imgs])
            I = (I[0][0]+1)*127.5
            I = I.astype(np.uint8)
            I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
            cv2.imshow("mx", I)
            cv2.waitKey(0)
            

        coord.request_stop()
        coord.join(threads)
