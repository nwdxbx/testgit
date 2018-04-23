from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
import cv2
import numpy as np
from multiprocessing import Queue   ## xinchen add
#from queue import Queue 
from threading import Thread


class TextDataSet:
  default_param={}
  default_param['filename']=""
  default_param['batch_size']=128
  default_param['thread_num']=4
  default_param['max_record_queue_size']=100000
  default_param['max_image_queue_size']=1024
  default_param['one_record_to_multisamples']=False
  
  
  def __init__(self, common_params):
    #process params
    self.filename = str(common_params['filename'])
    self.batch_size = int(common_params['batch_size'])
    self.thread_num = int(common_params['thread_num'])
    self.max_record_queue_size = int(common_params['max_record_queue_size'])
    self.max_image_queue_size = int(common_params['max_image_queue_size'])
    self.one_record_to_multisamples = bool(common_params['one_record_to_multisamples'])

    #record and image_label queue
    self.record_queue = Queue(maxsize=self.max_record_queue_size)
    self.image_label_queue = Queue(maxsize=self.max_image_queue_size)

    self.record_list = []  

    # filling the record_list
    input_file = open(self.filename, 'r')

    for line in input_file:
      line = line.strip()
      ss = self._parse_line(line)
      self.record_list.append(ss)

    self.record_point = 0
    self.record_number = len(self.record_list)

    self.num_batch_per_epoch = int(self.record_number / self.batch_size)


    self.t_record_producer = Thread(target=self._record_producer)
    self.t_record_producer.setDaemon(True)
    self.t_record_producer.start()

    self.t_record_customer=[]
    for i in range(self.thread_num):
      self.t_record_customer.append(Thread(target=self._record_customer))
      self.t_record_customer[-1].setDaemon(True)
      self.t_record_customer[-1].start() 
      
  
  def _record_producer(self):
    while True:
      if self.record_point % self.record_number == 0:
        random.shuffle(self.record_list)
        self.record_point = 0
      self.record_queue.put(self.record_list[self.record_point])
      self.record_point += 1

 
  def _parse_line(self, line):
    raise NotImplementedError


  def _record_process(self, record):
    raise NotImplementedError
  
  def _record_customer(self):
    while True:
      item = self.record_queue.get()
      out = self._record_process(item)
      #self.image_label_queue.put(out)
      if out is None:
        continue
      if not self.one_record_to_multisamples:
        self.image_label_queue.put(out)
      else:
        for i in out:
          self.image_label_queue.put(i)


  def batch(self):
    batch=[]
    for i in range(self.batch_size):
      single = self.image_label_queue.get()
      batch.append(single)
    batch=self._compose(batch)
    return batch
  
  
  def _compose(self, list_single):
    raise NotImplementedError
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
