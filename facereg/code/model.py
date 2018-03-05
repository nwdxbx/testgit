import os
import re
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import cv2
from resnet_v1 import resnet_v1_50, resnet_arg_scope

def face_model(imgs, image_size, classes, is_training=True):
    with slim.arg_scope(resnet_arg_scope()):
        logits, end_points = resnet_v1_50(imgs, classes, is_training=is_training)

    return logits, end_points


