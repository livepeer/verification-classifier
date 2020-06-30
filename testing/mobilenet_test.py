checkpoint_name = 'mobilenet_v2_1.0_224' #@param
checkpoint = checkpoint_name + '.ckpt'

import tensorflow as tf
from tf.keras.applications import MobileNetV2