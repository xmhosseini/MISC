from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, ZeroPadding2D
from keras.layers import Conv2D, BatchNormalization, DepthwiseConv2D, Lambda, Concatenate
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D, Reshape
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import os
import math
import numpy
import time
from PIL import Image
import numpy as np
from numpy import *
# from math import *
from keras.optimizers import Adam
from keras.layers import AveragePooling2D, Input, Flatten
from keras.models import Model

# from MyConv2_K import my_conv2d
keras.backend.set_image_data_format('channels_last')
data_format = 'channels_last'


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
from keras.utils import multi_gpu_model

from keras.metrics import top_k_categorical_accuracy
import tensorflow as tf



def getCustomTensor(H, W, C, N, F, S):
  CustomTensor = np.full((H, W, C, N), 0)
  for f in range (0, F):
    for w in range (0, W):
      for h in range (0, H):
        CustomTensor [h, w, 0, f*S]= 1
  for c in range (1, C):
    for n in range (0, N):
      for w in range (0, W):
        for h in range (0, H):
          CustomTensor [h, w, c, n] = CustomTensor [h, w, c-1, (n-1)%N]
  return CustomTensor


def getCustomMatrix(C, N, F, S):
  CustomMatrix = np.full((C, N), 0)
  for f in range (0, F):
    CustomMatrix [0, f*S]= 1
  for c in range (1, C):
    for n in range (0, N):
      CustomMatrix [c, n] = CustomMatrix [c-1, (n-1)%N]
  return CustomMatrix


# def preprocess(x):
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    # x /= 255.0
    # x -= 0.5
    # x *= 2.0
    # return x

# def preprocess(x):
    # x -= 128.0
    # x /= 128.0
    # return x

def preprocess(x):
    return x

import keras as J

def relu6(x):
    return tf.floor(1.0*J.activations.relu(x, max_value=6.0))

# def swish(x, dequant_weight=1.0, quant_activation=1.0):
    # return tf.floor(quant_activation*dequant_weight*J.activations.relu(x, max_value=tf.floor(6.0/dequant_weight) ))

def swish(x, dequant_weight=1.0, quant_activation=1.0):
    return J.activations.relu(tf.round(x*quant_activation*dequant_weight), max_value=255 )



from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({'swish': Activation(swish)})
from keras.layers import Layer
class Swish(Layer):
    def __init__(self, dequant_weight=1.0,quant_activation=1.0, trainable=False, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True
        self.dequant_weight = dequant_weight
        self.quant_activation = quant_activation
        self.trainable = trainable
    def build(self, input_shape):
        self.dequant_weight_factor = K.variable(self.dequant_weight,
                                      dtype=K.floatx(),
                                      name='dequant_weight_factor')
        self.quant_activation_factor = K.variable(self.quant_activation,
                                      dtype=K.floatx(),
                                      name='quant_activation_factor')
        if self.trainable:
            self._trainable_weights.append(self.dequant_weight_factor)
            self._trainable_weights.append(self.quant_activation_factor)
        super(Swish, self).build(input_shape)
    def call(self, inputs, mask=None):
        return swish(inputs, self.dequant_weight_factor, self.quant_activation_factor)
    def get_config(self):
        config = {'dequant_weight': self.get_weights()[0] if self.trainable else self.dequant_weight,
                  'quant_activation': self.get_weights()[0] if self.trainable else self.quant_activation,
                  'trainable': self.trainable}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def compute_output_shape(self, input_shape):
        return input_shape



def swish2(x, dequant_weight=1.0, quant_activation=1.0, offset_activation=0.0):
    return tf.round(quant_activation*dequant_weight*J.activations.linear(x)+offset_activation)

# def swish2(x, dequant_weight=1.0, quant_activation=1.0, offset_activation=0.0):
    # return tf.floor((dequant_weight*J.activations.linear(x)))

from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({'swish2': Activation(swish2)})
from keras.layers import Layer
class Swish2(Layer):
    def __init__(self, dequant_weight=1.0,quant_activation=1.0,offset_activation=0.0, trainable=False, **kwargs):
        super(Swish2, self).__init__(**kwargs)
        self.supports_masking = True
        self.dequant_weight = dequant_weight
        self.quant_activation = quant_activation
        self.offset_activation = offset_activation
        self.trainable = trainable
    def build(self, input_shape):
        self.dequant_weight_factor = K.variable(self.dequant_weight,
                                      dtype=K.floatx(),
                                      name='dequant_weight_factor')
        self.quant_activation_factor = K.variable(self.quant_activation,
                                      dtype=K.floatx(),
                                      name='quant_activation_factor')
        self.offset_activation_factor = K.variable(self.offset_activation,
                                      dtype=K.floatx(),
                                      name='offset_activation_factor')
        if self.trainable:
            self._trainable_weights.append(self.dequant_weight_factor)
            self._trainable_weights.append(self.quant_activation_factor)
            self._trainable_weights.append(self.offset_activation_factor)
        super(Swish2, self).build(input_shape)
    def call(self, inputs, mask=None):
        return swish2(inputs, self.dequant_weight_factor, self.quant_activation_factor, self.offset_activation_factor)
    def get_config(self):
        config = {'dequant_weight': self.get_weights()[0] if self.trainable else self.dequant_weight,
                  'quant_activation': self.get_weights()[0] if self.trainable else self.quant_activation,
                  'offset_activation': self.get_weights()[0] if self.trainable else self.offset_activation,
                  'trainable': self.trainable}
        base_config = super(Swish2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def compute_output_shape(self, input_shape):
        return input_shape



def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 150*180//200:
        lr *= 0.5e-3
    elif epoch > 150*160//200:
        lr *= 1e-3
    elif epoch > 150*120//200:
        lr *= 1e-2
    elif epoch > 150*80//200:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr



def Mobilenet():
    inputs = Input(shape=(224,224,3))
    x = BatchNormalization()(inputs)
    x = Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.00009127283556153998 , trainable=True)(x)
    x = DepthwiseConv2D(kernel_size=(3,3), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.0053282855078577995, trainable=True)(x)
    x = Conv2D(filters=32, kernel_size=(1,1), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.0008068716851994395, trainable=True)(x)
    x = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.001169452560134232, trainable=True)(x)
    x = Conv2D(filters=64, kernel_size=(1,1), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.0010654556099325418, trainable=True)(x)
    x = DepthwiseConv2D(kernel_size=(3,3), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.0024632480926811695, trainable=True)(x)
    x = Conv2D(filters=64, kernel_size=(1,1), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.00038408805266954005, trainable=True)(x)
    x = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.0003968162345699966, trainable=True)(x)
    x = Conv2D(filters=128, kernel_size=(1,1), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.00023213528038468212, trainable=True)(x)
    x = DepthwiseConv2D(kernel_size=(3,3), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.0013331472873687744, trainable=True)(x)
    x = Conv2D(filters=128, kernel_size=(1,1), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.00025847030337899923, trainable=True)(x)
    x = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.00032361180637963116, trainable=True)(x)
    x = Conv2D(filters=256, kernel_size=(1,1), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.00020607728220056742, trainable=True)(x)
    x = DepthwiseConv2D(kernel_size=(3,3), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.0008712161798030138, trainable=True)(x)
    x = Conv2D(filters=256, kernel_size=(1,1), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.00016649124154355377, trainable=True)(x)
    x = DepthwiseConv2D(kernel_size=(3,3), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.0007646824233233929, trainable=True)(x)
    x = Conv2D(filters=256, kernel_size=(1,1), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.00016895134467631578, trainable=True)(x)
    x = DepthwiseConv2D(kernel_size=(3,3), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.0005055166548117995, trainable=True)(x)
    x = Conv2D(filters=256, kernel_size=(1,1), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.00018946158525068313, trainable=True)(x)
    x = DepthwiseConv2D(kernel_size=(3,3), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.0005571268266066909, trainable=True)(x)
    x = Conv2D(filters=256, kernel_size=(1,1), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.00017237465362995863, trainable=True)(x)
    x = DepthwiseConv2D(kernel_size=(3,3), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.0005147207411937416, trainable=True)(x)
    x = Conv2D(filters=256, kernel_size=(1,1), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.0002089811023324728, trainable=True)(x)
    x = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.00017645212938077748, trainable=True)(x)
    x = Conv2D(filters=512, kernel_size=(1,1), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.00028664665296673775, trainable=True)(x)
    x = DepthwiseConv2D(kernel_size=(3,3), padding='same')(x)
    x = Swish(quant_activation=1.0/0.023528477177023888, dequant_weight=0.0016346947522833943, trainable=True)(x)
    # x = Conv2D(filters=512, kernel_size=(1,1), padding='same')(x)
    # x = Swish(quant_activation=1.0, dequant_weight=0.0007260440033860505, trainable=True)(x)
    # x = GlobalAveragePooling2D()(x)
    # # x = Swish2(quant_activation=1.0, dequant_weight=1.0)(x)
    # x = Reshape((1,1,512))(x)
    # x = Conv2D(filters=1000, kernel_size=(1,1), padding='same')(x)
    # # x = Swish2(quant_activation=1.0/0.15294449031352997, dequant_weight= 0.0001272138033527881, offset_activation = 76.0)(x)
    # x = Flatten()(x)
    # outputs = Activation('softmax')(x)
    outputs = x
    model = Model(inputs=inputs, outputs=outputs)
    return model


def top_5_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


# from keras.utils.conv_utils import convert_kernel
    # bf=bf[::-1]

import numpy as np
from keras import backend as KK
from keras.utils import to_categorical
from keras.applications.vgg19 import VGG19, preprocess_input


def top_k_accuracy(y_true, y_pred, k=1):
    argsorted_y = np.argsort(y_pred)[:,-k:]
    return np.any(argsorted_y.T == y_true.argmax(axis=1), axis=0).mean()

if __name__ == '__main__':
    batch_size = 128
    inital_epoch = 0
    ds = '/rhome/morteza/ImageNet/ILSVRC2012_RGB_cropped/'
    model = Mobilenet()

    gamma = 1.
    dequant_weight = 0.
    mean=128
    # mean=0
    variance = 128*128
    variance = 1*1
    epsilon = 0.
    l = [(np.zeros(3)+gamma).astype(float32),(np.zeros(3)+dequant_weight).astype(float32),(np.zeros(3)+mean).astype(float32),(np.zeros(3)+variance).astype(float32)]
    model.layers[1].set_weights(l)
    model.layers[1].epsilon=0.0
    model.layers[1].trainable=False

    q=(np.load('w01.npy')).astype('float32')
    wf=(q - 117) 
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b01.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+1].set_weights(l)

    q=(np.load('w02.npy')).astype('float32')
    wf=(q - 135) 
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b02.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+3].set_weights(l)

    q=(np.load('w03.npy')).astype('float32')
    wf=(q - 155) 
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b03.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+5].set_weights(l)

    q=(np.load('w04.npy')).astype('float32')
    wf=(q - 179)
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b04.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+7].set_weights(l)

    q=(np.load('w05.npy')).astype('float32')
    wf=(q - 77) 
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b05.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+9].set_weights(l)

    q=(np.load('w06.npy')).astype('float32')
    wf=(q - 87) 
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b06.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+11].set_weights(l)

    q=(np.load('w07.npy')).astype('float32')
    wf=(q - 112) 
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b07.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+13].set_weights(l)

    q=(np.load('w08.npy')).astype('float32')
    wf=(q - 111)
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b08.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+15].set_weights(l)

    q=(np.load('w09.npy')).astype('float32')
    wf=(q - 101) 
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b09.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+17].set_weights(l)

    q=(np.load('w10.npy')).astype('float32')
    wf= (q - 106)
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b10.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+19].set_weights(l)

    q=(np.load('w11.npy')).astype('float32')
    wf=(q - 124)
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b11.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+21].set_weights(l)

    q=(np.load('w12.npy')).astype('float32')
    wf=(q - 118)
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b12.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+23].set_weights(l)

    q=(np.load('w13.npy')).astype('float32')
    wf=(q - 147)
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b13.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+25].set_weights(l)

    q=(np.load('w14.npy')).astype('float32')
    wf= (q - 133)
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b14.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+27].set_weights(l)

    q=(np.load('w15.npy')).astype('float32')
    wf= (q - 139) 
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b15.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+29].set_weights(l)

    q=(np.load('w16.npy')).astype('float32')
    wf=(q - 121) 
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b16.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+31].set_weights(l)

    q=(np.load('w17.npy')).astype('float32')
    wf=(q - 115)
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b17.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+33].set_weights(l)

    q=(np.load('w18.npy')).astype('float32')
    wf= (q - 144) 
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b18.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+35].set_weights(l)

    q=(np.load('w19.npy')).astype('float32')
    wf=(q - 136)
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b19.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+37].set_weights(l)

    q=(np.load('w20.npy')).astype('float32')
    wf=(q - 157) 
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b20.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+39].set_weights(l)

    q=(np.load('w21.npy')).astype('float32')
    wf= (q - 123)
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b21.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+41].set_weights(l)

    q=(np.load('w22.npy')).astype('float32')
    wf=(q - 129)
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b22.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+43].set_weights(l)

    q=(np.load('w23.npy')).astype('float32')
    wf=(q - 156) 
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b23.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+45].set_weights(l)

    q=(np.load('w24.npy')).astype('float32')
    wf=(q - 109)
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b24.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+47].set_weights(l)

    q=(np.load('w25.npy')).astype('float32')
    wf=(q - 90)
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b25.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+49].set_weights(l)

    q=(np.load('w26.npy')).astype('float32')
    wf=(q - 207)
    wf=np.rollaxis(wf,0,4)
    q=(np.load('b26.npy')).astype('float32')
    bf=q
    l=[]
    l.append(wf)
    l.append(bf)
    model.layers[1+51].set_weights(l)

    # q=(np.load('w27.npy')).astype('float32')
    # wf=(q - 94)
    # wf=np.rollaxis(wf,0,4)
    # q=(np.load('b27.npy')).astype('float32')
    # bf=q
    # l=[]
    # l.append(wf)
    # l.append(bf)
    # model.layers[1+53].set_weights(l)

    # q=(np.load('w29.npy')).astype('float32')
    # wf= 0.005406801588833332 *(q - 89)
    # wf=wf[1:,:,:,:]
    # wf=np.rollaxis(wf,0,4)
    # q=(np.load('b29.npy')).astype('float32')
    # bf= 0.0001272138033527881*q
    # bf=bf[1:]
    # l=[]
    # l.append(wf)
    # l.append(bf)
    # model.layers[1+57+1-1].set_weights(l)


    model = multi_gpu_model(model, gpus=2)

    # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0)), metrics=['accuracy', top_5_acc])
    # print(model.summary())
    # # model.load_weights('%s.hdf5' % model.name, by_name=True)
    # checkpoint = ModelCheckpoint(filepath='%s.hdf5' % model.name, verbose=1,
                                 # save_best_only=True, monitor='val_acc', mode='max')

    # lr_scheduler = LearningRateScheduler(lr_schedule)
    # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               # cooldown=0,
                               # patience=5,
                               # min_lr=0.5e-6)
    # callbacks = [checkpoint, lr_reducer, lr_scheduler]



    # print('Using real-time data augmentation.')

    datagen = ImageDataGenerator()
    # datagen = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
    # datagen = ImageDataGenerator(preprocessing_function=preprocess,
                                       # zoom_range=0.1,
                                       # width_shift_range=0.05,
                                       # height_shift_range=0.05,
                                       # horizontal_flip=True)


    # train_generator = datagen.flow_from_directory(
            # '%s/train/' % ds,
            # target_size=(224, 224),
            # batch_size=batch_size)

    # count_samples = datagen.flow_from_directory(
            # '%s/train/' % ds,
            # target_size=(224, 224),
            # shuffle = False,
            # batch_size=batch_size)
    # test_generator = datagen.flow_from_directory(
            # '%s/train/' % ds,
            # target_size=(224, 224),
            # shuffle = False,
            # batch_size=count_samples.samples//4)
    # x_train, y_train1 = test_generator.next()
    # del x_train
    # x_train, y_train2 = test_generator.next()
    # del x_train
    # x_train, y_train3 = test_generator.next()
    # del x_train
    # x_train, y_train4 = test_generator.next()
    # del x_train
    # x_train, y_train5 = test_generator.next()
    # del x_train
    # y_train1=np.append(y_train1,y_train2,axis=0)
    # y_train1=np.append(y_train1,y_train3,axis=0)
    # y_train1=np.append(y_train1,y_train4,axis=0)
    # y_train1=np.append(y_train1,y_train5,axis=0)
    # print(y_train1.shape)
    # np.save('y_train_augm.npy',y_train1)


    test_generator = datagen.flow_from_directory(
            '%s/train/' % ds,
            target_size=(224, 224),
            shuffle = False,
            batch_size=batch_size)
    f_train = model.predict_generator(
            test_generator,
            steps = test_generator.samples // batch_size,
            verbose=1,
            max_queue_size=98,
            workers=20*50)
    np.save('f_train_augm.npy',f_train)


    # model.fit_generator(
            # train_generator,
            # steps_per_epoch=train_generator.samples // batch_size,
            # epochs=150,
            # verbose=1,
            # max_queue_size=98,
            # workers=20*50,
            # initial_epoch=inital_epoch,
            # use_multiprocessing=False,
            # validation_data=test_generator,
            # validation_steps=test_generator.samples // batch_size,
            # callbacks=callbacks)

