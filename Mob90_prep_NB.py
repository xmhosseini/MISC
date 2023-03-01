from __future__ import print_function
import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, ZeroPadding2D
from tensorflow.keras.layers import Conv2D, BatchNormalization, DepthwiseConv2D, Lambda, Concatenate
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D, Reshape
from keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import os
import math
import numpy
import time
from PIL import Image
import numpy as np
from numpy import *

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.models import Model

from MyConv2_K import my_conv2d
keras.backend.set_image_data_format('channels_last')
data_format = 'channels_last'


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
from tensorflow.keras.utils import multi_gpu_model

from tensorflow.keras.metrics import top_k_categorical_accuracy
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

import keras as K
def relu6(x):
    """Relu 6
    """
    return K.activations.relu(x, max_value=6.0)


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
    if epoch > 40*180//200:
        lr *= 0.5e-3
    elif epoch > 40*160//200:
        lr *= 1e-3
    elif epoch > 40*120//200:
        lr *= 1e-2
    elif epoch > 40*80//200:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr



def Mobilenet_bottom():
    inputs = Input(shape=(7,7,512))
    # x = Conv2D(filters=512, kernel_size=(1,1), padding='same')(inputs)
    # x = Activation(relu6)(x)
    # x = GlobalAveragePooling2D()(x)
    # x = Reshape((1,1,512))(x)
    # x = Conv2D(filters=1000, kernel_size=(1,1), padding='same')(x)
    # x = Flatten()(x)
    # outputs = Activation('softmax')(x)

    x = my_conv2d(inputs=inputs, my_filter = getCustomTensor(1, 1, 512, 512, 4, 1), filters=512, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 512, 512, 4, 4), filters=512, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 512, 512, 4, 16), filters=512, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 512, 512, 4, 64), filters=512, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 512, 512, 4, 128), filters=512, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 512, 512, 4, 1), filters=512, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 512, 512, 4, 4), filters=512, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 512, 512, 4, 16), filters=512, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 512, 512, 4, 64), filters=512, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 512, 512, 4, 128), filters=512, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 512, 512, 4, 1), filters=512, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 512, 512, 4, 4), filters=512, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 512, 512, 4, 16), filters=512, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 512, 512, 4, 64), filters=512, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 512, 512, 4, 128), filters=512, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')

    x = Activation(relu6)(x)
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,512))(x)
    # x = Dropout(rate = 0.001)(x)
# smh: this rate seems to be degrated to 0.001 through the course of epochs
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 512, 1000, 10, 1), filters=1000, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 1000, 1000, 10, 10), filters=1000, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 1000, 1000, 10, 100), filters=1000, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 1000, 1000, 10, 1), filters=1000, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 1000, 1000, 10, 10), filters=1000, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 1000, 1000, 10, 100), filters=1000, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 1000, 1000, 10, 1), filters=1000, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 1000, 1000, 10, 10), filters=1000, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')
    x = my_conv2d(inputs=x, my_filter = getCustomTensor(1, 1, 1000, 1000, 10, 100), filters=1000, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'glorot_uniform')

    x = Flatten()(x)
    outputs = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def top_5_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


# from keras.utils.conv_utils import convert_kernel
    # bf=bf[::-1]

if __name__ == '__main__':
    batch_size = 128
    inital_epoch = 0
    ds = '/rhome/morteza/ImageNet/ILSVRC2012_RGB_TOP_FMAP_0.5/'
    model = Mobilenet_bottom()

    model = multi_gpu_model(model, gpus=2)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0)), metrics=['accuracy', top_5_acc])
    print(model.summary())
    # model.load_weights('%s.hdf5' % model.name, by_name=True)
    checkpoint = ModelCheckpoint(filepath='%s.hdf5' % 'model80', verbose=1,
                                 save_best_only=True, monitor='val_acc', mode='max')

    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    x_train = np.load('%s/' % ds+'f_train_orig.npy')
    y_train = np.load('%s/' % ds+'y_train.npy')
    x_test = np.load('%s/' % ds+'f_val.npy')
    y_test = np.load('%s/' % ds+'y_val.npy')
# ValueError: Input arrays should have the same number of samples as target arrays. Found 1281152 input samples and 1281167 target samples.
# smh: 1281152=128*(1281167//128)
    y_train=y_train[0:1281152,:]

# ValueError: Input arrays should have the same number of samples as target arrays. Found 49920 input samples and 50000 target samples.
    y_test=y_test[0:49920,:]
    x_train /= 127.5-1.
    x_test /= 127.5-1.


    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=40,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)



    # train_generator = datagen.flow_from_directory(
            # '%s/train/' % ds,
            # target_size=(224, 224),
            # batch_size=batch_size)

    # test_generator = datagen.flow_from_directory(
            # '%s/val/' % ds,
            # target_size=(224, 224),
            # batch_size=batch_size)

    # model.fit_generator(
            # train_generator,
            # steps_per_epoch=train_generator.samples // batch_size,
            # epochs=20,
            # verbose=1,
            # max_queue_size=98,
            # workers=20*50,
            # initial_epoch=inital_epoch,
            # use_multiprocessing=False,
            # validation_data=test_generator,
            # validation_steps=test_generator.samples // batch_size,
            # callbacks=callbacks)

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
