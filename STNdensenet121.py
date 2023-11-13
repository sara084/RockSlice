#-*- coding:utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications import densenet
from tensorflow.python.keras.preprocessing.image import  ImageDataGenerator
from tensorflow.python.keras import backend
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from pltshow import LossHistory
from confusion_matrix import plot_confusion_matrix
from utils import initial_weights_without_scale
from stn_layers3 import BilinearInterpolation
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard

image_size = (512, 512, 3) #Resize image size


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# backend.set_session(tf.Session(config=config))

datagen=ImageDataGenerator(samplewise_center=True) #init dataloader

train_generator=datagen.flow_from_directory(    #load train dataset
    '/home/lht-lht/RockSlice/hunruwu_train',
    target_size=(512,512),batch_size=4, save_format='jpg'
)

test_generator=datagen.flow_from_directory(  #load test dataset
  '/home/lht-lht/RockSlice/hunruwu_test',
  target_size=(512,512),batch_size=4, save_format='jpg'
)
x_input = layers.Input(shape=image_size, name='input')

weight_path="/home/lht-lht/RockSlice/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5" #load weight
# x = x_input 

scales = [1.0, 0.8]

x_input = layers.Input(shape=image_size, name='input')


def stn_conv(input_tensor): # model code for stn
    xloc = layers.Conv2D(16, (7, 7), activation='relu')(input_tensor)
    xloc = layers.MaxPooling2D((2, 2), strides=(2, 2))(xloc) 

    xloc = layers.Conv2D(32, (5, 5), activation='relu')(xloc)
    xloc = layers.MaxPooling2D((2, 2), strides=(2, 2))(xloc)

    xloc = layers.Conv2D(64, (3, 3), activation='relu')(xloc)
    xloc = layers.MaxPooling2D((2, 2), strides=(2, 2))(xloc)

    xloc = layers.GlobalAveragePooling2D()(xloc)

    return xloc


def stn_fc(input_tensor, input_xloc, scale):
    xloc = layers.BatchNormalization(name='stn_bn_'+str(scale))(input_xloc)
    xloc = layers.Dropout(0.5, name='stn_drop_'+str(scale))(xloc)

    weights = initial_weights_without_scale(64) #init weight
    xloc = layers.Dense(2, weights=weights, name='theta'+str(scale), activation='tanh')(xloc)
    # xloc = layers.Dense(2, weights=weights, name='fc_'+str(scale))(xloc)

    def loc_scale(trans_para):
        xscale = trans_para[:, 0:1]
        trans_x = backend.maximum(backend.minimum(trans_para[:, 0:1], 1 - scale), scale - 1)
        trans_y = backend.maximum(backend.minimum(trans_para[:, 1:2], 1 - scale), scale - 1)
        return backend.concatenate([backend.ones_like(xscale) * scale,
                                    backend.zeros_like(xscale), trans_x,
                                    backend.zeros_like(xscale),
                                    backend.ones_like(xscale) * scale, trans_y
                                    ], axis=1)

    xloc = layers.Lambda(loc_scale, name='fc_theta_scale' + str(scale))(xloc)

    x_trans = BilinearInterpolation(image_size, name='bi_inter'+str(scale))([input_tensor, xloc])

    return x_trans


x_stn_conv = stn_conv(x_input)
x = layers.Concatenate(axis=0)([x_input] +
                               [stn_fc(x_input, x_stn_conv, scales[i]) for i in range(1, len(scales))])

# x = x_input
base_model = densenet.DenseNet121(weights=weight_path, include_top=False, pooling='avg')
for layer in base_model.layers:
    layer.trainable = True

x = base_model(x)
x = layers.Lambda(lambda input_tensor: tf.split(input_tensor, len(scales)))(x)
x_scales = []
for idx, sx in enumerate(x):
    sx = layers.BatchNormalization(name='btn_' + str(idx))(sx)
    x_scales.append(sx)

x = layers.Concatenate(axis=0)(x_scales)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(7, activation='softmax')(x)

x = layers.Lambda(lambda input_tensor: tf.split(input_tensor, len(scales)))(x)
x = layers.Average()(x)

model= Model(inputs=x_input,outputs=x)
model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy',metrics = ['accuracy'])
history = LossHistory()
checkpoint = ModelCheckpoint("/home/lht-lht/RockSlice/result/hunruwu/STNdensenet121hunruwu_weights.h5",
                                 monitor="val_acc",
                                 mode='max',
                                 save_weights_only=True,
                                 save_best_only=True,
                                 verbose=1,
                                 period=1)
callback_lists=[history,checkpoint]
model.fit(train_generator,validation_data=test_generator,epochs=300,callbacks=callback_lists,verbose=2)

history.loss_plot('epoch', '/home/lht-lht/RockSlice/result/hunruwu/STNdensenet121_hunruwu.png',
                           '/home/lht-lht/RockSlice/result/hunruwu/STNdensenet121_hunruwu.txt')
#labels=['粉砂质,'钙质','硅质','含粉砂质','其他', '炭质', '铁质']
labels=["fenshazhi","gaizhi","guizhi","hanfenshazhi","qita","tanzhi","tiezhi"] #rock label
model.load_weights('\home\lht-lht\RockSlice\result\hunruwu\STNdensenet121hunruwu_weights.h5') #load weight
y_true,y_pred=[],[]
for i in range(len(test_generator)):
    y_pred.extend(np.argmax(model.predict(test_generator[i][0]),axis=1))
    y_true.extend(np.argmax(test_generator[i][1],axis=1))
print(y_true) 
print(y_pred)
plot_confusion_matrix(y_true,y_pred,labels,save_path='/home/lht-lht/RockSlice/result/hunruwu/STNdensenet121_hunruwu-confusion_matrix.png') save eval result
