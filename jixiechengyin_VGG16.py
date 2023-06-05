import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.preprocessing.image import  ImageDataGenerator
from tensorflow.python.keras import backend
from tensorflow.python.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from pltshow import LossHistory
from confusion_matrix import plot_confusion_matrix
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard

image_size = (512, 512, 3)

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# backend.set_session(tf.Session(config=config))

datagen=ImageDataGenerator(samplewise_center=True)

train_generator=datagen.flow_from_directory(
  r'E:\TS\bishe\new\jixiechengyin_train',
  target_size=(512,512),batch_size=8
)

test_generator=datagen.flow_from_directory(
  r'E:\TS\bishe\new\jixiechengyin_test',
  target_size=(512,512),batch_size=8
)

x_input = layers.Input(shape=image_size, name='input')

weight_path=r"E:\TS\bishe\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
# x = x_input
base_model = vgg16.VGG16(weights=weight_path, include_top=False, pooling='avg')
for layer in base_model.layers:
    layer.trainable = True
x = base_model(x_input)
x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.9)(x)
x = layers.Dense(64,activation="relu")(x)
x = layers.Dense(5, activation='softmax')(x)

model= Model(inputs=x_input,outputs=x)
model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy',metrics = ['accuracy'])
history = LossHistory()
checkpoint = ModelCheckpoint(r"E:\TS\bishe\new\result\jixiechengyin\vgg16jixiechengyin.h5",
                                 monitor="val_acc",
                                 mode='max',
                                 save_weights_only=True,
                                 save_best_only=True,
                                 verbose=1,
                                 period=1)
callback_lists=[history,checkpoint]

model.fit_generator(train_generator,validation_data=test_generator,epochs=300,callbacks=callback_lists,verbose=2)

history.loss_plot('epoch',r'E:/TS/bishe/new/result/jixiechengyin/VGG_jixiechengyin.png',
                  r'E:/TS/bishe/new/result/jixiechengyin/VGG_jixiechengyin.txt')
labels=[
'粉屑','砾屑','其他','砂屑',
'团粒',]
model.load_weights(r"E:\TS\bishe\new\result\jixiechengyin\vgg16jixiechengyin.h5")
y_true,y_pred=[],[]
for i in range(len(test_generator)):
    y_pred.extend(np.argmax(model.predict(test_generator[i][0]),axis=1))
    y_true.extend(np.argmax(test_generator[i][1],axis=1))
print(y_true)
print(y_pred)
plot_confusion_matrix(y_true,y_pred,labels,save_path=r'E:/TS/bishe/new/result/jixiechengyin/VGG_jixiechengyin-confusion_matrix.png')