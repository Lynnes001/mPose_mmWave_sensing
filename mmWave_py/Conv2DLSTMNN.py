# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:38:14 2020

@author: Song
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
# from tensorflow.compat.v1 import keras

# from keras.metrics import categorical_crossentropy
# from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing import image
# from keras.models import Model
# from keras.applications import imagenet_utils
# from keras.applications.mobilenet import preprocess_input

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from keras import backend as K
from keras.models import Model

from keras.layers import MaxPooling2D, Reshape, Dropout, ConvLSTM2D, Conv2D
from keras.layers import InputLayer, TimeDistributed, concatenate,Flatten, Input, LeakyReLU

from keras.losses import CategoricalCrossentropy, huber_loss

# from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam, SGD
# tf.compat.v2.keras.layers.Reshape
from keras.callbacks import LearningRateScheduler
import math,random

import tensorflow as tf
from keras.backend import set_session
from IPython.display import clear_output
import mdn

from utility import checkDataTrainPhase
from utility_tf import PlotLosses, acc_cust, joint_error_loss

# updatable plot
# a minimal example (sort of)

def fnloss(y_true, y_pred):
    
    y_true_t = tf.reshape(y_true[0], [1, -1, 3])
    y_pred_t = tf.reshape(y_pred[0], [1, -1, 3])
    
    # y_pred_t = tf.cast(y_pred_t, tf.float32)
    # y_true_t = tf.cast(y_true_t, tf.float32)
   

    # weights =  [1, 1, 1, 1, 
    #             1, 5, 50, 100, 
    #             1, 5, 50, 100, 
    #             1, 5, 50, 100,  
    #             1, 5, 50, 100, 
    #             1, 50, 50, 50, 50]
    # # print(y_pred_t)

    joints_err = joint_error_loss(y_pred_t, y_true_t, axis=2);
    # weighted_err = joints_err * weights
    # joints_error_avg = K.sum(weighted_err)/sum(weights)
    joints_error_avg = K.mean(joints_err)
    
    # return huber_loss(y_true, y_pred)
    
    azimuth_y_pred = y_pred_t[:, :, 0:2]
    azimuth_y_true = y_true_t[:, :, 0:2]
    azimuth_joints_error_avg = K.sum(joint_error_loss(azimuth_y_pred, azimuth_y_true, axis=2))/25;
    
    elevation_y_pred = y_pred_t[:, :, 1:3]
    elevation_y_true = y_true_t[:, :, 1:3]
    elevation_joints_error_avg = K.sum(joint_error_loss(elevation_y_pred, elevation_y_true, axis=2))/25;
    
    
    a = 0.5;
    b = 0.5;
    
    # return joints_error_avg + a * azimuth_joints_error_avg + b * elevation_joints_error_avg
    # return joints_error_avg
    return K.mean(joints_err)
    # return mdn.get_mixture_loss_func(42, 20)



def getModel():
         
    # mobilenet=MobileNetV2(input_shape=(46, 46, 3 ), weights=None, include_top=False, pooling=None) #imports the mobilenet model and discards the last 1000 neuron layer.
    # mobilenet=ResNet50(input_shape=(46, 46, 3 ), weights=None, include_top=False, pooling=None) #imports the mobilenet model and discards the last 1000 neuron layer.
    # Freeze the layers
    
    heatmaps = Input(shape = (4, 141, 141, 1))
    
    # conv1 = TimeDistributed(Conv2D(8, 3, activation='relu', padding='same'))(heatmaps)
    # # pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    
    # conv2 = TimeDistributed(Conv2D(16, 3, activation='relu', padding='same'))(conv1)
    # pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
    
    # conv3 = TimeDistributed(Conv2D(32, 3, activation='relu', padding='same'))(conv2)
    # pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)
    
    convlstm1 = ConvLSTM2D(
        32,
        kernel_size = 3,
        strides=3,
        padding="same",
        return_sequences=True,
        dropout=0.0
    )(heatmaps)
    
    convlstm2 = ConvLSTM2D(
        64,
        kernel_size = 2,
        strides=2,
        padding="same",
        return_sequences=True,
        dropout=0.0
    )(convlstm1)
        
    convlstm3 = ConvLSTM2D(
        128,
        kernel_size = 1,
        strides=1,
        padding="same",
        return_sequences=True,
        dropout=0.0
    )(convlstm2)
        
    convlstm4 = ConvLSTM2D(
        256,
        kernel_size = 1,
        strides=1,
        padding="same",
        return_sequences=False,
        dropout=0.0
    )(convlstm3)
    
    # features = Reshape((-1))(convlstm4)
    features = Flatten()(convlstm4)

    
    dense_opt1 = Dense(units=(128))(features)
    # dense_opt1 = Dropout(0.5)(dense_opt1)
    dense_opt1 = LeakyReLU(alpha=0.1)(dense_opt1)

    dense_opt2 = Dense(units=(64))(dense_opt1)
    dense_opt2 = LeakyReLU(alpha=0.1)(dense_opt2)

    posture_pred = Dense(units=(42), name='posture_opt')(dense_opt2)
    # posture_pred = mdn.MDN(42, 20, name='posture_opt')(posture_pred)

    
    model = Model(
        inputs=heatmaps,
        outputs=[posture_pred]
    )

    return model
                


model = getModel()
# opt = Adam(learning_rate=0.001)
opt = Adam()

# 
model.compile(
    # loss=mdn.get_mixture_loss_func(42, 20), 
    # loss='huber_loss', 
    loss=fnloss, 
    optimizer=opt,
    metrics=[acc_cust])

len(model.trainable_weights)
keras.utils.plot_model(model, "model.png", show_shapes=True)


model.summary()
plot_model(model, to_file='model.png')


#%% prepare dataset and label
print("\n\n ============= Loading traing data and label ============= \n")
import pdb 
from sklearn.model_selection import train_test_split
import glob
import numpy as np
# folder1 =  "./data_0523_song_apt/data_train/*"
folder1 =  "./data_0604_song_lab/data_train/*"
# folder2 =  "./data_0612_xin_lab_1/data_train/*"
# files1 = glob.glob(folder1)
files2 = glob.glob(folder1)

# new
# JointType_Head = 3
# JointType_ShoulderLeft = 4
# JointType_ElbowLeft = 5
# JointType_WristLeft = 6
# JointType_ShoulderRight = 8
# JointType_ElbowRight = 9
# JointType_WristRight = 10
# JointType_HipLeft = 12
# JointType_KneeLeft = 13
# JointType_AnkleLeft = 14
# JointType_HipRight = 16
# JointType_KneeRight = 17
# JointType_AnkleRight = 18
# JointType_SpineShoulder = 20


def stack10to5(data):
    data_a = data[:, 0::2, :, :, :]
    data_b = data[:, 1::2, :, :, :]
    
    return np.add( data_a * 0.5, data_b * 0.5 )

def getDataAll(files, domain, simplifyList = []):
    seed = 12
    jointLen = 25
    
    if len(simplifyList)>0:
        jointLen = len(simplifyList)
    
    data = np.empty([1, 4, 141, 141],dtype='float32')
    label = np.empty([1, 25, 3])
    data_val = np.empty([1, 4, 141, 141],dtype='float32')
    label_val = np.empty([1, 25, 3])
    
    for i, file in enumerate(files):
        # if ('side' in file or 'wave' in file) and ('45' not in file):
            # if 'side_left' in file :
            # if i == 0:
        
        data_zip = np.load(file)
        dataSub = data_zip['data']
        labelSub = data_zip['label']
        print(str(labelSub.shape[0]), ' ', file)

        # selectIndex = np.random.choice(labelSub.shape[0], 49, replace=False)
        
        # print('select 49 from '+ str(labelSub.shape[0]))
        
        # dataSub = dataSub[selectIndex]
        # labelSub = labelSub[selectIndex]
        
        X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(dataSub, labelSub, test_size=0.2, random_state=seed)

        data = np.concatenate((data, X_train_sub), axis=0)
        label = np.concatenate((label, y_train_sub), axis=0)
        data_val = np.concatenate((data_val, X_test_sub), axis=0)
        label_val = np.concatenate((label_val, y_test_sub), axis=0)
        

    if len(simplifyList) > 0:
        label = label[:, simplifyList, :]
        label_val = label_val[:, simplifyList, :]

        
        # print(str(i), label.shape)
        # break
    
    # pdb.set_trace()

    # data = stack10to5(data)
    # data_val = stack10to5(data_val)
    # data = np.einsum('abcde->abdec', np.array(c, dtype='float32'), dtype='float32')
    data = data.reshape((-1, 4, 141, 141, 1))
    label = np.array(label).reshape((-1, jointLen*3))
    # data_val = np.einsum('abcde->abdec', np.array(data_val, dtype='float32'), dtype='float32')
    data_val = data_val.reshape((-1, 4, 141, 141, 1))
    label_val = np.array(label_val).reshape((-1, jointLen*3))
    
    # Check data
    (data_val, label_val) = checkDataTrainPhase(data_val, label_val)
    (data, label) = checkDataTrainPhase(data, label)
    
    domain_label = np.zeros((label.shape[0], 2))
    domain_label[:,domain]+=1
    
    domain_label_val = np.zeros((label_val.shape[0], 2))
    domain_label_val[:,domain]+=1
    
    return data, label, domain_label, data_val, label_val, domain_label_val 


simplifyList = [3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20];
    
# data1, label1, domain_label1, data_val1, label_val1, domain_label_val1 = getDataAll(files1, 0)
data2, label2, domain_label2, data_val2, label_val2, domain_label_val2 = getDataAll(files2, 1, simplifyList)

# data = np.vstack((data1, data2))
# label = np.vstack((label1, label2))
# data_val = np.vstack((data_val1, data_val2))
# label_val = np.vstack((label_val1, label_val2))
# domain_label = np.vstack((domain_label1, domain_label2))
# domain_label_val = np.vstack((domain_label_val1, domain_label_val2))


# shuffle dataset

# np.random.seed(200)
# np.random.shuffle(data) 
# np.random.seed(200)
# np.random.shuffle(label)

np.random.seed(200)
np.random.shuffle(data_val2) 
np.random.seed(200)
np.random.shuffle(label_val2)

np.random.seed(200)
np.random.shuffle(data2) 
np.random.seed(200)
np.random.shuffle(label2)


#%% Train!
# from keras.callbacks import TensorBoard
from datetime import datetime

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.1
    epochs_drop = 30
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    
    if lrate < 0.001:
        lrate = 0.001
    return lrate


folder = "/media/daisylab3/disk2/mPose_model/server_new_feature/"
# folder = "./cache/"
# folder = "test"
# filepath="./models/server_trainTest9_loss_ave_joint_error6/weights_best_joint_error_loss_{epoch:03d}_{val_loss:.6f}_{joint_err_avg:.6f}.hdf5"
filepath=folder + "weights_best_joint_error_loss_{epoch:03d}.hdf5"
# {val_posture_opt_loss:.5f}
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
early_stopping = EarlyStopping(monitor='val_loss', patience=30)
lrate = LearningRateScheduler(step_decay)
plot_losses = PlotLosses()

history = model.fit(
                data2,
                {"posture_opt": label2},
                batch_size=32, 
                epochs=200,
                # validation_split=0.2,
                validation_data = (data_val2, label_val2),
                shuffle=True,
                callbacks=[plot_losses, checkpoint, early_stopping, lrate]
)



# print(history.history)
np.save(folder+'history', history)

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc_cust'])
plt.plot(history.history['val_acc_cust'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# import matplotlib.pyplot as plt

# plt.plot(history.history['move_joints_loss'])
# plt.plot(history.history['val_move_joints_loss'])
# plt.plot(history.history['torso_loss'])
# plt.plot(history.history['val_torso_loss'])

# plt.plot(history.history['move_joints_accuracy'])
# plt.plot(history.history['val_move_joints_accuracy'])
# plt.plot(history.history['torso_accuracy'])
# plt.plot(history.history['val_torso_accuracy'])
# plt.title('model loss/accuracy')
# plt.ylabel('Loss')
# plt.xlabel('epoch')
# plt.legend(['move_joints_loss', 
#             'val_move_joints_loss', 
#             'torso_loss', 
#             'val_torso_loss',
#             'move_joints_accuracy',
#             'val_move_joints_accuracy',
#             'val_torso_accuracy'
#             ], 
#            loc='upper left')
# plt.savefig(filepath=folder + "weights_best_joint_error_loss_{epoch:03d}_{val_posture_opt_loss:.5f}", fname='1')
# plt.show()


#%% Pose-wise evaluation

from keras.models import load_model

import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import pdb 
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from utility_tf import PlotLosses, acc_cust, joint_error_loss

model = load_model("/home/daisylab3/Dropbox/Projects/project_mPose/code_python/trials/17pose_newfeature_songlab1_0604_finetuned2/weights_best_joint_error_loss_094.hdf5", 
                   custom_objects={'fnloss': fnloss, 'acc_cust': acc_cust})
simplifyList = [3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20];

import glob
import numpy as np
# folder =  "./data_0523_2/data_train/*"
folder1 =  "./data_0604_song_lab/data_train/*"
files = glob.glob(folder1)

results = []
seed = 12


for i, file in enumerate(files):

    data_zip = np.load(file)
    dataSub = data_zip['data']
    labelSub = data_zip['label']
    print(str(labelSub.shape[0]), ' ', file)

    dataSub = dataSub.reshape((-1, 4, 141, 141, 1))
    
    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(dataSub, labelSub, test_size=0.2, random_state=seed)
    
    # y_train_sub = y_train_sub[:, simplifyList, :]
    y_test_sub = y_test_sub[:, simplifyList, :]

    result = model.evaluate(X_test_sub, y_test_sub, batch_size=1)
    print(result)
    results.append(result)
    
print(results)


#%% Plot all
# results = results_domain1

import matplotlib.pyplot as plt

plt.bar(range(len(results)), 1000*np.array(results)[:,0])
# plt.xticks(range(7), x_label, rotation = 30, fontsize = 16)
plt.yticks(fontsize = 16)
plt.ylim([0, 300])
plt.title('Domain 1: lab environment', fontsize = 24)
plt.xlabel('Posture', fontsize = 14)
plt.ylabel('Average error for 25 joints (mm)', fontsize = 14)
plt.grid()

plt.show()

#%% Left Arm

import matplotlib.pyplot as plt
x_label = ['front_lift_45', 'front_lift_90', 'front_lift_180', 
           'side_lift_45', 'side_lift_90', 'side_lift_180']

plotList = np.array([4,2,17,3,12,1])-1
barresults = np.array(results)
plt.bar(range(len(plotList)), 1000*np.array(results)[plotList,0].T)
plt.xticks(range(len(plotList)), x_label, rotation = 30, fontsize = 16);
plt.yticks(fontsize = 16)
plt.ylim([0, 100])
plt.title('Error for all 17 postures: Left arm error', fontsize = 20)
plt.xlabel('Posture', fontsize = 14)
plt.ylabel('Average error for 25 joints (mm)', fontsize = 14)
plt.grid()

plt.show()

#%% Right Arm Eval
x_label = ['front_lift_45', 'front_lift_90', 'front_lift_180', 
           'side_lift_45', 'side_lift_90', 'side_lift_180', ]

plotList = np.array([10, 9, 15, 5, 11, 7])-1
plt.bar(range(len(plotList)), 1000*np.array(results)[plotList,0].T)
plt.xticks(range(6), x_label, rotation = 20,  fontsize = 16);
plt.yticks(fontsize = 16)
plt.ylim([0, 100])
plt.title('Error for all 17 postures: right_arm', fontsize = 20)
plt.xlabel('Posture Index', fontsize = 14)
plt.ylabel('Average error for 25 joints (mm)', fontsize = 14)
plt.grid()

plt.show()

#%% Legs, activities

x_label = ['left_leg_lift_90', 'right_leg_lift_90', 
           'random_move', 'walk', 'wave_hands', ]

plotList = np.array([8, 16, 6, 14, 13])-1

plt.bar(range(5), 1000*np.array(results)[plotList,0 ].T)
plt.xticks(range(5), x_label, rotation = 20, fontsize = 16);
plt.yticks(fontsize = 16)
plt.ylim([0, 100])
plt.title('Error for all 17 postures: Legs/Act', fontsize = 20)
plt.xlabel('Posture Index', fontsize = 14)
plt.ylabel('Average error for 25 joints (mm)', fontsize = 14)
plt.grid()

plt.show()


#%% Visualize


for i, file in enumerate(files):
    
    if 'wave_hands' in file:

        data_zip = np.load(file)
        dataSub = data_zip['data']
        labelSub = data_zip['label']
        print(str(labelSub.shape[0]), ' ', file)
    
        dataSub = dataSub.reshape((-1, 4, 141, 141, 1))
        
        X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(dataSub, labelSub, test_size=0.2, random_state=seed)
        
        y_test_sub = y_test_sub[:, simplifyList, :]
    
        
print(results)
# break


import glob
import numpy as np
import matplotlib.pyplot as plt
from utility import plotSkeleton3D, padJoints
from keras.models import load_model


# model = load_model("./cache/weights_best_joint_error_loss_054.hdf5", custom_objects={'fnloss': fnloss, 'acc_cust': acc_cust})


for i in range(len(y_test_sub)):
    
    fig = plt.figure(figsize=(16,8))


    pred_test = model.predict(X_test_sub[i:i+1])
    truth_test = y_test_sub[i:i+1]
    
    
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    plotSkeleton3D(padJoints(pred_test, needReshape = True), ax3, 'pred test', isPadded = True)
    
    ax4 = fig.add_subplot(1, 2, 2, projection='3d')
    plotSkeleton3D(padJoints(truth_test, needReshape = True), ax4, 'truth test', isPadded = True)
    

    plt.show()



#%%

import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import pdb 
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from utility_tf import PlotLosses, acc_cust, joint_error_loss

import glob
import numpy as np
import matplotlib.pyplot as plt
from utility import plotSkeleton3D, padJoints
from keras.models import load_model
simplifyList = [3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20];
from utility import checkDataTrainPhase

# model = load_model("./cache/weights_best_joint_error_loss_004.hdf5", custom_objects={'fnloss': fnloss, 'acc_cust': acc_cust})

import glob
import numpy as np
# folder =  "./data_0523_2/data_train/*"
# folder =  "./data_0613_cong_lab_1/data_train/*"

folder1 =  "./data_0604_song_lab/data_train/*"

files = glob.glob(folder1)

results = []
# for j in range(16):
seed = 120


for i, file in enumerate(files):
    
    if 'side_right_arm_lift_180' in file:
        data_zip = np.load(file)
        dataSub = data_zip['data']
        labelSub = data_zip['label']
        print(str(labelSub.shape[0]), ' ', file)
    
        dataSub = dataSub.reshape((-1, 4, 141, 141, 1))
        
        labelSub = labelSub[:, simplifyList, :]
            
        X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(dataSub, labelSub, test_size=0.2, random_state=seed)



(dataSub, labelSub) = checkDataTrainPhase(dataSub, labelSub)

print(labelSub.shape)
for i in range(len(labelSub)):
    
    fig = plt.figure(figsize=(8,8))


    # pred_test = model.predict(X_test_sub[i:i+1])
    truth_test = labelSub[i:i+1]
    
    
    # ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    # plotSkeleton3D(padJoints(pred_test, needReshape = True), ax3, 'pred test', isPadded = True)
    
    ax4 = fig.add_subplot(1, 1, 1, projection='3d')
    plotSkeleton3D(padJoints(truth_test, needReshape = True), ax4, 'truth test', isPadded = True)
    plt.title(str(i))
    

    plt.show()

#%%

