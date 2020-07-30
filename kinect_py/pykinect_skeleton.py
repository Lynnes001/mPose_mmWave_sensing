# -*- coding: utf-8 -*-
"""
Created on Fri May 22 21:30:40 2020

@author: Song
"""

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import time;


""" 
Config:
    numFrame: # of skeleton frames we want to capture.
    filename: dump npy file name.
    folder: folder to save the skeleton file.

Saved file format:
    [folder/filename].npz
    |
    |--joints_captured_ts: timestamp of the captured skeleton. (1 x numFrame)
    |
    |--joints_captured: detected skeleton joints coordinates. (25 x numFrame)

"""


numFrame = 800
filename = 'act_wave_hands'
folder = 'data_0704_song_lab_1_stationary/'



# =========== End Config ===============

# Kinect runtime object, we want only color and body frames 
# kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Body)

# here we will store skeleton data 
bodies = None

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')

skeleton_counter = 0;
joints_bank = []
joints_time = []
color_frames = []
joints= []

while skeleton_counter < numFrame:
    # --- Cool! We have a body frame, so can get skeletons
    if kinect.has_new_body_frame():
        bodies = kinect.get_last_body_frame()
    
        # --- draw skeletons to _frame_surface
        if bodies is not None:
            # i=0;
            for i in range(0, kinect.max_body_count):
                body = bodies.bodies[i]
                if body.is_tracked:
                    joints = body.joints
                    joints_bank.append(joints)
                    joints_time.append(time.time())
                    # color_frames.append(frame)
                    # draw_body_kinect(joints,skeleton_line,ax,plt)
                    skeleton_counter+=1
                    print('Skeleton Detected!!! Counter: '+str(skeleton_counter))

kinect.close()

# Process joints_bank
joints_captured = np.empty([len(joints_bank), 25, 3], dtype = 'float')

for i, joints in enumerate(joints_bank):
    temp = []
    for j in range(25):
        joints_captured[i,j,:] = np.array([joints[j].Position.x, joints[j].Position.y, joints[j].Position.z])
        
joints_captured_ts = np.array(joints_time)

np.savez(folder+'data_raw/'+filename+'.npz', joints_captured_ts=joints_captured_ts, joints_captured=joints_captured)
print('File Saved!')
 
 
#%% the following codes are supposed to plot out 3D posture skeleton plot, but are not verified.


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')


x=[]
y=[]
z=[]

for row in skeleton_line:
    x=[]
    y=[]
    z=[]

    for joint_id in row:
        x.append(joints[joint_id].Position.x)
        y.append(joints[joint_id].Position.y)
        z.append(joints[joint_id].Position.z)

        ax.plot(x, y, z, label='parametric curve')



# ax.legend()

ax.view_init(-100,  90)
plt.draw()


#%%

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import time;
from pykinect2 import PyKinectV2

skeleton_line = np.empty([7,5], dtype = 'int')
# main body
skeleton_line[0] = [PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder, 
                    PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase]
# right leg
skeleton_line[1] = [PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight, 
                    PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight]
# left leg
skeleton_line[2] = [PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft, 
                    PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft]
# right arm
skeleton_line[3] = [PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight,
                    PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight]
# left arm
skeleton_line[4] = [PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft,
                    PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft]
# right hand
skeleton_line[5] = [PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight,
                    PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight]
# left hand
skeleton_line[6] = [PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft,
                    PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft]

def draw_body(joints, skeleton_line, ax, plt):

    for row in skeleton_line:
        x=[]
        y=[]
        z=[]
    
        for joint_id in row:
            # x.append(joints[joint_id].Position.x)
            # y.append(joints[joint_id].Position.y)
            # z.append(joints[joint_id].Position.z)
            x.append(joints[joint_id,0])
            y.append(joints[joint_id,1])
            z.append(joints[joint_id,2])
    
            ax.plot(x, y, z, label='parametric curve')
    
    
    
    # ax.legend()
    
    ax.view_init(90,  -90)
    plt.draw()

def draw_body_kinect(joints, skeleton_line, ax, plt):
    for row in skeleton_line:
        x=[]
        y=[]
        z=[]
        print(row.shape)
        for joint_id in row:
            
            x.append(joints[joint_id].Position.x)
            y.append(joints[joint_id].Position.y)
            z.append(joints[joint_id].Position.z)
            ax.plot(x, y, z, label='parametric curve')    
    # ax.legend()
    
    ax.view_init(90,  -90)
    plt.draw()


for i, joints_sample in enumerate(label):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    draw_body(joints_sample, skeleton_line, ax, plt)
    ax.set_zlim(0,10)
    plt.title(str(i))
    plt.show()
    plt.pause(.1)
    # break
#%% plot the kinect output

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')


for joints_sample in joints_bank[:1:]:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    draw_body_kinect(joints_sample, skeleton_line, ax, plt)
    ax.set_zlim(0,10)
    plt.show()
    plt.pause(.1)
    # break


#%% plot mpose captured

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')

for ts, joints_sample in zip(joints_captured_ts, joints_captured):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    draw_body(joints_sample, skeleton_line, ax, plt)
    ax.set_zlim(0,10)
    plt.title(ts)
    plt.show()
    plt.pause(.1)
    # break
