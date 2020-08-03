# -*- coding: utf-8 -*-
"""
Created on Wed May 13 21:00:00 2020

@author: Song
"""


# Read Json Output
import json, csv, datetime, os
import numpy as np
from dateutil import parser
import matplotlib.pyplot as plt
import pdb



import numpy as np
from mmwave.dataloader import DCA1000
import mmwave.dsp as dsp
# from mmwave.dsp import range_resolution
from mmwave.dsp.utils import Window


def normalize(X):
    X -= np.mean(X, axis = 1) # zero-center
    X /= np.std(X, axis = 1) # normalize
    # X = X.astype('float32')
    
    return X
    

def plotSkeleton(coordinates, c='black'):


    coordinates = coordinates.reshape(3,25)
    
    x = coordinates[0,:]
    y = coordinates[1,:]
    z = coordinates[2,:]
    bone_list = [[17, 15], [15, 0], [16, 18], [16, 0], [0, 1], # head
                 [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], # arms, hands
                 [1, 8], # body 
                 [8, 9], [9, 10], [10, 11], [11, 24], [11, 22], [22, 23], # left leg
                 [8, 12], [12, 13], [13, 14], [14, 21], [14, 19], [19, 20]] # right leg
    bone_list = np.array(bone_list)
    # number_of_postures = int(len(movement)/20)
    
    # for i in range(number_of_postures):
    # fig, ax = plt.subplots(1, figsize=(3, 8))
    plt.title('Skeleton')
    plt.scatter(x, y, s=20, c=c, edgecolors='black')
    for bone in bone_list:
        plt.plot([x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]], c)
    plt.xlim(-4, 4)
    plt.ylim(-2, 2)
    plt.gca().invert_yaxis()
    
    
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
# from pykinect2 import PyKinectV2
# 
def padJoints(joints, needReshape = True):
    # input: 13*3
    if needReshape:
        joints = joints.reshape((-1,3))
        
        
    padded_list = np.zeros((25, 3))
    simplifyList = [3, 4, 5, 6, 
                    8, 9, 10, 12, 13, 14, 
                    16, 17, 18, 20];
    padded_list[simplifyList] = joints
    
    padded_list = padded_list.reshape((1,-1))
    
    return padded_list

def plotSkeleton3D(joints, ax, title, isPadded = False, needReshape = True):
    # input: 75

    
    # JointType_SpineBase = 0
    # JointType_SpineMid = 1
    # JointType_Neck = 2
    # JointType_Head = 3
    
    # JointType_ShoulderLeft = 4
    # JointType_ElbowLeft = 5
    # JointType_WristLeft = 6
    # JointType_HandLeft = 7
    
    # JointType_ShoulderRight = 8
    # JointType_ElbowRight = 9
    # JointType_WristRight = 10
    # JointType_HandRight = 11
    
    # JointType_HipLeft = 12
    # JointType_KneeLeft = 13
    # JointType_AnkleLeft = 14
    # JointType_FootLeft = 15
    
    # JointType_HipRight = 16
    # JointType_KneeRight = 17
    # JointType_AnkleRight = 18
    # JointType_FootRight = 19
    
    # JointType_SpineShoulder = 20
    # JointType_HandTipLeft = 21
    # JointType_ThumbLeft = 22
    # JointType_HandTipRight = 23
    # JointType_ThumbRight = 24
    
    class PyKinectV2:
        JointType_SpineBase = 0
        JointType_SpineMid = 1
        JointType_Neck = 2
        JointType_Head = 3
        JointType_ShoulderLeft = 4
        JointType_ElbowLeft = 5
        JointType_WristLeft = 6
        JointType_HandLeft = 7
        JointType_ShoulderRight = 8
        JointType_ElbowRight = 9
        JointType_WristRight = 10
        JointType_HandRight = 11
        JointType_HipLeft = 12
        JointType_KneeLeft = 13
        JointType_AnkleLeft = 14
        JointType_FootLeft = 15
        JointType_HipRight = 16
        JointType_KneeRight = 17
        JointType_AnkleRight = 18
        JointType_FootRight = 19
        JointType_SpineShoulder = 20
        JointType_HandTipLeft = 21
        JointType_ThumbLeft = 22
        JointType_HandTipRight = 23
        JointType_ThumbRight = 24
        JointType_Count = 25
    
    
    if isPadded:
        
        skeleton_line = np.empty([4,5], dtype = 'int')
        # right arm
        skeleton_line[0] = [PyKinectV2.JointType_Head, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight,
                            PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight]
        # left arm
        skeleton_line[1] = [PyKinectV2.JointType_Head, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft,
                            PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft]
        # right leg
        skeleton_line[2] = [PyKinectV2.JointType_Head, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_HipRight,  
                            PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight]
        # left leg
        skeleton_line[3] = [PyKinectV2.JointType_Head, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_HipLeft, 
                            PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft]
        
    else:
        
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
        
    if needReshape:
        # joints = np.einsum('ab->ba', joints.reshape((3,25)))
        joints = joints.reshape((-1,3))
        

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
    
            ax.plot(x, z, y, linewidth=4)
            ax.scatter(x, z, y, s = 48, c='w', edgecolors='black', linewidths = 2)

    # ax.legend()
    
    ax.view_init(5, -60)
    # ax.view_init(90, -90)

    plt.draw()
    
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title(title, fontsize=32)
    
    return joints




from sklearn.preprocessing import robust_scale, StandardScaler, scale, minmax_scale
from skimage.transform import resize
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

def featureExtraction(folder, optPath):
    """
    Return opt: numFrame, 3 (heatmaps), 46*46(angle bin), nLoopsPerFrame 
    
    """
    
    startTime = ''
    with open(folder+'adc_data_Raw_LogFile.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if 'Capture start time' in str(row):
                startTimeRow = row
                break
    startTimeStr = startTimeRow[0].split(' - ')[1]
    timestamp = datetime.datetime.timestamp(parser.parse(startTimeStr))
    timestampList = []

    filename = folder + 'adc_data.bin'
    
    numFrames = 640
    numADCSamples = 512
    numTxAntennas = 2
    numRxAntennas = 4
    numLoopsPerFrame = 76
    numChirpsPerFrame = numTxAntennas * numLoopsPerFrame
    
    BINS_PROCESSED = 55
    ANGLE_RES = 1
    ANGLE_RANGE = 70
    ANGLE_BINS = (ANGLE_RANGE * 2) // ANGLE_RES + 1
    
    reso = dsp.range_resolution(numADCSamples, dig_out_sample_rate=6000, freq_slope_const=39.010)
    reso = round(reso[0], 4)
    
    # adc_data = np.fromfile(filename)
    adc_data = np.fromfile(filename, dtype=np.uint16)

    # (1) Load Data
    if adc_data.shape[0]<398458880:
        padSize = 398458880 - adc_data.shape[0];
        print("padded size: ", padSize)
        adc_data = np.pad(adc_data, padSize)[padSize:]
        
    # print(adc_data.shape)
    adc_data = adc_data.reshape(numFrames, -1)
    adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=numChirpsPerFrame,
                       num_rx=numRxAntennas, num_samples=numADCSamples)
    
    # heatmaps = np.zeros((adc_data.shape[0], 3, ANGLE_BINS, BINS_PROCESSED), dtype = 'float32')
    heatmaps = np.zeros((adc_data.shape[0], ANGLE_BINS, ANGLE_BINS), dtype = 'float32')
    
    numVirAntHori = 3
    numVirAntVer = 4
    
    # (2) Start DSP processing
    # num_vec_4va, steering_vec_4va = dsp.gen_steering_vec(40, ANGLE_RES, numVirAntVer)
    # num_vec_3va, steering_vec_3va = dsp.gen_steering_vec(70, ANGLE_RES, numVirAntHori)
    num_vec_4va, steering_vec_4va = dsp.gen_steering_vec(ANGLE_RANGE, ANGLE_RES, numVirAntVer)
    num_vec_3va, steering_vec_3va = dsp.gen_steering_vec(ANGLE_RANGE, ANGLE_RES, numVirAntHori)
    
    # (3) Process each frame
    for i, frame in enumerate(adc_data):
        timestamp+=0.033
        timestampList.append(timestamp)
        # print(frame.shape)
    
        # range_azimuth = np.zeros(((70 * 2) // 1 + 1, BINS_PROCESSED))
        # range_azimuth2 = np.zeros(((70 * 2) // 1 + 1, BINS_PROCESSED))
        # range_elevation = np.zeros(((40 * 2) // 1 + 1, BINS_PROCESSED))
        range_azimuth = np.zeros((ANGLE_BINS, BINS_PROCESSED-10))
        range_azimuth2 = np.zeros((ANGLE_BINS, BINS_PROCESSED-10))
        range_elevation = np.zeros((ANGLE_BINS, BINS_PROCESSED-10))
    
        # Range Processing
        radar_cube = dsp.range_processing(frame, window_type_1d=Window.BLACKMAN)
        # radar_cube = dsp.range_processing(frame)
    
        """ (Capon Beamformer) """
        # --- static clutter removal / normalize
        # radar_cube = normalize(radar_cube.astype("float32"))
        mean = radar_cube.mean(0)
        radar_cube = radar_cube - mean
        

        # --- capon beamforming  
        beamWeights_azimuth = np.zeros((numVirAntHori, BINS_PROCESSED), dtype=np.complex_)
        beamWeights_azimuth2 = np.zeros((numVirAntHori, BINS_PROCESSED), dtype=np.complex_)
        beamWeights_elevation = np.zeros((numVirAntVer, BINS_PROCESSED), dtype=np.complex_)
        
        # Separate TX, rx 1234, vrx1234
        radar_cube = np.concatenate((radar_cube[0::2, ...], radar_cube[1::2, ...]), axis=1)
        
                
        # Note that when replacing with generic doppler estimation functions, radarCube is interleaved and
        # has doppler at the last dimension.
        # Range bin processed
        for j in range(10,BINS_PROCESSED):
            
            # V1
            # range_azimuth[:,j], beamWeights_azimuth[:,j] = dsp.aoa_capon( radar_cube[j, 1:4 ,:], steering_vec_3va, magnitude=True)
            # range_azimuth2[:,j], beamWeights_azimuth2[:,j] = dsp.aoa_capon( radar_cube[j, 5:8 ,:], steering_vec_3va, magnitude=True)
            # range_elevation[:,j], beamWeights_elevation[:,j] = dsp.aoa_capon(
            #                                                 np.concatenate((radar_cube[:, 0::4 ,:], radar_cube[:, 1::4,:]), axis=1)[:,:, j].T, 
            #                                                 steering_vec_4va, magnitude=True)
            range_azimuth[:, j-10], beamWeights_azimuth[:, j-10] = dsp.aoa_capon( radar_cube[:, 1:4 , j-10].T, steering_vec_3va, magnitude=True)
            range_azimuth2[:, j-10], beamWeights_azimuth2[:, j-10] = dsp.aoa_capon( radar_cube[:, 5:8 , j-10].T, steering_vec_3va, magnitude=True)
            range_elevation[:, j-10], beamWeights_elevation[:, j-10] = dsp.aoa_capon( radar_cube[:, [0,4,1,5] , j-10].T,  steering_vec_4va, magnitude=True)
            
            
            # # V2
            # range_azimuth[:,j], beamWeights_azimuth[:,j] = dsp.aoa_capon( radar_cube[:, 0:3 , j].T, steering_vec_3va, magnitude=True)
            # range_azimuth2[:,j], beamWeights_azimuth2[:,j] = dsp.aoa_capon( radar_cube[:, 4:7 ,j].T, steering_vec_3va, magnitude=True)
            # range_elevation[:,j], beamWeights_elevation[:,j] = dsp.aoa_capon( radar_cube[:, [2,6,3,7], j].T, steering_vec_4va, magnitude=True)
            
        
        # print('AFTER \nmean: ', np.mean(range_azimuth), np.mean(range_azimuth2), np.mean(range_elevation) )
        # print('std: ', np.std(range_azimuth), np.std(range_azimuth2), np.std(range_elevation) )
        
        # print(np.mean(np.mean(range_azimuth,axis = 1)), np.std(range_azimuth))
        # minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
        # range_azimuth = minmax_scale(range_azimuth, axis = 1, copy = False)
        # range_azimuth2 = minmax_scale(range_azimuth2, axis = 1, copy = False)
        # range_elevation = minmax_scale(range_elevation, axis = 1, copy = False)
        
        # normalize

        prescale_factor = 10000000
        range_azimuth = scale(range_azimuth/prescale_factor, axis = 1)
        range_azimuth2 = scale(range_azimuth2/prescale_factor, axis = 1) 
        range_elevation = scale(range_elevation/prescale_factor, axis = 1) 
            
        # if i == 22:
        #     break
        # plt.imshow(range_azimuth)
        # plt.show()
        # plt.pause(0.5)
        
        range_azimuth = resize(range_azimuth, (ANGLE_BINS, ANGLE_BINS/3))
        range_azimuth2 = resize(range_azimuth2, (ANGLE_BINS, ANGLE_BINS/3))
        range_elevation = resize(range_elevation, (ANGLE_BINS, ANGLE_BINS/3))
        
        heatmaps[i,:,:] = np.concatenate((range_azimuth, range_azimuth2, range_elevation), axis = 1)
        


        # print("Maxs: ", range_azimuth.max(), " === ", range_azimuth2.max(), " === ", range_elevation.max(), " === ")
        # print("Mins: ", range_azimuth.min(), " === ", range_azimuth2.min(), " === ", range_elevation.min(), " === ")
        # if i == 100:
        #     break
        
        # if 11< i <=21:
        #     fig = plt.figure(figsize=(3,3),frameon=False)
        #     plt.imshow(heatmaps[i,:,:])
        #     plt.title('input heatmap')
        #     plt.savefig('./intermediate/fig'+str(figid)+'_'+str(i))

        #     plt.show()
            
        #     plt.close()
        #     if i == 21:
        #         break
                
    # opt
    heatmaps=heatmaps.astype('float32')
    
    if np.isnan(heatmaps).sum() + np.isinf(heatmaps).sum() > 0:
        print('dtype: ', heatmaps.dtype, ', has NAN or INF error')
    else:
        print('dtype: ', heatmaps.dtype)


    if optPath != '':
        # if not os.path.isdir(optPath + '//'):
        #     os.mkdir(optPath + '//')
        np.savez(optPath+'.npz', heatmaps=heatmaps, timestampList=timestampList)
        
    return heatmaps, np.array(timestampList), reso


def checkData(data, label, ts):
    data_check = np.isinf(data)
    if np.sum(data_check):
        errListData = np.argwhere(data_check==True)
        print('Bad data Error Data INF', errListData )
        errListUnqData = np.unique(errListData[:,0])
        
        data = np.delete(data, errListUnqData, axis = 0)
        label = np.delete(label, errListUnqData, axis = 0)
        ts = np.delete(ts, errListUnqData, axis = 0)
        
    data_check = np.isnan(data)
    if np.sum(data_check):
        errListData = np.argwhere(data_check==True)
        print('Bad data Error Data Nan', errListData )
        errListUnqData = np.unique(errListData[:,0])
        
        data = np.delete(data, errListUnqData, axis = 0)
        label = np.delete(label, errListUnqData, axis = 0)
        ts = np.delete(ts, errListUnqData, axis = 0)

    label_check = np.isinf(label)
    if np.sum(label_check):
        errLostLabel = np.argwhere(label_check==True)
        print('Bad data Error Label INF', errLostLabel )
        errListUnqLabel = np.unique(errLostLabel[:,0])

        data = np.delete(data, errListUnqLabel, axis = 0)
        label = np.delete(label, errListUnqLabel, axis = 0)
        ts = np.delete(ts, errListUnqLabel, axis = 0)

    label_check = np.isnan(label)
    if np.sum(label_check):
        errLostLabel = np.argwhere(label_check==True)
        print('Bad data Error Label Nan', errLostLabel )
        errListUnqLabel = np.unique(errLostLabel[:,0])

        data = np.delete(data, errListUnqLabel, axis = 0)
        label = np.delete(label, errListUnqLabel, axis = 0)
        ts = np.delete(ts, errListUnqLabel, axis = 0)

    return data, label, ts

def centeringSkeleton(sample):
    ref_x = sample[0, 0]; ref_z = sample[0, 1]; ref_y = sample[0, 2];
    for j in sample:
        j[0] -= ref_x
        j[1] -= ref_z
        j[2] -= ref_y
    
    return sample

def centeringSkeletonLoop(skeletons):
    
    for sample in skeletons:
        # assert sample.shape == 75
        sample = centeringSkeleton(sample)
        
    return skeletons


def checkDataTrainPhase(data, label):
    data_check = np.isinf(data)
    if np.sum(data_check):
        errListData = np.argwhere(data_check==True)
        errListUnqData = np.unique(errListData[:,0])
        print('Bad data Error Data INF', errListUnqData )

        data = np.delete(data, errListUnqData, axis = 0)
        label = np.delete(label, errListUnqData, axis = 0)
        
    data_check = np.isnan(data)
    if np.sum(data_check):
        errListData = np.argwhere(data_check==True)
        errListUnqData = np.unique(errListData[:,0])
        print('Bad data Error Data NAN', errListUnqData )

        data = np.delete(data, errListUnqData, axis = 0)
        label = np.delete(label, errListUnqData, axis = 0)
        
    label_check = np.isinf(label)
    if np.sum(label_check):
        errListLabel = np.argwhere(label_check==True)
        errListUnqLabel = np.unique(errListLabel[:,0])
        print('Bad data Error Label INF', errListUnqLabel )

        data = np.delete(data, errListUnqLabel, axis = 0)
        label = np.delete(label, errListUnqLabel, axis = 0)
        
    label_check = np.isnan(label)
    if np.sum(label_check):
        errListLabel = np.argwhere(label_check==True)
        errListUnqLabel = np.unique(errListLabel[:,0])
        print('Bad data Error Label NAN', errListUnqLabel )

        data = np.delete(data, errListUnqLabel, axis = 0)
        label = np.delete(label, errListUnqLabel, axis = 0)
    
    return data, label
