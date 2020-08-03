# -*- coding: utf-8 -*-
"""
Created on Tue May 26 00:11:18 2020

@author: Song
"""

#%% Step 1. Extract heatmaps from ADC data

envFolder = "..//data//"

from utility import featureExtraction
import numpy as np
import glob, os


if not os.path.isdir(envFolder+'data_features//'):
    os.mkdir(envFolder+'data_features//')

data_folders = glob.glob(envFolder+"data_raw//*")
for i, folder in enumerate(data_folders):
    if i > -1:
        filename = folder.split('/')[5]
        folder = folder+'//'
        print(filename, ' ', str(i+1)+'/'+str(len(data_folders)))
        [heatmaps, timestampList, rangeReso] = featureExtraction(folder, envFolder+'data_features//'+filename, i)

    
#%% Step 2: Sync heatmaps and 3d skeletons

import glob, os
import numpy as np
import re, time
import pdb
from utility import centeringSkeletonLoop

# envFolder = "data_0604_song_lab"

skeletonFolders = glob.glob(".//"+envFolder+"//data_raw//*")
radarFeatureFolder = ".//"+envFolder+"//data_features//"
writeFolder = ".//"+envFolder+"//data_train//"


if not os.path.isdir(writeFolder):
    os.mkdir(writeFolder)

for k, skeletonFile in enumerate(skeletonFolders):
    
    data = []
    label = []
    
    ts = []
    # data = np.ones([])
    
    skeletonFile = glob.glob(skeletonFile+'//*.npz')[0]
    filename = skeletonFile.split('/')[-1].split('.')[0]
    heatmapFile = radarFeatureFolder+filename+'.npz'

    skeletonSet = np.load(skeletonFile)
    skeletonFiles = skeletonSet.files
    heatmapSet = np.load(heatmapFile)
    heatmapFiles = heatmapSet.files
    
    ts_heatmap = heatmapSet['timestampList']
    ts_skeleton = skeletonSet['joints_captured_ts']
    data_heatmap = heatmapSet['heatmaps']
    data_skeleton = centeringSkeletonLoop(skeletonSet['joints_captured'])
    # pdb.set_trace()

    counter_s = 0

    # Sync heatmap and skeleton
    
    radarFrameEndTime = 0
    radarFrameEndIndex = 0

    for i, ts_s in enumerate(ts_skeleton):
        heatmap_list = []
        # print('heatmaplist clean')

        for j, ts_h in enumerate(ts_heatmap):
            if (j >= radarFrameEndIndex) and (ts_s - 0.033 <= ts_h <= ts_s + 0.18):
                # print(i, '-', ts_s, ' ===', j, '-', ts_h)

                heatmap_list.append(data_heatmap[j])
                
                # skeleton_list.append(data_skeleton[i])
                # print(data_skeleton[i].shape)
                # print(data_heatmap[j].shape)

                # If have enough heatmap data
                if len(heatmap_list)>3:
                    data.append(heatmap_list[0:4])
                    # print('heatmaplen:', len(heatmap_list), '  Breaked')
                    label.append(data_skeleton[i])
                    ts.append(ts_s)
                    counter_s += 1
                    radarFrameEndIndex = j+1
                    # print(radarFrameEndIndex)
                    # print('===========')
                    # pdb.set_trace()
                    break
        # pdb.set_trace()
    # break
    data = np.array(data)
    label = np.array(label)
    ts = np.array(ts)
    
    # pdb.set_trace()
    np.savez(writeFolder+filename+'.npz', data=data[6:], label=label[6:], ts=ts[6:])
    print(str(k+1)+'-'+str(label.shape[0])+skeletonFile)
    print(heatmapFile)

#%%

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(ts_skeleton)
plt.plot(ts_heatmap)

plt.legend(['skeleton','heatmap'])

plt.show()
# data_skeleton[100]


#%% Extract feature framebyframe


# import glob
# import numpy as np
# import re, time
# import pdb 
# from utility import centeringSkeletonLoop

# envFolder = "data_0604_song_lab"

# skeletonFolders = glob.glob("./"+envFolder+"/data_raw/*")
# radarFeatureFolder = "./"+envFolder+"/data_features/";
# writeFolder = "./"+envFolder+"/data_train/"

# for k, skeletonFile in enumerate(skeletonFolders):
    
#     data = []
#     label = []
    
#     ts = []
#     # data = np.ones([])
    
#     skeletonFile = glob.glob(skeletonFile+'/*.npz')[0];
#     filename = skeletonFile.split('/')[-1].split('.')[0];
#     heatmapFile = radarFeatureFolder+filename+'.npz'

#     skeletonSet = np.load(skeletonFile)
#     skeletonFiles = skeletonSet.files
#     heatmapSet = np.load(heatmapFile)
#     heatmapFiles = heatmapSet.files
    
#     ts_heatmap = heatmapSet['timestampList']
#     ts_skeleton = skeletonSet['joints_captured_ts']
#     data_heatmap = heatmapSet['heatmaps']
#     data_skeleton = centeringSkeletonLoop(skeletonSet['joints_captured'])
#     # pdb.set_trace()

#     counter_s = 0;

#     # Sync heatmap and skeleton
    
#     radarFrameEndTime = 0;
#     radarFrameEndIndex = 0;

#     for i, ts_s in enumerate(ts_skeleton):
#         heatmap_list = []

#         for j, ts_h in enumerate(ts_heatmap):
#             # print(ts_h)
#             # pdb.set_trace()
#             if (j >= radarFrameEndIndex) and (ts_s - 0.09 <= ts_h <= ts_s + 0.18):
#                 # print(ts_h)

#                 heatmap_list.append(data_heatmap[j])
                
#                 # skeleton_list.append(data_skeleton[i])
#                 # print(data_skeleton[i].shape)
#                 # print(data_heatmap[j].shape)

#                 # If have enough heatmap data (5 frames)
#                 if len(heatmap_list)>3:
#                     data.append(heatmap_list[0:4])
#                     label.append(data_skeleton[i])
#                     ts.append(ts_s)
#                     counter_s += 1
#                     radarFrameEndIndex = j+1;
#                     # print(radarFrameEndIndex)
#                     # print('===========')
#                     break
#         # pdb.set_trace()
#     # break
#     data = np.array(data)
#     label = np.array(label)
#     ts = np.array(ts)
    
#     # pdb.set_trace()
#     np.savez(writeFolder+filename+'.npz', data=data, label=label, ts=ts)
#     print(str(k+1)+'-'+str(label.shape[0])+skeletonFile)
#     print(heatmapFile)
