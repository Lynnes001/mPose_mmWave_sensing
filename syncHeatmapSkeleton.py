#%%

import glob, os
import numpy as np
import re, time
import pdb
from utility import centeringSkeletonLoop

# Step 2: Sync heatmaps and 3d skeletons

envFolder = "..//data//"
skipFrames = 6 # skip frist 6 frames when saving.



skeletonFolders = glob.glob(envFolder+"data_raw//*")
radarFeatureFolder = envFolder+"data_features//"
writeFolder = envFolder+"data_train//"

# make write dir
if not os.path.isdir(writeFolder):
    os.mkdir(writeFolder)

for k, skeletonFile in enumerate(skeletonFolders):
    
    data = []
    label = []
    ts = []
    
    # load heatmaps and skeletons
    skeletonFile = glob.glob(skeletonFile+'//*.npz')[0]
    filename = os.path.basename(skeletonFile)
    heatmapFile = radarFeatureFolder+filename

    skeletonSet = np.load(skeletonFile)
    skeletonFiles = skeletonSet.files
    heatmapSet = np.load(heatmapFile)
    heatmapFiles = heatmapSet.files
    
    ts_heatmap = heatmapSet['timestampList']
    ts_skeleton = skeletonSet['joints_captured_ts']
    data_heatmap = heatmapSet['heatmaps']
    data_skeleton = centeringSkeletonLoop(skeletonSet['joints_captured'])

    # Sync heatmap and skeleton.

    """
    Algorithm: Sync radar frame with skeleton frames with timestamps.
    
    Radar heatmaps are linearly distributed while the 3D skeletons from 
    Xbox Kinect are not. So based on the timestamps, we traverse all Kinect 
    frames and match each skeleton with radar frame(s).

    The output is 'heatmap_list'

    radarFrameNeeded: the # of frame(s) that each skeleton match with.
        e.g., radarFrameNeeded = 4 means 1 skeleton match with 4 radar frames.

    """
    radarFrameEndIndex = 0
    radarFrameNeeded = 4

    for i, ts_s in enumerate(ts_skeleton):
        heatmap_list = []

        for j, ts_h in enumerate(ts_heatmap):

            # In my radar chirp setting, each radar frames runs 0.017s. here i search before 2 radar frame times.
            if (j >= radarFrameEndIndex) and (ts_s - 0.033 <= ts_h <= ts_s + 0.18):  # search range in time (s).
                heatmap_list.append(data_heatmap[j])
        
                # If have enough heatmaps, save all data to list
                if len(heatmap_list)>radarFrameNeeded:
                    data.append(heatmap_list[0:radarFrameNeeded])
                    label.append(data_skeleton[i])
                    ts.append(ts_s)
                    radarFrameEndIndex = j+1
                    break

    data = np.array(data)
    label = np.array(label)
    ts = np.array(ts)
    
    np.savez(writeFolder+filename, data=data[skipFrames:], label=label[skipFrames:], ts=ts[skipFrames:])
    print(str(k+1), ' Output Match Number: ', str(label.shape[0]), ' ', filename)

#%% Visualize

